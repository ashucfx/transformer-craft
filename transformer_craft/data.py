from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpecialTokens:
    pad: int = 256
    mask: int = 257
    bos: int = 258
    eos: int = 259


class ByteTokenizer:
    def __init__(self) -> None:
        self.special = SpecialTokens()
        self.vocab_size = 260

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.special.bos)
        if add_eos:
            tokens.append(self.special.eos)
        return tokens

    def decode(self, tokens: list[int] | Tensor) -> str:
        values = tokens.tolist() if isinstance(tokens, Tensor) else tokens
        return bytes(token for token in values if 0 <= token < 256).decode(
            "utf-8", errors="replace"
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text('{"type":"byte","vocab_size":260}\n', encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return cls()


def discover_code_files(
    roots: list[str | Path],
    extensions: set[str] | None = None,
) -> list[Path]:
    allowed = {
        extension.lower()
        for extension in (
            extensions
            or {".py", ".js", ".ts", ".java", ".cpp", ".c", ".rs", ".go"}
        )
    }
    files: set[Path] = set()
    for root in roots:
        path = Path(root)
        if path.is_file():
            if not allowed or path.suffix.lower() in allowed:
                files.add(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for candidate in path.rglob("*"):
            if candidate.is_file() and (not allowed or candidate.suffix.lower() in allowed):
                files.add(candidate.resolve())
    return sorted(files, key=lambda path: str(path).lower())


def load_corpus_text(paths: list[str | Path], extensions: set[str] | None = None) -> str:
    files = discover_code_files(paths, extensions)
    if not files:
        raise ValueError("no matching corpus files found")
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


class CodeDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, tokens: list[int], sequence_length: int, pad_id: int = 256) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if not tokens:
            raise ValueError("tokens must not be empty")
        self.sequence_length = sequence_length
        self.pad_id = pad_id
        self.windows = [
            tokens[start : start + sequence_length]
            for start in range(0, len(tokens), sequence_length)
        ]

    @classmethod
    def from_text(
        cls, path: str | Path, tokenizer: ByteTokenizer, sequence_length: int
    ) -> "CodeDataset":
        text = Path(path).read_text(encoding="utf-8")
        return cls(tokenizer.encode(text), sequence_length, tokenizer.special.pad)

    def __len__(self) -> int:
        return len(self.windows)

    def split(
        self, validation_fraction: float, seed: int = 42
    ) -> tuple["CodeDataset", "CodeDataset"]:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        item_count = len(self.windows) if hasattr(self, "windows") else len(self.pairs)
        if item_count < 2:
            raise ValueError("at least two windows are required for a split")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(item_count, generator=generator).tolist()
        validation_size = max(1, round(item_count * validation_fraction))
        validation_indices = permutation[:validation_size]
        training_indices = permutation[validation_size:]
        if not training_indices:
            training_indices, validation_indices = permutation[:-1], permutation[-1:]
        training = copy(self)
        validation = copy(self)
        if hasattr(self, "windows"):
            training.windows = [self.windows[index] for index in training_indices]
            validation.windows = [self.windows[index] for index in validation_indices]
        elif hasattr(self, "pairs"):
            training.pairs = [self.pairs[index] for index in training_indices]
            validation.pairs = [self.pairs[index] for index in validation_indices]
        else:
            raise TypeError(f"unsupported dataset type: {type(self).__name__}")
        return training, validation

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        window = self.windows[index]
        input_ids = window[: self.sequence_length]
        attention_mask = [1] * len(input_ids)
        input_ids = input_ids + [self.pad_id] * (self.sequence_length - len(input_ids))
        attention_mask += [0] * (self.sequence_length - len(attention_mask))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        }


class CausalDataset(CodeDataset):
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = super().__getitem__(index)
        item["labels"] = item["input_ids"].clone()
        item["labels"][~item["attention_mask"]] = -100
        return item


class MaskedDataset(CodeDataset):
    def __init__(
        self,
        tokens: list[int],
        sequence_length: int,
        mask_id: int = 257,
        pad_id: int = 256,
        probability: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(tokens, sequence_length, pad_id)
        if not 0 < probability <= 1:
            raise ValueError("probability must be in (0, 1]")
        self.mask_id = mask_id
        self.probability = probability
        self.seed = seed

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = super().__getitem__(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        candidates = item["attention_mask"]
        selected = torch.rand(self.sequence_length, generator=generator) < self.probability
        selected &= candidates
        labels = item["input_ids"].clone()
        labels[~selected] = -100
        item["input_ids"][selected] = self.mask_id
        item["labels"] = labels
        return item


class SequencePairDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        pairs: list[tuple[list[int], list[int]]],
        sequence_length: int,
        *,
        pad_id: int = 256,
        bos_id: int = 258,
        eos_id: int = 259,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if not pairs:
            raise ValueError("pairs must not be empty")
        self.pairs = pairs
        self.sequence_length = sequence_length
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    @classmethod
    def from_text_pairs(
        cls,
        pairs: list[tuple[str, str]],
        tokenizer: ByteTokenizer,
        sequence_length: int,
    ) -> "SequencePairDataset":
        encoded = [(tokenizer.encode(source), tokenizer.encode(target)) for source, target in pairs]
        return cls(
            encoded,
            sequence_length,
            pad_id=tokenizer.special.pad,
            bos_id=tokenizer.special.bos,
            eos_id=tokenizer.special.eos,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def split(
        self, validation_fraction: float, seed: int = 42
    ) -> tuple["SequencePairDataset", "SequencePairDataset"]:
        return CodeDataset.split(self, validation_fraction, seed)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        source, target = self.pairs[index]
        source = source[: self.sequence_length]
        target = target[: self.sequence_length - 1]
        source_ids = source + [self.pad_id] * (self.sequence_length - len(source))
        target_input = [self.bos_id] + target
        target_labels = target + [self.eos_id]
        target_input += [self.pad_id] * (self.sequence_length - len(target_input))
        target_labels += [-100] * (self.sequence_length - len(target_labels))
        return {
            "encoder_input_ids": torch.tensor(source_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(target_input, dtype=torch.long),
            "encoder_attention_mask": torch.tensor(
                [token != self.pad_id for token in source_ids], dtype=torch.bool
            ),
            "decoder_attention_mask": torch.tensor(
                [token != self.pad_id for token in target_input], dtype=torch.bool
            ),
            "labels": torch.tensor(target_labels, dtype=torch.long),
        }


def stream_text_files(paths: list[str | Path], chunk_size: int = 1 << 20) -> Iterator[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            while chunk := handle.read(chunk_size):
                yield chunk
