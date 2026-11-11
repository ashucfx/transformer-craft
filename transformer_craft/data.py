from collections.abc import Iterator
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
        if len(self.windows) < 2:
            raise ValueError("at least two windows are required for a split")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(self.windows), generator=generator).tolist()
        validation_size = max(1, round(len(self.windows) * validation_fraction))
        validation_indices = permutation[:validation_size]
        training_indices = permutation[validation_size:]
        if not training_indices:
            training_indices, validation_indices = permutation[:-1], permutation[-1:]
        training = type(self).__new__(type(self))
        validation = type(self).__new__(type(self))
        for dataset, indices in ((training, training_indices), (validation, validation_indices)):
            dataset.sequence_length = self.sequence_length
            dataset.pad_id = self.pad_id
            dataset.windows = [self.windows[index] for index in indices]
        if isinstance(self, MaskedDataset):
            for dataset in (training, validation):
                dataset.mask_id = self.mask_id
                dataset.probability = self.probability
                dataset.seed = self.seed
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
