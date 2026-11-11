from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    max_sequence_length: int = 256
    embedding_dimension: int = 256
    attention_heads: int = 8
    layers: int = 6
    feed_forward_dimension: int = 1024
    dropout: float = 0.1
    positional_encoding: str = "learned"

    def __post_init__(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if self.embedding_dimension < 1 or self.embedding_dimension % self.attention_heads:
            raise ValueError(
                "embedding_dimension must be positive and divisible by attention_heads"
            )
        if self.attention_heads < 1 or self.layers < 1 or self.feed_forward_dimension < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.positional_encoding not in {"learned", "sinusoidal"}:
            raise ValueError("positional_encoding must be learned or sinusoidal")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.warmup_steps < 0 or self.gradient_accumulation_steps < 1:
            raise ValueError("warmup_steps must be non-negative and accumulation steps positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
