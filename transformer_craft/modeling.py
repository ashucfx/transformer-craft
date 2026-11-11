import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.positional_encoding == "learned":
            self.values: nn.Module = nn.Embedding(
                config.max_sequence_length, config.embedding_dimension
            )
        else:
            positions = torch.arange(config.max_sequence_length).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, config.embedding_dimension, 2)
                * (-math.log(10000.0) / config.embedding_dimension)
            )
            values = torch.zeros(config.max_sequence_length, config.embedding_dimension)
            values[:, 0::2] = torch.sin(positions * frequencies)
            values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
            self.register_buffer("sinusoidal", values, persistent=False)
            self.values = nn.Identity()

    def forward(self, input_ids: Tensor) -> Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        if isinstance(self.values, nn.Embedding):
            return self.values(positions).unsqueeze(0)
        return self.sinusoidal[positions].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dimension = config.embedding_dimension // config.attention_heads
        self.query_key_value = nn.Linear(config.embedding_dimension, config.embedding_dimension * 3)
        self.output = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.dropout = config.dropout

    def forward(
        self, x: Tensor, attention_mask: Tensor | None = None, causal: bool = False
    ) -> Tensor:
        batch, sequence, dimension = x.shape
        query, key, value = self.query_key_value(x).chunk(3, dim=-1)
        shape = (batch, sequence, self.heads, self.head_dimension)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        combined_mask = None
        if causal:
            combined_mask = torch.ones(sequence, sequence, device=x.device, dtype=torch.bool).tril()
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].to(torch.bool)
            combined_mask = padding_mask if combined_mask is None else combined_mask & padding_mask
        if combined_mask is not None:
            combined_mask = torch.where(
                combined_mask,
                torch.zeros((), device=x.device, dtype=query.dtype),
                torch.full((), float("-inf"), device=x.device, dtype=query.dtype),
            )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, sequence, dimension))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dimension)
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.feed_forward_dimension),
            nn.GELU(),
            nn.Linear(config.feed_forward_dimension, config.embedding_dimension),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None, causal: bool) -> Tensor:
        x = x + self.attention(self.attention_norm(x), attention_mask, causal)
        return x + self.feed_forward(self.feed_forward_norm(x))


class TransformerBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dimension)
        self.position_embedding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.embedding_dimension)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, causal: bool = False
    ) -> Tensor:
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        x = self.token_embedding(input_ids) + self.position_embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask, causal)
        return self.norm(x)


class CausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(config)
        self.lm_head = nn.Linear(config.embedding_dimension, config.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.backbone(input_ids, attention_mask, causal=True)
        logits = self.lm_head(hidden)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


class MaskedTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(config)
        self.lm_head = nn.Linear(config.embedding_dimension, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        hidden = self.backbone(input_ids, attention_mask, causal=False)
        logits = self.lm_head(hidden)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return result
