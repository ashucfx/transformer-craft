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


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_dimension = config.embedding_dimension // config.attention_heads
        self.query = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.key_value = nn.Linear(config.embedding_dimension, config.embedding_dimension * 2)
        self.output = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.dropout = config.dropout

    def forward(
        self, queries: Tensor, keys: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        batch, query_length, dimension = queries.shape
        key_length = keys.shape[1]
        query = self.query(queries)
        key, value = self.key_value(keys).chunk(2, dim=-1)
        query_shape = (batch, query_length, self.heads, self.head_dimension)
        key_shape = (batch, key_length, self.heads, self.head_dimension)
        query = query.view(query_shape).transpose(1, 2)
        key = key.view(key_shape).transpose(1, 2)
        value = value.view(key_shape).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
            mask = torch.where(
                mask,
                torch.zeros((), device=queries.device, dtype=query.dtype),
                torch.full((), float("-inf"), device=queries.device, dtype=query.dtype),
            )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, query_length, dimension))


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


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(config.embedding_dimension)
        self.self_attention = MultiHeadSelfAttention(config)
        self.cross_attention_norm = nn.LayerNorm(config.embedding_dimension)
        self.cross_attention = MultiHeadCrossAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.feed_forward_dimension),
            nn.GELU(),
            nn.Linear(config.feed_forward_dimension, config.embedding_dimension),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        decoder_attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> Tensor:
        normalized = self.self_attention_norm(x)
        x = x + self.self_attention(normalized, decoder_attention_mask, causal=True)
        x = x + self.cross_attention(
            self.cross_attention_norm(x), encoder_output, encoder_attention_mask
        )
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


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TransformerBackbone(config)
        self.decoder_token_embedding = nn.Embedding(config.vocab_size, config.embedding_dimension)
        self.decoder_position_embedding = PositionalEncoding(config)
        self.decoder_dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.layers)
        )
        self.decoder_norm = nn.LayerNorm(config.embedding_dimension)
        self.lm_head = nn.Linear(config.embedding_dimension, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder_token_embedding.weight

    def forward(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor,
        encoder_attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if encoder_input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("encoder sequence exceeds max_sequence_length")
        if decoder_input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("decoder sequence exceeds max_sequence_length")
        encoder_output = self.encoder(
            encoder_input_ids, encoder_attention_mask, causal=False
        )
        decoder = self.decoder_token_embedding(decoder_input_ids)
        decoder = decoder + self.decoder_position_embedding(decoder_input_ids)
        decoder = self.decoder_dropout(decoder)
        for block in self.decoder_blocks:
            decoder = block(
                decoder,
                encoder_output,
                decoder_attention_mask,
                encoder_attention_mask,
            )
        logits = self.lm_head(self.decoder_norm(decoder))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return result
