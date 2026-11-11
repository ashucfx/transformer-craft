import torch

from transformer_craft.config import ModelConfig
from transformer_craft.modeling import (
    CausalTransformer,
    EncoderDecoderTransformer,
    MaskedTransformer,
)


def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=260,
        max_sequence_length=8,
        embedding_dimension=16,
        attention_heads=4,
        layers=2,
        feed_forward_dimension=32,
        dropout=0.0,
    )


def test_causal_model_shapes_and_loss() -> None:
    model = CausalTransformer(config())
    input_ids = torch.randint(0, 260, (2, 8))
    result = model(input_ids, labels=input_ids)
    assert result["logits"].shape == (2, 8, 260)
    assert result["loss"].ndim == 0


def test_masked_model_shapes_and_loss() -> None:
    model = MaskedTransformer(config())
    input_ids = torch.randint(0, 260, (2, 8))
    labels = torch.full_like(input_ids, -100)
    labels[:, 0] = input_ids[:, 0]
    result = model(input_ids, labels=labels)
    assert result["logits"].shape == (2, 8, 260)
    assert torch.isfinite(result["loss"])


def test_causal_attention_does_not_use_future_tokens() -> None:
    model = CausalTransformer(config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = first.clone()
    second[0, -1] = 9
    with torch.no_grad():
        first_logits = model(first)["logits"]
        second_logits = model(second)["logits"]
    assert torch.allclose(first_logits[:, :-1], second_logits[:, :-1], atol=1e-5)


def test_encoder_decoder_shapes_and_teacher_forced_loss() -> None:
    model = EncoderDecoderTransformer(config())
    encoder_input_ids = torch.randint(0, 260, (2, 6))
    decoder_input_ids = torch.randint(0, 260, (2, 5))
    labels = decoder_input_ids.clone()
    result = model(
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=torch.ones(2, 6, dtype=torch.bool),
        decoder_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        labels=labels,
    )
    assert result["logits"].shape == (2, 5, 260)
    assert result["loss"].ndim == 0


def test_encoder_decoder_uses_encoder_context() -> None:
    model = EncoderDecoderTransformer(config()).eval()
    encoder_input_ids = torch.ones(1, 4, dtype=torch.long)
    changed_encoder_input_ids = encoder_input_ids.clone()
    changed_encoder_input_ids[0, 0] = 2
    decoder_input_ids = torch.ones(1, 3, dtype=torch.long)
    with torch.no_grad():
        first = model(encoder_input_ids, decoder_input_ids)["logits"]
        second = model(changed_encoder_input_ids, decoder_input_ids)["logits"]
    assert not torch.allclose(first, second)
