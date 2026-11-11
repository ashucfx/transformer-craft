import torch

from transformer_craft.config import ModelConfig
from transformer_craft.inference import evaluate_causal, generate
from transformer_craft.modeling import CausalTransformer


def test_generation_preserves_prompt_and_length() -> None:
    config = ModelConfig(
        max_sequence_length=8,
        embedding_dimension=8,
        attention_heads=2,
        layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    model = CausalTransformer(config)
    prompt = torch.tensor([[1, 2]])
    result = generate(model, prompt, max_new_tokens=3, temperature=1.0)
    assert result.shape == (1, 5)
    assert torch.equal(result[:, :2], prompt)


def test_causal_evaluation_returns_loss_and_perplexity() -> None:
    config = ModelConfig(
        max_sequence_length=4,
        embedding_dimension=8,
        attention_heads=2,
        layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    model = CausalTransformer(config)
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[True, True, True, True]]),
        }
    ]
    metrics = evaluate_causal(model, batches)
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 1
