from transformer_craft.benchmark import benchmark, count_parameters
from transformer_craft.config import ModelConfig
from transformer_craft.modeling import CausalTransformer


def test_benchmark_reports_runtime_metrics() -> None:
    config = ModelConfig(
        max_sequence_length=4,
        embedding_dimension=8,
        attention_heads=2,
        layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    model = CausalTransformer(config)
    result = benchmark(model, "causal", config, batch_size=1, steps=1, device="cpu")
    assert result["parameters"] == count_parameters(model)
    assert result["tokens_per_second"] > 0
    assert result["milliseconds_per_step"] > 0
