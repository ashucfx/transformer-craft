from pathlib import Path

from torch.utils.data import DataLoader

from transformer_craft.config import ModelConfig, TrainingConfig
from transformer_craft.data import CausalDataset
from transformer_craft.modeling import CausalTransformer
from transformer_craft.training import load_checkpoint, train


def test_training_writes_loadable_checkpoint(tmp_path: Path) -> None:
    model_config = ModelConfig(
        max_sequence_length=4,
        embedding_dimension=8,
        attention_heads=2,
        layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    training_config = TrainingConfig(max_steps=2, batch_size=1, device="cpu", mixed_precision=False)
    dataset = CausalDataset([1, 2, 3, 4], sequence_length=4)
    model = CausalTransformer(model_config)
    train(model, DataLoader(dataset, batch_size=1), model_config, training_config, tmp_path)
    checkpoint = load_checkpoint(tmp_path / "latest.pt")
    assert checkpoint["step"] == 2
    assert "model_config" in checkpoint
