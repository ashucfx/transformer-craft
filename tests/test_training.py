from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer_craft.config import ModelConfig, TrainingConfig
from transformer_craft.data import CausalDataset
from transformer_craft.modeling import CausalTransformer
from transformer_craft.training import evaluate, load_checkpoint, train


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


def test_training_can_resume_from_checkpoint(tmp_path: Path) -> None:
    model_config = ModelConfig(
        max_sequence_length=4,
        embedding_dimension=8,
        attention_heads=2,
        layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    first_config = TrainingConfig(max_steps=1, batch_size=1, device="cpu", mixed_precision=False)
    dataset = CausalDataset([1, 2, 3, 4], sequence_length=4)
    first_model = CausalTransformer(model_config)
    train(first_model, DataLoader(dataset, batch_size=1), model_config, first_config, tmp_path)

    second_config = TrainingConfig(max_steps=2, batch_size=1, device="cpu", mixed_precision=False)
    second_model = CausalTransformer(model_config)
    losses = train(
        second_model,
        DataLoader(dataset, batch_size=1),
        model_config,
        second_config,
        tmp_path,
        resume_from=tmp_path / "latest.pt",
    )
    assert len(losses) == 1
    assert load_checkpoint(tmp_path / "latest.pt")["step"] == 2


def test_training_writes_validation_metrics(tmp_path: Path) -> None:
    model_config = ModelConfig(
        max_sequence_length=4,
        embedding_dimension=8,
        attention_heads=2,
        layers=1,
        feed_forward_dimension=16,
        dropout=0.0,
    )
    training_config = TrainingConfig(max_steps=2, batch_size=1, device="cpu", mixed_precision=False)
    dataset = CausalDataset([1, 2, 3, 4, 5, 6, 7, 8], sequence_length=4)
    model = CausalTransformer(model_config)
    loader = DataLoader(dataset, batch_size=1)
    train(model, loader, model_config, training_config, tmp_path, validation_loader=loader)
    assert (tmp_path / "metrics.json").is_file()
    assert evaluate(model, loader, torch.device("cpu")) > 0
