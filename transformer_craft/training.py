import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig, TrainingConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> LambdaLR:
    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return max(step, 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item())

    return LambdaLR(optimizer, schedule)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    step: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
        },
        target,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def train(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
) -> list[float]:
    seed_everything(training_config.seed)
    device = resolve_device(training_config.device)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, training_config)
    use_amp = training_config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    for step in tqdm(range(training_config.max_steps), desc="training"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = model(**batch)["loss"] / training_config.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % training_config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        losses.append(float(loss.detach().item() * training_config.gradient_accumulation_steps))
        if (step + 1) % max(training_config.max_steps // 10, 1) == 0:
            save_checkpoint(
                Path(output_dir) / "latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step + 1,
                model_config,
                training_config,
            )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "config.json").write_text(
        json.dumps(
            {"model": model_config.to_dict(), "training": training_config.to_dict()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return losses
