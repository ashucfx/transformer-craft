import time
from typing import Any

import torch
from torch import nn

from .config import ModelConfig
from .modeling import CausalTransformer, EncoderDecoderTransformer, MaskedTransformer
from .training import resolve_device


def build_model(objective: str, config: ModelConfig) -> nn.Module:
    if objective == "causal":
        return CausalTransformer(config)
    if objective == "masked":
        return MaskedTransformer(config)
    if objective == "seq2seq":
        return EncoderDecoderTransformer(config)
    raise ValueError(f"unsupported objective: {objective}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def benchmark(
    model: nn.Module,
    objective: str,
    config: ModelConfig,
    *,
    batch_size: int,
    steps: int,
    device: str,
) -> dict[str, Any]:
    if batch_size < 1 or steps < 1:
        raise ValueError("batch_size and steps must be positive")
    resolved_device = resolve_device(device)
    model = model.to(resolved_device).eval()
    inputs = torch.randint(
        config.vocab_size,
        (batch_size, config.max_sequence_length),
        device=resolved_device,
    )
    if objective == "seq2seq":
        def forward() -> Any:
            return model(inputs, inputs)
    else:
        def forward() -> Any:
            return model(inputs)
    for _ in range(min(3, steps)):
        forward()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    start = time.perf_counter()
    for _ in range(steps):
        forward()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed = time.perf_counter() - start
    tokens = batch_size * config.max_sequence_length * steps
    result: dict[str, Any] = {
        "device": str(resolved_device),
        "objective": objective,
        "parameters": count_parameters(model),
        "batch_size": batch_size,
        "sequence_length": config.max_sequence_length,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000 / steps,
        "tokens_per_second": tokens / elapsed,
    }
    if resolved_device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(resolved_device) / 1024**2
    return result
