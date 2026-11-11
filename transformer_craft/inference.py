import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import ByteTokenizer, CodeDataset
from .modeling import CausalTransformer


@torch.no_grad()
def generate(
    model: CausalTransformer,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> Tensor:
    if prompt.ndim != 2 or prompt.shape[1] == 0:
        raise ValueError("prompt must have shape [batch, sequence] and be non-empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided")
    model.eval()
    tokens = prompt
    for _ in range(max_new_tokens):
        context = tokens[:, -model.backbone.config.max_sequence_length :]
        logits = model(context)["logits"][:, -1] / temperature
        if top_k is not None:
            values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(1, indices, values)
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_id is not None and torch.all(next_token == eos_id):
            break
    return tokens


@torch.no_grad()
def evaluate_causal(
    model: CausalTransformer,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    resolved_device = torch.device(device)
    total_nll = 0.0
    total_tokens = 0
    for batch in batches:
        input_ids = batch["input_ids"].to(resolved_device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(resolved_device)
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[~attention_mask] = -100
        logits = model(input_ids, attention_mask)["logits"]
        shifted_logits = logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        total_nll += float(
            F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
        total_tokens += int((shifted_labels != -100).sum())
    if total_tokens == 0:
        raise ValueError("evaluation data contains no valid target tokens")
    loss = total_nll / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss)}


def build_causal_dataset(
    text: str, tokenizer: ByteTokenizer, sequence_length: int
) -> CodeDataset:
    return CodeDataset(tokenizer.encode(text), sequence_length, tokenizer.special.pad)
