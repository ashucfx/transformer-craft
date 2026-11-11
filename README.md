# Transformer Craft

Transformer Craft is a from-scratch PyTorch implementation of a compact, extensible language-modeling stack for source code. It provides the core architecture, causal and masked objectives, deterministic data preparation, checkpointing primitives, and a CLI suitable for reproducible experiments.

## What this project demonstrates

- Multi-head self-attention with causal masking and optional padding masks
- Pre-layer-normalized Transformer blocks with residual connections
- Learned and sinusoidal positional encodings
- Decoder-only causal language modeling
- Encoder-only masked language modeling
- Deterministic byte-level tokenization with a reserved vocabulary
- Gradient accumulation, clipping, mixed precision, and checkpoint resume
- Small, testable modules with typed public interfaces

## Architecture

```text
Text files
   |
   v
ByteTokenizer -> fixed-length token windows -> DataLoader
   |
   +--> CausalTransformer -> next-token loss
   |
   +--> MaskedTransformer  -> masked-token loss
```

The default tokenizer is intentionally dependency-free. It preserves every UTF-8 byte, which makes it robust for arbitrary code and avoids an implicit vocabulary-training phase. A future tokenizer can implement the same protocol without changing the model or training APIs.

## Quick start

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Run a short causal-language-model experiment:

```powershell
transformer-craft train `
  --objective causal `
  --text-file data\sample.txt `
  --output-dir checkpoints\demo `
  --steps 100 `
  --batch-size 8 `
  --sequence-length 128
```

Evaluate a checkpoint:

```powershell
transformer-craft evaluate `
  --checkpoint checkpoints\demo\latest.pt `
  --text-file data\sample.txt `
  --objective causal
```

## Repository layout

```text
transformer_craft/
  config.py       validated model and training configuration
  data.py         tokenizer, windowing, masking, and datasets
  modeling.py     attention, Transformer blocks, and objectives
  training.py     reproducible training and checkpoint lifecycle
  cli.py          train and evaluate commands
tests/
  test_data.py
  test_modeling.py
  test_training.py
```

## Design choices

### Attention

Attention uses fused scaled dot-product attention when available in the installed PyTorch version and falls back to an equivalent explicit implementation. Causal masking is applied inside the attention module, while padding masks remain an explicit caller concern.

### Objectives

The causal objective shifts labels by one position and ignores padding. The masked objective selects tokens independently with a configurable masking probability and uses `-100` as the ignored label, matching PyTorch's cross-entropy convention.

### Reproducibility

Experiments accept a seed and persist the resolved configuration alongside checkpoints. Data ordering, model initialization, and masking use seeded generators. Checkpoints contain model, optimizer, scheduler, scaler, step, and configuration state.

### Operational boundaries

This repository intentionally keeps storage and orchestration local: input is a text file and output is a checkpoint directory. That makes the core easy to test and deploy while leaving room for object storage, experiment tracking, distributed training, and a production tokenizer as replaceable integrations.

## Quality gates

```powershell
ruff check .
pytest
```

The test suite covers tensor shapes, causal information flow, masking invariants, checkpoint round trips, and deterministic tokenization.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
