import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .data import ByteTokenizer, CausalDataset, CodeDataset, MaskedDataset
from .inference import evaluate_causal, generate
from .modeling import CausalTransformer, MaskedTransformer
from .training import load_checkpoint, resolve_device, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformer-craft")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--objective", choices=("causal", "masked"), default="causal")
    train_parser.add_argument("--text-file", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--sequence-length", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--resume-from", type=Path)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--text-file", type=Path, required=True)
    evaluate_parser.add_argument("--objective", choices=("causal", "masked"), default="causal")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--device", default="auto")
    return parser


def run_train(args: argparse.Namespace) -> None:
    tokenizer = ByteTokenizer()
    model_config = ModelConfig(max_sequence_length=args.sequence_length)
    training_config = TrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    if args.objective == "masked":
        dataset = MaskedDataset(
            tokenizer.encode(args.text_file.read_text(encoding="utf-8")),
            args.sequence_length,
            mask_id=tokenizer.special.mask,
            pad_id=tokenizer.special.pad,
            seed=args.seed,
        )
        model = MaskedTransformer(model_config)
    else:
        dataset = CausalDataset(
            tokenizer.encode(args.text_file.read_text(encoding="utf-8")),
            args.sequence_length,
            tokenizer.special.pad,
        )
        model = CausalTransformer(model_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train(
        model,
        loader,
        model_config,
        training_config,
        args.output_dir,
        resume_from=args.resume_from,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = (
        CausalTransformer(model_config)
        if args.objective == "causal"
        else MaskedTransformer(model_config)
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = ByteTokenizer()
    dataset = CodeDataset.from_text(args.text_file, tokenizer, model_config.max_sequence_length)
    if args.objective == "masked":
        dataset = MaskedDataset(
            tokenizer.encode(args.text_file.read_text(encoding="utf-8")),
            model_config.max_sequence_length,
            mask_id=tokenizer.special.mask,
            pad_id=tokenizer.special.pad,
        )
    loader = DataLoader(dataset, batch_size=1)
    if args.objective == "causal":
        print(evaluate_causal(model, loader))
        return
    losses = []
    with torch.no_grad():
        for batch in loader:
            losses.append(float(model(**batch)["loss"].item()))
    print({"loss": sum(losses) / len(losses)})


def run_generate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = CausalTransformer(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    tokenizer = ByteTokenizer()
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    tokens = generate(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.special.eos,
    )
    print(tokenizer.decode(tokens[0]))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_generate(args)
