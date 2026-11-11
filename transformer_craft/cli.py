import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .benchmark import benchmark, build_model
from .config import ModelConfig, TrainingConfig
from .data import (
    ByteTokenizer,
    CausalDataset,
    CodeDataset,
    MaskedDataset,
    SequencePairDataset,
    load_corpus_text,
)
from .inference import evaluate_causal, generate
from .modeling import CausalTransformer, EncoderDecoderTransformer, MaskedTransformer
from .training import (
    finalize_distributed,
    initialize_distributed,
    load_checkpoint,
    resolve_device,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformer-craft")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument(
        "--objective", choices=("causal", "masked", "seq2seq"), default="causal"
    )
    _add_input_arguments(train_parser)
    train_parser.add_argument("--target-file", type=Path)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--steps", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--sequence-length", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--embedding-dimension", type=int, default=256)
    train_parser.add_argument("--attention-heads", type=int, default=8)
    train_parser.add_argument("--layers", type=int, default=6)
    train_parser.add_argument("--feed-forward-dimension", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument(
        "--positional-encoding", choices=("learned", "sinusoidal"), default="learned"
    )
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--warmup-steps", type=int, default=100)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    _add_input_arguments(evaluate_parser)
    evaluate_parser.add_argument("--target-file", type=Path)
    evaluate_parser.add_argument(
        "--objective", choices=("causal", "masked", "seq2seq"), default="causal"
    )
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--device", default="auto")
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--checkpoint", type=Path, required=True)
    benchmark_parser.add_argument(
        "--objective", choices=("causal", "masked", "seq2seq"), default="causal"
    )
    benchmark_parser.add_argument("--batch-size", type=int, default=1)
    benchmark_parser.add_argument("--steps", type=int, default=20)
    benchmark_parser.add_argument("--device", default="auto")
    return parser


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text-file", type=Path)
    inputs.add_argument("--corpus-dir", type=Path)
    parser.add_argument(
        "--extensions",
        default=".py,.js,.ts,.java,.cpp,.c,.rs,.go",
        help="comma-separated corpus extensions",
    )


def _load_input_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8")
    extensions = {
        extension.strip()
        for extension in args.extensions.split(",")
        if extension.strip()
    }
    return load_corpus_text([args.corpus_dir], extensions)


def _load_sequence_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.target_file is None:
        raise ValueError("--target-file is required when objective is seq2seq")
    source_lines = _load_input_text(args).splitlines()
    target_lines = args.target_file.read_text(encoding="utf-8").splitlines()
    if len(source_lines) != len(target_lines):
        raise ValueError("source and target files must contain the same number of lines")
    return list(zip(source_lines, target_lines))


def run_train(args: argparse.Namespace) -> None:
    distributed = initialize_distributed(args.device)
    tokenizer = ByteTokenizer()
    model_config = ModelConfig(
        max_sequence_length=args.sequence_length,
        embedding_dimension=args.embedding_dimension,
        attention_heads=args.attention_heads,
        layers=args.layers,
        feed_forward_dimension=args.feed_forward_dimension,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
    )
    training_config = TrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        mixed_precision=args.mixed_precision,
    )
    if args.objective == "seq2seq":
        dataset = SequencePairDataset.from_text_pairs(
            _load_sequence_pairs(args), tokenizer, args.sequence_length
        )
        model = EncoderDecoderTransformer(model_config)
    elif args.objective == "masked":
        text = _load_input_text(args)
        dataset = MaskedDataset(
            tokenizer.encode(text),
            args.sequence_length,
            mask_id=tokenizer.special.mask,
            pad_id=tokenizer.special.pad,
            seed=args.seed,
        )
        model = MaskedTransformer(model_config)
    else:
        text = _load_input_text(args)
        dataset = CausalDataset(
            tokenizer.encode(text),
            args.sequence_length,
            tokenizer.special.pad,
        )
        model = CausalTransformer(model_config)
    train_dataset, validation_dataset = dataset.split(args.validation_fraction, args.seed)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    validation_sampler = (
        DistributedSampler(validation_dataset, shuffle=False) if distributed else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
    )
    try:
        train(
            model,
            loader,
            model_config,
            training_config,
            args.output_dir,
            resume_from=args.resume_from,
            validation_loader=validation_loader,
            run_metadata={"objective": args.objective},
        )
    finally:
        if distributed:
            finalize_distributed()


def run_evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    model_config = ModelConfig(**checkpoint["model_config"])
    if args.objective == "seq2seq":
        model = EncoderDecoderTransformer(model_config)
    else:
        model = (
            CausalTransformer(model_config)
            if args.objective == "causal"
            else MaskedTransformer(model_config)
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = ByteTokenizer()
    if args.objective == "seq2seq":
        dataset = SequencePairDataset.from_text_pairs(
            _load_sequence_pairs(args), tokenizer, model_config.max_sequence_length
        )
        loader = DataLoader(dataset, batch_size=1)
        losses = []
        with torch.no_grad():
            for batch in loader:
                losses.append(float(model(**batch)["loss"].item()))
        print({"loss": sum(losses) / len(losses)})
        return
    dataset = CodeDataset(
        tokenizer.encode(_load_input_text(args)),
        model_config.max_sequence_length,
        tokenizer.special.pad,
    )
    if args.objective == "masked":
        dataset = MaskedDataset(
            tokenizer.encode(_load_input_text(args)),
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
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
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


def run_benchmark(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(args.objective, config)
    model.load_state_dict(checkpoint["model"])
    print(
        benchmark(
            model,
            args.objective,
            config,
            batch_size=args.batch_size,
            steps=args.steps,
            device=args.device,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "generate":
        run_generate(args)
    else:
        run_benchmark(args)
