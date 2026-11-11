from pathlib import Path

from transformer_craft.cli import build_parser


def test_train_parser_accepts_corpus_directory(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--corpus-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert args.corpus_dir == tmp_path
    assert args.text_file is None


def test_evaluate_parser_accepts_text_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--text-file",
            str(tmp_path / "sample.py"),
        ]
    )
    assert args.text_file == tmp_path / "sample.py"
    assert args.corpus_dir is None


def test_seq2seq_parser_accepts_target_file(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--objective",
            "seq2seq",
            "--text-file",
            str(tmp_path / "source.txt"),
            "--target-file",
            str(tmp_path / "target.txt"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert args.objective == "seq2seq"
    assert args.target_file == tmp_path / "target.txt"
