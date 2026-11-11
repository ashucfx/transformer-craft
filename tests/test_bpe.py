from pathlib import Path

from transformer_craft.data import BPETokenizer


def test_bpe_training_round_trip_and_persistence(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train("abababab code code", vocab_size=270)
    text = "abab code"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.encode(text) == tokenizer.encode(text)


def test_bpe_rejects_small_vocabulary() -> None:
    try:
        BPETokenizer.train("text", vocab_size=259)
    except ValueError as error:
        assert "at least 260" in str(error)
    else:
        raise AssertionError("expected a vocabulary validation error")
