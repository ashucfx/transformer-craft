from pathlib import Path

import pytest

from transformer_craft.data import discover_code_files, load_corpus_text


def test_discover_code_files_is_sorted_and_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.js").write_text("c", encoding="utf-8")

    files = discover_code_files([tmp_path])

    assert [path.name for path in files] == ["a.py", "b.py", "c.js"]


def test_load_corpus_text_preserves_file_order(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("second", encoding="utf-8")
    (tmp_path / "a.py").write_text("first", encoding="utf-8")

    assert load_corpus_text([tmp_path]) == "first\nsecond"


def test_corpus_loader_rejects_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no matching corpus files"):
        load_corpus_text([tmp_path])
