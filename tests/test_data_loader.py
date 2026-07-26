"""Regression tests for src/data_loader.py."""

from __future__ import annotations

import pytest

from src.data_loader import load_directory, load_document, load_txt


def test_load_txt_creates_single_page(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Patient reports no known drug allergies.\n\n\nSecond paragraph.")
    pages = load_txt(f)
    assert len(pages) == 1
    assert pages[0].source == "note.txt"
    assert pages[0].page_number == 1
    # clean_text() should collapse 3+ newlines to a single paragraph break
    assert "\n\n\n" not in pages[0].text


def test_load_document_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "notes.exe"
    f.write_text("not a real document")
    with pytest.raises(ValueError):
        load_document(f)


def test_load_document_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        load_document(missing)


def test_load_directory_skips_unsupported_files_without_crashing(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "ignore_me.exe").write_text("binary-ish")
    pages = load_directory(tmp_path)
    assert len(pages) == 1
    assert pages[0].source == "a.txt"