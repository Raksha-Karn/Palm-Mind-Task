import pytest

from app.errors import AppError
from app.schemas import ChunkStrategy
from app.services.chunking import TextPage, chunk_text, extract_text


def test_fixed_chunks_overlap_and_preserve_pages() -> None:
    chunks = chunk_text([TextPage("abcdefghijklmnopqrstuvwxyz", 2)], ChunkStrategy.FIXED, 10, 3, 20)
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
    assert all(chunk.page == 2 for chunk in chunks)


def test_recursive_split_preserves_boundaries() -> None:
    text = "a" * 25 + "\n\n" + "b" * 50
    chunks = chunk_text([TextPage(text, None)], ChunkStrategy.RECURSIVE, 40, 5, 20)
    assert chunks[0].text == "a" * 25
    assert all(len(chunk.text) <= 40 for chunk in chunks)
    assert chunks[-1].text.endswith("b" * 20)


def test_invalid_chunk_options_and_empty_files() -> None:
    with pytest.raises(AppError):
        chunk_text([TextPage("text", None)], ChunkStrategy.FIXED, 10, 10, 20)
    with pytest.raises(AppError):
        extract_text(b" ", ".txt", 100)
    with pytest.raises(AppError):
        extract_text(b"not a PDF", ".pdf", 100)
