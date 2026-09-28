import pytest

from src.chunking import Chunk, chunk_page_text, chunk_pages
from src.ingestion import Page

LONG_TEXT = (
    "Artificial intelligence is the study of building systems that behave "
    "intelligently. Machine learning is a subfield of artificial intelligence "
    "that learns patterns from data. Deep learning uses neural networks with "
    "many layers. Natural language processing helps computers understand "
    "text. Computer vision helps computers understand images. "
) * 5  # repeat to make it long enough to require multiple chunks


def test_chunks_are_generated():
    chunks = chunk_page_text(LONG_TEXT, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunk_size_is_respected_approximately():
    chunks = chunk_page_text(LONG_TEXT, chunk_size=200, chunk_overlap=50)
    # Allow some slack since we align to sentence boundaries rather than
    # cutting mid-sentence.
    for chunk in chunks:
        assert len(chunk) <= 260


def test_overlap_exists_between_consecutive_chunks():
    chunks = chunk_page_text(LONG_TEXT, chunk_size=200, chunk_overlap=80)
    assert len(chunks) >= 2
    # At least one sentence from the end of chunk N should also appear at the
    # start of chunk N+1.
    first_chunk_sentences = chunks[0].split(". ")
    last_sentence_of_first = first_chunk_sentences[-1]
    assert last_sentence_of_first[:15] in chunks[1] or any(
        s[:15] in chunks[1] for s in first_chunk_sentences
    )


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_page_text(LONG_TEXT, chunk_size=100, chunk_overlap=100)


def test_empty_text_produces_no_chunks():
    assert chunk_page_text("", chunk_size=200, chunk_overlap=50) == []


def test_metadata_is_preserved_on_chunks():
    pages = [
        Page(text=LONG_TEXT, source="machine_learning.pdf", page=4),
        Page(text=LONG_TEXT, source="machine_learning.pdf", page=5),
    ]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "machine_learning.pdf" for c in chunks)
    assert {c.page for c in chunks} == {4, 5}

    # chunk_id must be unique and deterministic given (source, page, index).
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all("machine_learning.pdf" in cid for cid in chunk_ids)
