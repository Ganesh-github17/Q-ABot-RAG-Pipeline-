"""
Text chunking module.

Strategy: fixed-size character chunking with overlap, aligned to sentence
boundaries where reasonably possible. This is the simplest chunking strategy
that still keeps semantically related sentences together, is easy to explain
in an interview, and needs no extra dependencies. See README.md for a full
discussion of why this strategy was chosen and its limitations.

Each chunk retains the source filename and page number of the page it came
from, plus a unique, deterministic chunk_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion import Page

# Sentence boundary heuristic: split after '.', '!', or '?' followed by
# whitespace. This is intentionally simple (no NLP library dependency) and
# works well for the kind of prose found in the sample documents.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


def _split_into_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or ([text] if text.strip() else [])


def chunk_page_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split a single page's text into overlapping chunks of approximately
    `chunk_size` characters, breaking on sentence boundaries when possible so
    that chunks remain readable and semantically coherent.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    def flush() -> str:
        return " ".join(current_sentences).strip()

    for sentence in sentences:
        # A single sentence longer than chunk_size is kept whole rather than
        # cut mid-word, to avoid destroying meaning; it simply becomes an
        # oversized chunk on its own.
        if current_len + len(sentence) + 1 > chunk_size and current_sentences:
            chunks.append(flush())

            # Build overlap: carry trailing sentences from the previous chunk
            # forward until we've covered approximately chunk_overlap chars.
            overlap_sentences: list[str] = []
            overlap_len = 0
            for prev_sentence in reversed(current_sentences):
                if overlap_len >= chunk_overlap:
                    break
                overlap_sentences.insert(0, prev_sentence)
                overlap_len += len(prev_sentence) + 1

            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += len(sentence) + 1

    if current_sentences:
        chunks.append(flush())

    return [c for c in chunks if c]


def chunk_pages(
    pages: list[Page], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every page and attach deterministic metadata to each chunk."""
    all_chunks: list[Chunk] = []
    for page in pages:
        page_texts = chunk_page_text(page.text, chunk_size, chunk_overlap)
        for local_index, chunk_text in enumerate(page_texts):
            chunk_id = f"{page.source}::p{page.page}::c{local_index}"
            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    source=page.source,
                    page=page.page,
                    chunk_id=chunk_id,
                )
            )
    return all_chunks
