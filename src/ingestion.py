"""
Document ingestion layer.

Responsible for:
  1. Discovering supported documents in a directory.
  2. Extracting text from each page of each document.
  3. Cleaning obviously repeated headers/footers, standalone page numbers,
     and excessive whitespace, without aggressively touching real content.
  4. Producing a structured list of page objects, each carrying the source
     filename and page number as metadata.

PDF extraction prefers PyMuPDF (fitz) when it is installed, since it is the
library recommended for this assignment, and falls back to pypdf otherwise.
This makes the ingestion layer usable in environments where PyMuPDF cannot be
installed (for example, an offline sandbox), while still working out of the
box with the recommended library when it is available.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf"}


class IngestionError(Exception):
    """Raised when a document cannot be discovered or read."""


@dataclass
class Page:
    """A single page of extracted text plus its metadata."""

    text: str
    source: str  # filename, e.g. "machine_learning.pdf"
    page: int  # 1-indexed page number


def _extract_with_pymupdf(path: Path) -> list[str]:
    import fitz  # PyMuPDF

    pages: list[str] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pages.append(page.get_text("text"))
    return pages


def _extract_with_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise IngestionError(f"Could not read '{path.name}': invalid or corrupted PDF.") from exc

    return [page.extract_text() or "" for page in reader.pages]


def _extract_raw_pages(path: Path) -> list[str]:
    """Extract raw (uncleaned) text for every page of a PDF file."""
    try:
        import fitz  # noqa: F401  (PyMuPDF availability check)

        return _extract_with_pymupdf(path)
    except ImportError:
        return _extract_with_pypdf(path)
    except Exception as exc:  # pragma: no cover - defensive, PyMuPDF failure
        raise IngestionError(f"Could not read '{path.name}': {exc}") from exc


def _find_repeated_lines(pages: list[str], min_occurrences: int = 3) -> set[str]:
    """
    Detect lines (typically headers/footers) that repeat across many pages of
    the same document. A line that appears on most pages, verbatim, is almost
    certainly boilerplate rather than meaningful content.
    """
    if len(pages) < min_occurrences:
        return set()

    counts: Counter[str] = Counter()
    for page_text in pages:
        seen_this_page = {line.strip() for line in page_text.splitlines() if line.strip()}
        counts.update(seen_this_page)

    threshold = max(min_occurrences, int(len(pages) * 0.6))
    return {line for line, count in counts.items() if count >= threshold and len(line) < 120}


_STANDALONE_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)


def clean_page_text(raw_text: str, repeated_lines: set[str]) -> str:
    """
    Remove repeated headers/footers, standalone page numbers, and collapse
    excessive whitespace, while leaving meaningful sentences untouched.
    """
    cleaned_lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in repeated_lines:
            continue
        if _STANDALONE_PAGE_NUMBER_RE.match(stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines / excessive internal whitespace without
    # touching normal paragraph spacing.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Page]:
    """Load a single PDF file into a list of cleaned Page objects."""
    if not path.exists():
        raise IngestionError(f"File not found: {path}")

    raw_pages = _extract_raw_pages(path)
    if not raw_pages:
        raise IngestionError(f"'{path.name}' contains no pages.")

    repeated_lines = _find_repeated_lines(raw_pages)

    pages: list[Page] = []
    for index, raw_text in enumerate(raw_pages, start=1):
        cleaned = clean_page_text(raw_text, repeated_lines)
        pages.append(Page(text=cleaned, source=path.name, page=index))
    return pages


def discover_documents(data_dir: str | Path) -> list[Path]:
    """Find all supported documents under data_dir, sorted by filename."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise IngestionError(
            f"Data directory not found: {data_path}. Create it and add "
            "documents, or see data/README.md for instructions."
        )

    documents = sorted(
        p for p in data_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not documents:
        raise IngestionError(
            f"No supported documents (.pdf) found in {data_path}. "
            "See data/README.md for what to add."
        )
    return documents


def load_documents(data_dir: str | Path) -> list[Page]:
    """Discover and load every supported document in data_dir."""
    all_pages: list[Page] = []
    for doc_path in discover_documents(data_dir):
        pages = load_pdf(doc_path)
        non_empty_pages = [p for p in pages if p.text]
        if not non_empty_pages:
            raise IngestionError(
                f"'{doc_path.name}' appears to be empty or contains no "
                "extractable text (it may be a scanned image without OCR)."
            )
        all_pages.extend(pages)
    return all_pages
