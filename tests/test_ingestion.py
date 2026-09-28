from pathlib import Path

import pytest

from src.ingestion import (
    IngestionError,
    clean_page_text,
    discover_documents,
    load_documents,
    load_pdf,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "documents" / "machine_learning.pdf"


def test_discover_documents_finds_sample_pdfs():
    docs = discover_documents(PROJECT_ROOT / "data" / "documents")
    names = {p.name for p in docs}
    assert "machine_learning.pdf" in names
    assert len(docs) >= 4  # at least 4-5 documents as required


def test_discover_documents_missing_dir_raises():
    with pytest.raises(IngestionError):
        discover_documents(PROJECT_ROOT / "data" / "does_not_exist")


def test_pdf_extraction_produces_pages_with_text():
    pages = load_pdf(SAMPLE_PDF)
    assert len(pages) >= 2  # at least 2 pages, per assignment requirement
    assert all(page.text for page in pages)
    assert any("machine learning" in page.text.lower() for page in pages)


def test_pdf_page_metadata_is_correct():
    pages = load_pdf(SAMPLE_PDF)
    assert all(page.source == "machine_learning.pdf" for page in pages)
    page_numbers = [page.page for page in pages]
    assert page_numbers == list(range(1, len(pages) + 1))


def test_clean_page_text_removes_standalone_page_numbers():
    raw = "Some real sentence about AI.\n4\nAnother real sentence."
    cleaned = clean_page_text(raw, repeated_lines=set())
    assert "4" not in cleaned.splitlines()
    assert "Some real sentence about AI." in cleaned


def test_clean_page_text_removes_repeated_header():
    raw = "CONFIDENTIAL DRAFT\nReal content line one."
    cleaned = clean_page_text(raw, repeated_lines={"CONFIDENTIAL DRAFT"})
    assert "CONFIDENTIAL DRAFT" not in cleaned
    assert "Real content line one." in cleaned


def test_load_documents_raises_on_empty_pdf(tmp_path):
    # Build a minimal, structurally valid but text-less PDF to simulate an
    # empty / scanned document with no extractable text.
    empty_pdf_bytes = (
        b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    empty_dir = tmp_path / "documents"
    empty_dir.mkdir()
    (empty_dir / "empty.pdf").write_bytes(empty_pdf_bytes)

    with pytest.raises(IngestionError):
        load_documents(empty_dir)
