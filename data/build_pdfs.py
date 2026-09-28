"""
One-time helper script used to generate the sample PDF documents for this
project from the original plain-text content in data/source_text/.

This script is NOT part of the RAG application itself (it is not imported by
index.py or app.py). It exists purely so the document collection required by
the assignment could be produced locally, from original written content,
without needing to download any external files.

Run with: python data/build_pdfs.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SOURCE_DIR = Path(__file__).parent / "source_text"
OUTPUT_DIR = Path(__file__).parent / "documents"

FILES = {
    "ai_intro.txt": "ai_intro.pdf",
    "machine_learning.txt": "machine_learning.pdf",
    "deep_learning.txt": "deep_learning.pdf",
    "nlp.txt": "nlp.pdf",
    "computer_vision.txt": "computer_vision.pdf",
}


def parse_source(text: str):
    """Split the plain-text source into (kind, content) blocks."""
    blocks = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("TITLE:"):
            blocks.append(("title", line.replace("TITLE:", "").strip()))
        elif line.startswith("SECTION:"):
            blocks.append(("heading", line.replace("SECTION:", "").strip()))
        else:
            blocks.append(("body", line))
    return blocks


def build_pdf(src_path: Path, out_path: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=20, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "DocHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        spaceAfter=10,
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=out_path.stem,
    )

    story = []
    for kind, content in parse_source(src_path.read_text(encoding="utf-8")):
        safe = re.sub(r"&(?!amp;)", "&amp;", content)
        if kind == "title":
            story.append(Paragraph(safe, title_style))
            story.append(Spacer(1, 6))
        elif kind == "heading":
            story.append(Paragraph(safe, heading_style))
        else:
            story.append(Paragraph(safe, body_style))

    doc.build(story)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, out_name in FILES.items():
        src_path = SOURCE_DIR / src_name
        out_path = OUTPUT_DIR / out_name
        build_pdf(src_path, out_path)
        word_count = len(src_path.read_text(encoding="utf-8").split())
        print(f"Built {out_path.name} ({word_count} words of source text)")


if __name__ == "__main__":
    main()
