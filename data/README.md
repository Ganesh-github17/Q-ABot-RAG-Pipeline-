# Document Collection

This folder contains the document collection that `python index.py` indexes.

## What's included

`data/documents/` contains 5 PDF files on a single coherent topic, **Artificial
Intelligence and Machine Learning**, so that cross-document questions are
possible:

| File | Topic | Pages | Approx. words |
|---|---|---|---|
| `ai_intro.pdf` | Introduction to Artificial Intelligence | 3 | ~1,025 |
| `machine_learning.pdf` | Machine Learning Fundamentals | 3 | ~1,100 |
| `deep_learning.pdf` | Deep Learning Explained | 3 | ~1,075 |
| `nlp.pdf` | Natural Language Processing Overview | 3 | ~1,050 |
| `computer_vision.pdf` | Computer Vision Fundamentals | 3 | ~1,060 |

**Important, honest disclosure:** these five PDFs are **original written
content**, authored specifically for this assignment (see
`data/source_text/*.txt` for the plain-text source, and
`data/build_pdfs.py` for the script that renders them to PDF using
ReportLab). They are **not** official publications, textbooks, or
copyrighted third-party material, and they should not be presented as such.
They were generated this way, instead of downloaded, because the development
environment used to build this project has no internet access. The content
itself is genuine, non-trivial, factually grounded explanatory writing on
each topic (not Lorem Ipsum or placeholder text), which is why it works well
for demonstrating real retrieval and grounded question answering.

## Replacing these with your own real-world documents

If you would like to use real third-party PDFs instead (for example, actual
lecture notes, whitepapers, or textbook chapters you have the rights to use),
simply:

1. Delete or keep the existing files in `data/documents/`.
2. Drop your own PDF file(s) into `data/documents/`.
3. Re-run indexing with a clean database:

   ```bash
   python index.py --force
   ```

Requirements for any document you add:

- Must be a `.pdf` file.
- Should contain at least ~500 words or 2 pages of real text (scanned
  image-only PDFs without an OCR text layer will fail ingestion with a clear
  error, since no OCR step is implemented in this project).
- Ideally documents share a common topic so that questions can span multiple
  files, which better demonstrates the retrieval step of the RAG pipeline.
