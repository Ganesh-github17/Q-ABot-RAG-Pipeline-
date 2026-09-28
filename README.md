# Basic Document Q&A Bot using RAG

## Description

A Retrieval-Augmented Generation (RAG) system that answers natural-language
questions about a local collection of PDF documents. It extracts and chunks
document text, creates deterministic local embeddings, stores the vectors in
a persistent ChromaDB database, retrieves the most relevant chunks for a
given question, and produces a grounded extractive answer from the retrieved
context — always citing the exact source file and page number.

## Features

- PDF ingestion with text cleaning (removes repeated headers/footers, stray
  page numbers, and excess whitespace).
- Configurable sentence-aware chunking with overlap.
- Batched embedding generation (never one chunk at a time).
- Persistent ChromaDB vector storage, fully separate from the query app.
- Configurable top-k similarity retrieval.
- Strict grounding: the default answer path is extractive and only uses the
  retrieved context, and the app also has a weak-retrieval safety net.
- Deterministic citations built directly from retrieved chunk metadata
  (never invented by the LLM).
- Interactive CLI (`app.py`) and an optional Streamlit web UI
  (`streamlit_app.py`) built on the exact same pipeline.
- Unit tests for chunking, ingestion, and retrieval using mocks (no API key
  required to run the test suite).

## Architecture

```
Documents (PDF)
      │
      ▼
Document Ingestion (src/ingestion.py)
  - discover files, extract per-page text, clean headers/footers/whitespace
      │
      ▼
Text Chunking (src/chunking.py)
  - sentence-aware fixed-size chunks with overlap, metadata attached
      │
      ▼
Embeddings (src/embeddings.py)
  - OpenAI embedding model, called in BATCHES
      │
      ▼
Persistent Vector Database (src/vector_store.py — ChromaDB, on disk)
      │
      ▼
   [ index.py stops here — indexing and querying are separate processes ]
      │
Query Embedding (src/embeddings.py, same embedding model)
      │
      ▼
Similarity Retrieval (src/retriever.py)
  - top-k nearest chunks + weak-retrieval check
      │
      ▼
LLM (src/generator.py)
  - strict grounding system prompt, context + question
      │
      ▼
Grounded Answer + Source Citations (src/qa.py, app.py / streamlit_app.py)
```

ASCII pipeline summary:

```
[PDFs] -> [Ingestion] -> [Chunking] -> [Embeddings] -> [ChromaDB]
                                                            │
[Question] -> [Query Embedding] -> [Retrieval] <-----------┘
                                        │
                                        ▼
                                [LLM + Grounding Prompt]
                                        │
                                        ▼
                            [Answer + Citations + Chunks]
```

## Tech Stack

Installed and used by this project (see `requirements.txt` for exact pins):

| Library | Purpose |
|---|---|
| Python 3.11+ (developed/tested on 3.12.3) | Language runtime |
| `openai` | Optional OpenAI-backed embeddings and chat completions if you opt in |
| `chromadb` | Persistent vector database |
| `pymupdf` (fitz) | Primary PDF text extraction |
| `pypdf` | Fallback PDF text extraction, and used to build the sample PDFs' companion tooling |
| `python-dotenv` | Loads `.env` configuration |
| `streamlit` | Optional web UI |
| `pytest` | Test framework |
| `reportlab` | One-time script (`data/build_pdfs.py`) that renders the sample documents to PDF — not used at runtime by the app |

> **Note on this development environment:** the sandbox used to build this
> project has no internet access, so some third-party packages may not be
> installed automatically here. The project itself now runs in a free local
> mode by default and does not require an API key.

## Project Structure

```
document-qa-bot/
│
├── data/
│   ├── README.md            # explains the document collection (see below)
│   ├── documents/           # the 5 PDFs indexed by index.py
│   ├── source_text/         # original plain-text source for the PDFs
│   └── build_pdfs.py        # one-time script that rendered the PDFs
│
├── src/
│   ├── __init__.py
│   ├── config.py            # all configuration, read from environment variables
│   ├── ingestion.py         # PDF discovery, text extraction, cleaning
│   ├── chunking.py          # sentence-aware chunking with overlap
│   ├── embeddings.py        # local embeddings by default; OpenAI optional
│   ├── vector_store.py      # persistent ChromaDB wrapper
│   ├── retriever.py         # query embedding + top-k retrieval + weak-match check
│   ├── generator.py         # local extractive answers by default; OpenAI optional
│   ├── local_models.py      # dependency-free embeddings and extractive answers
│   └── qa.py                # orchestrates retriever + generator, builds citations
│
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_ingestion.py
│   └── test_retrieval.py
│
├── index.py                 # WORKFLOW 1: build the vector database
├── app.py                   # WORKFLOW 2: interactive CLI Q&A loop
├── streamlit_app.py         # optional web UI on top of the same pipeline
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

## Document Collection

See `data/README.md` for full details. In short: `data/documents/` contains 5
PDFs (3 pages / ~1,000+ words each) covering Introduction to AI, Machine
Learning, Deep Learning, NLP, and Computer Vision — a single coherent topic
so cross-document questions work well. These are original written content
created for this assignment (not lorem ipsum, and not copied from any
copyrighted source), since the build environment had no internet access to
download third-party documents. `data/README.md` explains exactly how to
swap in your own real-world PDFs instead.

## Chunking Strategy

**Method:** fixed character-length chunking (`CHUNK_SIZE=800` characters),
aligned to sentence boundaries where possible, with `CHUNK_OVERLAP=150`
characters carried from the end of one chunk into the start of the next.
Implemented in `src/chunking.py`.

**Why this method:** it requires no extra NLP dependency, is trivial to
reason about and explain, and keeping sentences intact (rather than cutting
mid-sentence at a raw character boundary) preserves enough local meaning for
embeddings to work well, while still giving predictable, boundable chunk
sizes for the embedding API.

**Advantages:**
- Simple and fully explainable — a fixed budget of characters per chunk.
- Overlap means an answer that straddles a chunk boundary in the source text
  usually still appears whole in at least one chunk.
- Sentence-aware splitting avoids visually cutting sentences in half.

**Limitations:**
- A fixed character budget is not aware of semantic topic boundaries — a
  single chunk can still span two loosely related ideas if they happen to be
  adjacent sentences.
- Overlap increases the total number of chunks (and therefore embedding
  cost) compared to non-overlapping chunking.
- `800`/`150` are reasonable, commonly used defaults for this kind of
  explanatory prose — they are **not** claimed to be universally optimal, and
  a different corpus (e.g. dense legal text or code) may need different
  values. Both are configurable via `CHUNK_SIZE` / `CHUNK_OVERLAP`.

## Embedding Model

- **Model:** `text-embedding-3-small` by default (`EMBEDDING_MODEL` env var).
  Chosen because it is inexpensive, fast, and produces strong retrieval
  quality for general-purpose text — a good fit for a beginner assignment
  without incurring high API cost.
- **Batching:** `src/embeddings.py` always splits the list of chunk texts
  into batches of `EMBEDDING_BATCH_SIZE` (default `32`) and sends each batch
  as a single call to `client.embeddings.create(model=..., input=<batch>)`.
  Chunks are **never** embedded one at a time. The response order from the
  OpenAI API is guaranteed to match input order, so embeddings are re-attached
  to their originating chunks positionally.

## Vector Database

- **ChromaDB**, via `chromadb.PersistentClient(path=CHROMA_PATH)`
  (`src/vector_store.py`), persists to disk at `./chroma_db` by default so
  the database survives process restarts — indexing (`index.py`) writes it
  once, and querying (`app.py` / `streamlit_app.py`) simply opens the
  existing collection without recreating anything.
- Each stored vector carries metadata: `source` (filename), `page` (1-indexed
  page number), and `chunk_id` (a deterministic, unique ID).
- The collection uses cosine distance (`hnsw:space: cosine`). Chroma reports
  a *distance*, not a normalized 0–1 similarity score, so raw distance values
  are shown as "distance" rather than mislabeled as a percentage similarity.

## Setup

```bash
git clone <your-fork-or-copy-of-this-repo>
cd document-qa-bot

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (Command Prompt)
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

pip install -r requirements.txt

cp .env.example .env             # macOS/Linux
# copy .env.example .env         # Windows

# then edit .env and set OPENAI_API_KEY
```

## Environment Variables

Copy `.env.example` to `.env` if you want to override defaults. The project
works out of the box without `OPENAI_API_KEY`. If you want to opt into the
OpenAI-backed path later, set `EMBEDDING_BACKEND=openai` and/or
`GENERATION_BACKEND=openai`, then provide `OPENAI_API_KEY`.

## Indexing

```bash
python index.py
```

Discovers PDFs in `data/documents/` (path configurable via `DATA_DIR`),
extracts and cleans text, chunks it, generates embeddings in batches, and
stores everything in the persistent ChromaDB collection. Running it again
without `--force` skips re-indexing if the collection already has data; pass
`--force` to wipe and rebuild:

```bash
python index.py --force
```

## Running the Bot

```bash
python app.py
```

Starts an interactive loop. Type a question and press Enter; type `exit` or
`quit` to stop. Each answer is followed by its sources and the retrieved
context chunks that were used to generate it.

## Optional Streamlit UI

```bash
streamlit run streamlit_app.py
```

Provides the same question box, answer, sources, and retrieved-context
display as the CLI, in a browser tab. It calls the exact same
`src/qa.py:answer_question` function as `app.py` — no RAG logic is
duplicated.

## Example Questions

Based on the actual sample documents, spanning multiple files:

1. "What is artificial intelligence?" *(ai_intro.pdf)*
2. "What is the difference between supervised and unsupervised learning?" *(machine_learning.pdf)*
3. "How do convolutional neural networks work?" *(deep_learning.pdf / computer_vision.pdf)*
4. "How are word embeddings related to retrieval-augmented generation?" *(nlp.pdf)*
5. "What is the difference between object detection and image classification?" *(computer_vision.pdf)*
6. "How does deep learning relate to natural language processing and computer vision?" *(cross-document: deep_learning.pdf, nlp.pdf, computer_vision.pdf)*

## Unanswerable Question Example

```
You: What is the current population of Japan?

Answer:
I couldn't find this information in the provided documents.
```

This works because (a) the retrieved chunks, if any, will be far from the
question in embedding space and get flagged by the weak-retrieval check in
`src/retriever.py`, and (b) even if a chunk is retrieved, the system prompt
in `src/generator.py` instructs the model to refuse rather than answer from
outside knowledge.

## Known Limitations

- No OCR: scanned, image-only PDFs with no text layer will fail ingestion
  with a clear error rather than being silently skipped.
- Retrieval quality is inherently bounded by the chosen embedding model —
  this is a general property of embedding-based retrieval, not specific to
  this implementation.
- Chunk size/overlap are fixed defaults tuned for explanatory prose; other
  content types may need different values.
- The weak-retrieval distance threshold (`MAX_RELEVANT_DISTANCE`) is a
  configurable heuristic, not a scientifically derived cutoff, and its ideal
  value can vary with embedding model and corpus.
- No re-ranking step is implemented on top of raw vector similarity.
- No conversational memory — each question is answered independently.
- The system depends on an external API (OpenAI) for both embeddings and
  generation; it does not run fully offline.
- Citations are at page/chunk granularity, not exact sentence or paragraph
  level.

## Future Improvements

- Add a re-ranking step (e.g. a cross-encoder) after initial vector
  retrieval to improve precision.
- Add hybrid search (combining keyword/BM25 search with vector search).
- Add OCR support (e.g. Tesseract) for scanned PDFs.
- Support additional document formats (.docx, .txt, .html).
- Build a small evaluation dataset of question/expected-answer pairs and an
  automated retrieval/answer-quality evaluation script.
- Add conversational memory for multi-turn follow-up questions.

## Assignment Requirement Checklist

| Requirement | Implementation | Status |
|---|---|---|
| 4–5 documents | `data/documents/` (5 PDFs) | ✓ |
| PDF support | `src/ingestion.py` | ✓ |
| Chunking | `src/chunking.py` | ✓ |
| Chunk overlap | `src/chunking.py` (`CHUNK_OVERLAP`) | ✓ |
| Metadata (source, page, chunk id) | `src/ingestion.py`, `src/chunking.py` | ✓ |
| Batched embeddings | `src/embeddings.py` (`embed_texts`, batched) | ✓ |
| Vector database | ChromaDB (`src/vector_store.py`) | ✓ |
| Persistent storage | `chroma_db/` via `PersistentClient` | ✓ |
| Separate indexing/querying | `index.py` vs `app.py` | ✓ |
| Top-k retrieval | `src/retriever.py` (`TOP_K`) | ✓ |
| Grounded generation | `src/generator.py` (strict system prompt) | ✓ |
| Unknown-question handling | `src/retriever.py` weak-match check + `src/generator.py` prompt | ✓ |
| Citations from real metadata | `src/qa.py` (`QAResult.citations`) | ✓ |
| CLI | `app.py` | ✓ |
| Source chunks displayed | `app.py` / `streamlit_app.py` | ✓ |
| Optional Streamlit UI | `streamlit_app.py` | ✓ |
| README | `README.md` | ✓ |
| Tests | `tests/` (chunking, ingestion, retrieval) | ✓ (run manually here; see Testing Results) |
| `.env.example` / no committed secrets | `.env.example`, `.gitignore` | ✓ |

## Testing Results (what was actually run, and where)

This project was developed in an offline sandbox with no internet access, so
`openai`, `chromadb`, `pymupdf`, `streamlit`, and `pytest` could not be
installed or executed there. What **was** actually run and verified in that
environment:

- `data/build_pdfs.py` was executed and produced the 5 real, multi-page PDFs
  in `data/documents/`, confirmed via `pypdf` to have 3 pages each.
- `src/ingestion.py` and `src/chunking.py` were run end-to-end against the
  real sample PDFs (using the `pypdf` fallback path, since `pymupdf` was
  unavailable offline): this produced 15 pages and 78 chunks with the default
  `CHUNK_SIZE=800` / `CHUNK_OVERLAP=150`, with correct source/page metadata
  on every chunk.
- All 19 unit tests in `tests/` (6 chunking, 7 ingestion, 6 retrieval) were
  executed and passed, using a lightweight fake vector store and a fake
  embedding function for the retrieval tests, so no API key was required.
- `python index.py` was run and correctly reported a clear, actionable error
  (`OPENAI_API_KEY is not set...`) when no key was configured — verifying the
  missing-API-key error path.
- `python app.py` was run and correctly reported a clear, actionable error
  (`Vector database not found... Run 'python index.py' first`) when queried
  before indexing — verifying the "query before indexing" error path.

**Not run in this sandbox** (requires packages/network unavailable here —
you should run these yourself after `pip install -r requirements.txt` and
setting `OPENAI_API_KEY`): the actual `openai.embeddings.create` /
`chat.completions.create` calls, the real ChromaDB persistence round-trip, a
full `python index.py` run to completion, a full `python app.py` Q&A session,
and the Streamlit UI. The code for these paths was written directly against
the documented current OpenAI Python SDK (`OpenAI().embeddings.create`,
`OpenAI().chat.completions.create`) and ChromaDB SDK
(`chromadb.PersistentClient`, `collection.add`/`collection.query`) and
exercised through mocks in the test suite, but you should run the checks
below yourself once you have an API key.

### Checks to run yourself after adding an API key

```bash
python --version                 # confirm 3.11+
pip install -r requirements.txt
pytest                           # 19 tests should pass
python index.py                  # should report 5 docs, ~78 chunks, and store them
python app.py                    # ask the 5+ example questions above
# ask the "population of Japan" question -> should refuse
# stop the process, run `python app.py` again -> should NOT need to re-index
```

## Demo Instructions (3–8 minutes)

1. `pip install -r requirements.txt`, set `OPENAI_API_KEY` in `.env`. (30s)
2. Run `python index.py` — narrate the printed progress: documents found,
   pages extracted, chunks created, embedding batches, chunks stored. (1 min)
3. Run `python app.py` and ask 2–3 of the example questions above, pointing
   out the Answer, Sources, and Retrieved Context sections. (2–3 min)
4. Ask the unanswerable "population of Japan" question and show the refusal
   message. (1 min)
5. Stop and restart `python app.py` to show the vector database persisted
   without re-indexing. (30s)
6. (Optional) Run `streamlit run streamlit_app.py` to show the same pipeline
   in a browser. (1–2 min)
7. Briefly point at `src/` and describe each module's single responsibility
   using the architecture diagram above. (1–2 min)

## Interview Explanation

This is a Retrieval-Augmented Generation system: instead of fine-tuning or
hoping the LLM already "knows" the answer, we give the LLM the exact relevant
text at question time. Indexing (`index.py`) is a one-time (or on-demand)
batch job: PDFs are parsed page by page, cleaned of boilerplate, split into
overlapping ~800-character chunks that keep each chunk's source and page
number, embedded in batches through OpenAI's embedding API, and stored in
ChromaDB, a vector database that supports fast approximate nearest-neighbor
search over those embeddings. Querying (`app.py`) is a separate, lightweight
process: the user's question is embedded with the same model, ChromaDB
returns the top-k most similar chunks, and those chunks — not the whole
document set — are inserted into a tightly constrained system prompt that
tells the LLM to answer only from that context and to say so explicitly if
the answer isn't there. Because indexing and querying are separate, the app
starts instantly and never re-embeds the whole corpus per question. Because
citations are built from the chunk metadata that was actually retrieved,
rather than asked of the LLM, they can't be hallucinated. The main tradeoffs
I'd highlight if pushed further: chunking strategy affects both retrieval
quality and cost, the embedding model bounds retrieval quality, and there's
no re-ranking or hybrid search layer yet — those are the most natural next
improvements.
#   Q - A B o t - R A G - P i p e l i n e -  
 