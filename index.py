#!/usr/bin/env python3
"""
Indexing workflow for the Document Q&A Bot.

Usage:
    python index.py            # index only if the store is empty
    python index.py --force    # wipe and rebuild the vector store from scratch

Pipeline: discover documents -> extract text -> clean -> chunk -> embed
(batched) -> store in persistent ChromaDB.
"""

from __future__ import annotations

import sys

from src.chunking import chunk_pages
from src.config import settings
from src.embeddings import EmbeddingError, embed_texts
from src.ingestion import IngestionError, discover_documents, load_documents
from src.vector_store import VectorStore, VectorStoreError

BANNER = "=" * 40


def main() -> int:
    force = any(flag in sys.argv for flag in ("--force", "--reset", "--rebuild"))

    print(BANNER)
    print("DOCUMENT INDEXING")
    print(BANNER)
    print()

    if settings.embedding_backend == "openai":
        try:
            settings.require_api_key()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

    # --- 1. Discover documents ------------------------------------------------
    try:
        doc_paths = discover_documents(settings.data_dir)
    except IngestionError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Found {len(doc_paths)} documents.\n")
    print("Loading:")
    for path in doc_paths:
        print(f"- {path.name}")
    print()

    # --- 2. Extract + clean text ------------------------------------------------
    try:
        pages = load_documents(settings.data_dir)
    except IngestionError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Extracted text successfully ({len(pages)} pages).\n")

    # --- 3. Open the vector store early (fail fast on storage issues) -------
    try:
        store = VectorStore()
    except VectorStoreError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not store.is_empty() and not force:
        print(
            f"Vector database already contains {store.count()} chunks at "
            f"'{settings.chroma_path}'."
        )
        print("Skipping re-indexing. Run 'python index.py --reset' or 'python index.py --rebuild' to rebuild.")
        return 0

    if force and not store.is_empty():
        print("Reset/rebuild flag specified: clearing existing vector database.\n")
        store.reset()

    # --- 4. Chunk ------------------------------------------------------------
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    print(f"Created {len(chunks)} chunks "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}).\n")

    if not chunks:
        print("ERROR: No chunks were produced from the documents.")
        return 1

    # --- 5. Embed (batched) ------------------------------------------------
    print("Generating embeddings...")
    texts = [c.text for c in chunks]

    def on_batch_start(batch_index: int, total: int) -> None:
        print(f"Batch {batch_index}/{total}")

    try:
        embeddings = embed_texts(
            texts,
            batch_size=settings.embedding_batch_size,
            on_batch_start=on_batch_start,
        )
    except EmbeddingError as exc:
        print(f"\nERROR: {exc}")
        return 1

    print()

    # --- 6. Store ------------------------------------------------------------
    print("Storing vectors in ChromaDB...")
    try:
        store.add_chunks(chunks, embeddings)
    except VectorStoreError as exc:
        print(f"ERROR: {exc}")
        return 1

    print()
    print("Indexing complete.")
    print()
    print(f"Documents: {len(doc_paths)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector database: {settings.chroma_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
