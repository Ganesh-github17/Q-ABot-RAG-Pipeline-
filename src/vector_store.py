"""
Persistent vector storage using ChromaDB.

The database persists to disk under settings.chroma_path (default:
./chroma_db) using ChromaDB's PersistentClient, so indexing (index.py) and
querying (app.py) are fully separate processes: indexing writes to disk once,
and querying simply opens the existing database without recreating it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking import Chunk
from src.config import settings


class VectorStoreError(Exception):
    """Raised when the vector store cannot be created, opened, or queried."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    distance: float | None  # smaller = more similar (Chroma's default metric)


def _get_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise VectorStoreError(
            "The 'chromadb' package is not installed. Run "
            "'pip install -r requirements.txt'."
        ) from exc
    return chromadb


class VectorStore:
    """Thin wrapper around a single persistent ChromaDB collection."""

    def __init__(self, path: str | None = None, collection_name: str | None = None):
        self.path = path or settings.chroma_path
        self.collection_name = collection_name or settings.collection_name
        chromadb = _get_chromadb()
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0

    def reset(self) -> None:
        """Delete all vectors in the collection (used before re-indexing)."""
        chromadb = _get_chromadb()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "Number of chunks and embeddings must match "
                f"({len(chunks)} vs {len(embeddings)})."
            )
        if not chunks:
            return

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "page": c.page, "chunk_id": c.chunk_id}
                for c in chunks
            ],
        )

    def query(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector store query failed: {exc}") from exc

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[None] * len(documents)])[0]

        retrieved = []
        for doc_text, meta, distance in zip(documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    text=doc_text,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", -1),
                    chunk_id=meta.get("chunk_id", ""),
                    distance=distance,
                )
            )
        return retrieved


def load_existing_store() -> VectorStore:
    """
    Open the existing persistent vector store for querying, raising a clear
    error if it does not exist yet or is empty (i.e. indexing was never run).
    """
    import os

    if not os.path.isdir(settings.chroma_path):
        raise VectorStoreError(
            f"Vector database not found at '{settings.chroma_path}'. "
            "Run 'python index.py' first to index your documents."
        )

    store = VectorStore()
    if store.is_empty():
        raise VectorStoreError(
            "The vector database exists but is empty. Run 'python index.py' "
            "to index your documents before asking questions."
        )
    return store
