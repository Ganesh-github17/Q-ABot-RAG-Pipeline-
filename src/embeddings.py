"""
Embedding generation.

The default path is a deterministic, dependency-free local embedding model so
the project works without any paid API usage. An OpenAI-backed path remains
available as an opt-in via EMBEDDING_BACKEND=openai.
"""

from __future__ import annotations

from typing import Iterable

from src.config import settings
from src.local_models import hashed_embedding


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_openai_client():
    """Create an OpenAI client, raising a clear error if misconfigured."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise EmbeddingError(
            "The 'openai' package is not installed. Run "
            "'pip install -r requirements.txt'."
        ) from exc

    api_key = settings.require_api_key()
    return OpenAI(api_key=api_key)


def _embed_texts_local(
    texts: list[str],
    *,
    batch_size: int,
    on_batch_start=None,
) -> list[list[float]]:
    batches = list(_batched(texts, batch_size))
    embeddings: list[list[float]] = []

    for batch_index, batch in enumerate(batches, start=1):
        if on_batch_start:
            on_batch_start(batch_index, len(batches))
        embeddings.extend(
            hashed_embedding(text, settings.local_embedding_dimensions) for text in batch
        )

    return embeddings


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int | None = None,
    client=None,
    on_batch_start=None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts, batching requests to the OpenAI
    embeddings API.

    Parameters
    ----------
    texts: the input strings to embed (e.g. chunk texts, or a single query).
    model: embedding model name; defaults to settings.embedding_model.
    batch_size: number of texts per API call; defaults to
        settings.embedding_batch_size.
    client: an already-constructed OpenAI client (used for dependency
        injection in tests); a real client is created automatically if
        omitted.
    on_batch_start: optional callback(batch_index, total_batches) used to
        print progress during indexing.

    Returns
    -------
    A list of embedding vectors, in the same order as `texts`.
    """
    if not texts:
        return []

    batch_size = batch_size or settings.embedding_batch_size

    if settings.embedding_backend != "openai":
        return _embed_texts_local(texts, batch_size=batch_size, on_batch_start=on_batch_start)

    model = model or settings.embedding_model
    client = client or get_openai_client()

    batches = list(_batched(texts, batch_size))
    all_embeddings: list[list[float]] = []

    for batch_index, batch in enumerate(batches, start=1):
        if on_batch_start:
            on_batch_start(batch_index, len(batches))
        try:
            response = client.embeddings.create(model=model, input=batch)
        except Exception as exc:  # network / API errors from the SDK
            raise EmbeddingError(
                f"Embedding API call failed on batch {batch_index}/{len(batches)}: {exc}"
            ) from exc

        # response.data is guaranteed to be in the same order as the input.
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings


def embed_query(text: str, *, model: str | None = None, client=None) -> list[float]:
    """Convenience wrapper to embed a single query string."""
    embeddings = embed_texts([text], model=model, batch_size=1, client=client)
    return embeddings[0]
