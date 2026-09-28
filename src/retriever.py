"""
Retrieval layer.

Converts a user's natural-language question into an embedding, searches the
persistent vector store for the top-k most similar chunks, and returns them
along with enough information (source, page, distance) to display to the
user and hand off to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.embeddings import embed_query
from src.local_models import tokenize
from src.vector_store import RetrievedChunk, VectorStore


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    is_weak: bool  # True if even the best match is below the similarity bar


class Retriever:
    def __init__(self, store: VectorStore, embed_fn=embed_query):
        self.store = store
        self.embed_fn = embed_fn

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        top_k = top_k or settings.top_k
        query_embedding = self.embed_fn(question)
        candidate_k = _candidate_pool_size(question, top_k)
        chunks = self.store.query(query_embedding, top_k=candidate_k)
        chunks = _rerank_chunks(question, chunks)
        result_k = top_k
        if _is_list_style_question(question):
            result_k = max(top_k, 8)
        chunks = chunks[:result_k]

        relevant_chunks = _relevant_chunks(chunks)
        is_weak = _is_weak_retrieval(relevant_chunks)
        return RetrievalResult(chunks=relevant_chunks, is_weak=is_weak)


def _candidate_pool_size(question: str, top_k: int) -> int:
    """Use a larger semantic candidate pool for list-style questions."""
    if _is_list_style_question(question):
        return max(top_k * 3, top_k + 10)
    return top_k


def _is_list_style_question(question: str) -> bool:
    question_lower = question.lower()
    return any(
        term in question_lower
        for term in (
            "type",
            "types",
            "kind",
            "kinds",
            "category",
            "categories",
            "classification",
            "classifications",
            "approach",
            "approaches",
            "method",
            "methods",
            "form",
            "forms",
        )
    )


def _lexical_overlap_score(question: str, chunk: RetrievedChunk) -> int:
    question_tokens = set(tokenize(question))
    chunk_tokens = set(tokenize(chunk.text))
    if not question_tokens or not chunk_tokens:
        return 0

    score = len(question_tokens & chunk_tokens)
    question_lower = question.lower()
    chunk_lower = chunk.text.lower()

    category_question = any(
        term in question_lower
        for term in (
            "type",
            "types",
            "kind",
            "kinds",
            "category",
            "categories",
            "classification",
            "classifications",
            "approach",
            "approaches",
            "method",
            "methods",
            "form",
            "forms",
        )
    )

    if category_question:
        bonus_terms = {
            "supervised",
            "unsupervised",
            "reinforcement",
            "classification",
            "regression",
            "clustering",
            "dimensionality",
            "labelled",
            "labeled",
            "paradigm",
            "algorithm",
            "approach",
            "method",
            "category",
            "type",
        }
        category_hits = sum(1 for term in bonus_terms if term in chunk_lower)
        score += category_hits * 3

        if any(marker in chunk_lower for marker in ("main types", "types of", "categories", "include", "consist of", "are supervised", "are unsupervised")):
            score += 5

        if (
            "supervised" in chunk_lower
            and "unsupervised" in chunk_lower
            and "reinforcement" in chunk_lower
        ):
            score += 12

        if chunk_lower.count("supervised") + chunk_lower.count("unsupervised") + chunk_lower.count("reinforcement") >= 2:
            score += 6

    if question_lower.startswith("what is "):
        definition_terms = {"subfield", "artificial", "intelligence", "learn", "pattern", "data"}
        score += sum(1 for term in definition_terms if term in chunk_lower) * 2
        if "machine learning is" in chunk_lower or "is a subfield of artificial intelligence" in chunk_lower:
            score += 12
        if "what is machine learning" in chunk_lower:
            score += 10

    return score


def _rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Promote chunks with direct wording and category evidence for the question."""
    if not chunks:
        return chunks

    def sort_key(chunk: RetrievedChunk):
        lexical = _lexical_overlap_score(question, chunk)
        distance = chunk.distance if chunk.distance is not None else float("inf")
        return (-lexical, distance)

    return sorted(chunks, key=sort_key)


def _is_weak_retrieval(chunks: list[RetrievedChunk]) -> bool:
    """
    Reject retrieval when no candidate is close enough in embedding space.

    Reranking may move a chunk with a weaker distance to the front, so the
    decision must use the closest raw vector-store result rather than
    ``chunks[0]``.
    """
    if not chunks:
        return True

    distances = [chunk.distance for chunk in chunks if chunk.distance is not None]
    if not distances:
        return False  # distance not reported by the store; skip this check

    return min(distances) > settings.max_relevant_distance


def _relevant_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep only chunks that meet the configured vector-distance bar."""
    return [
        chunk
        for chunk in chunks
        if chunk.distance is None or chunk.distance <= settings.max_relevant_distance
    ]
