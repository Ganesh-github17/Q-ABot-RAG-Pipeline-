"""
QA orchestration layer.

Ties together the retriever and generator into a single `answer_question`
call, and formats deterministic citations directly from retrieved chunk
metadata (never from the LLM's own output), satisfying the requirement that
citations cannot be fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.generator import NO_ANSWER_MESSAGE, generate_answer
from src.retriever import Retriever
from src.vector_store import RetrievedChunk


def _best_source_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep only the highest-confidence source to avoid mixing answers across PDFs."""
    if not chunks:
        return []
    best_source = chunks[0].source
    return [chunk for chunk in chunks if chunk.source == best_source]


@dataclass
class QAResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    is_weak: bool

    @property
    def citations(self) -> list[str]:
        """Deterministic, de-duplicated citations built from chunk metadata."""
        seen = set()
        citations = []
        for chunk in self.chunks:
            key = (chunk.source, chunk.page)
            if key not in seen:
                seen.add(key)
                citations.append(f"{chunk.source} - Page {chunk.page}")
        return citations


def answer_question(retriever: Retriever, question: str, top_k: int | None = None) -> QAResult:
    """Run the full retrieve -> generate pipeline for a single question."""
    result = retriever.retrieve(question, top_k=top_k)
    grounded_chunks = _best_source_chunks(result.chunks)

    if result.is_weak:
        # Weak retrieval: refuse without even calling the LLM, so a clearly
        # unrelated question cannot accidentally be answered from the model's
        # own pretrained knowledge.
        return QAResult(
            question=question,
            answer=NO_ANSWER_MESSAGE,
            chunks=[],
            is_weak=True,
        )

    answer = generate_answer(question, grounded_chunks)
    if answer == NO_ANSWER_MESSAGE:
        return QAResult(question=question, answer=answer, chunks=[], is_weak=False)

    return QAResult(question=question, answer=answer, chunks=grounded_chunks, is_weak=False)
