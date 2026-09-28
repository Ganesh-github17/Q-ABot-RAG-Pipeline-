"""
Answer generation.

The default path is a local extractive answer builder so the project works
without any paid API usage. An OpenAI-backed path remains available as an
opt-in via GENERATION_BACKEND=openai.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.local_models import extractive_answer, normalize_answer
from src.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer the user's question using ONLY the information in the provided
document context below. Do not use any outside knowledge, and do not rely on
anything you know from your training data.

Rules you must follow:
- If the answer is present in the context, answer clearly and concisely,
  and mention which source(s) support your answer using the format
  (source: filename - Page X).
- If the answer cannot be found in the provided context, respond with
  exactly: "I couldn't find this information in the provided documents."
  Do not guess, do not partially answer, and do not fabricate facts.
- Never invent a source filename or page number that is not present in the
  context you were given.
- Keep answers focused and avoid restating the entire context.
"""


class GenerationError(Exception):
    """Raised when the LLM call fails."""


NO_ANSWER_MESSAGE = "I couldn't find this information in the provided documents."


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {chunk.source} - Page {chunk.page}\n{chunk.text}")
    return "\n\n".join(parts)


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise GenerationError(
            "The 'openai' package is not installed. Run "
            "'pip install -r requirements.txt'."
        ) from exc
    api_key = settings.require_api_key()
    return OpenAI(api_key=api_key)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    model: str | None = None,
    client=None,
) -> str:
    """
    Generate a grounded answer to `question` using only `chunks` as context.

    If `chunks` is empty, the LLM is not even called -- there is nothing to
    ground an answer in, so the standard "not found" message is returned
    directly. This avoids relying solely on prompt-following behavior for the
    no-context case.
    """
    if not chunks:
        return NO_ANSWER_MESSAGE

    if settings.generation_backend != "openai":
        return normalize_answer(
            extractive_answer(
                question,
                chunks,
                max_sentences=settings.local_answer_sentence_limit,
            )
        )

    model = model or settings.llm_model
    client = client or get_openai_client()

    context = _format_context(chunks)
    user_message = (
        f"Document context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the document context above."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
    except Exception as exc:  # network / API errors from the SDK
        raise GenerationError(f"LLM generation failed: {exc}") from exc

    answer = response.choices[0].message.content
    return normalize_answer(answer) or NO_ANSWER_MESSAGE
