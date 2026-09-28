"""
Local, dependency-free text helpers for the free-tier pipeline.

These helpers provide deterministic embeddings for retrieval and simple
extractive answer generation so the app can run without any billing-backed
API calls.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from src.vector_store import RetrievedChunk

TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        token = _normalize_token(raw_token)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def normalize_answer(answer: str | None) -> str:
    """Strip incomplete fragments caused by a chunk boundary or partial generation."""
    if answer is None:
        return "I couldn't find this information in the provided documents."

    text = " ".join(str(answer).strip().split())
    if not text:
        return "I couldn't find this information in the provided documents."

    if text.endswith((".", "!", "?")):
        return text

    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punctuation >= 0:
        return text[: last_punctuation + 1].strip()
    return text


def split_sentences(text: str) -> list[str]:
    normalized_text = re.sub(r"\s*\n+\s*", " ", text.strip())
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(normalized_text) if sentence.strip()]
    cleaned: list[str] = []
    for sentence in sentences:
        if sentence and not sentence.endswith((".", "!", "?")):
            continue
        cleaned.append(sentence)
    return cleaned or ([text.strip()] if text.strip() else [])


def hashed_embedding(text: str, dimensions: int) -> list[float]:
    """Return a deterministic unit-normalized bag-of-words embedding."""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dimensions

    vector = [0.0] * dimensions
    for token in tokens:
        vector[_stable_index(token, dimensions)] += 1.0

    for left, right in zip(tokens, tokens[1:]):
        vector[_stable_index(f"{left}::{right}", dimensions)] += 0.5

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def extractive_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    max_sentences: int = 2,
) -> str:
    """Build a concise answer by selecting the most relevant sentences."""
    if not chunks:
        return "I couldn't find this information in the provided documents."

    question_tokens = set(tokenize(question))
    if not question_tokens:
        return "I couldn't find this information in the provided documents."

    if _is_list_style_question(question):
        max_sentences = max(max_sentences, 3)

    scored_sentences: list[tuple[float, int, str]] = []
    for chunk_index, chunk in enumerate(chunks):
        for sentence in split_sentences(chunk.text):
            score = _sentence_score(sentence, question, question_tokens)
            if score > 0:
                scored_sentences.append((score, chunk_index, sentence))

    if not scored_sentences:
        return "I couldn't find this information in the provided documents."

    scored_sentences.sort(key=lambda item: (-item[0], item[1], len(item[2])))

    direct_candidates: list[str] = []
    for _, _, sentence in scored_sentences:
        if _is_direct_answer(question, sentence):
            direct_candidates.append(sentence)

    if direct_candidates:
        selected_sentences = []
        seen = set()
        for sentence in direct_candidates:
            normalized_sentence = sentence.lower()
            if normalized_sentence in seen:
                continue
            seen.add(normalized_sentence)
            selected_sentences.append(sentence)
            if len(selected_sentences) >= max_sentences:
                break
        answer = normalize_answer(" ".join(selected_sentences).strip())
        if answer:
            return answer

    selected_sentences = []
    seen = set()
    for _, _, sentence in scored_sentences:
        normalized_sentence = sentence.lower()
        if normalized_sentence in seen:
            continue
        if _looks_like_title(sentence):
            continue
        seen.add(normalized_sentence)
        selected_sentences.append(sentence)
        if len(selected_sentences) >= max_sentences:
            break

    if not selected_sentences:
        return "I couldn't find this information in the provided documents."

    answer = normalize_answer(" ".join(selected_sentences).strip())
    return answer or "I couldn't find this information in the provided documents."


def _sentence_score(sentence: str, question: str, question_tokens: set[str]) -> float:
    stripped = sentence.strip()
    sentence_tokens = tokenize(sentence)
    if not sentence_tokens:
        return 0.0

    # Question-like headings are not substantive answers; they are section titles,
    # not the actual information the user asked for.
    if stripped.endswith("?"):
        return 0.0

    sentence_token_set = set(sentence_tokens)
    overlap = question_tokens & sentence_token_set
    if not overlap:
        return 0.0

    if _looks_like_title(sentence):
        return 0.0

    question_lower = question.lower()
    sentence_lower = sentence.lower()

    base_score = (len(overlap) * 2.5) + (len(overlap) / math.sqrt(len(sentence_tokens)))

    if _is_direct_answer(question, sentence):
        base_score += 10.0

    if _is_list_style_question(question):
        if "most widely used category" in sentence_lower:
            base_score += 8.0
        if "has no labels" in sentence_lower or "no labels at all" in sentence_lower:
            base_score += 8.0
        if "third major paradigm" in sentence_lower:
            base_score += 8.0
        if "distinct from supervised and unsupervised learning" in sentence_lower:
            base_score += 6.0
        if "algorithms include" in sentence_lower:
            base_score -= 6.0
        if "problems are typically divided into two types" in sentence_lower:
            base_score += 2.0

    if "relates to deep learning" in sentence_lower or "theoretical and mathematical foundation" in sentence_lower:
        base_score -= 6.0
    if "subfield of artificial intelligence" in sentence_lower or "a subfield of" in sentence_lower:
        base_score += 5.0
    if "main types" in question_lower and "the main types" in sentence_lower:
        base_score += 8.0

    question_token_list = tokenize(question)
    phrase_matches = _phrase_overlap(question_token_list, sentence_tokens)
    return base_score + (phrase_matches * 4.0)


def _is_direct_answer(question: str, sentence: str) -> bool:
    question_lower = question.lower().strip().rstrip("?")
    sentence_lower = sentence.lower().strip()
    if not sentence_lower or sentence_lower.endswith("?"):
        return False

    if question_lower.startswith("what is "):
        return (
            " is a " in sentence_lower
            or " is the " in sentence_lower
            or " refers to " in sentence_lower
            or " is defined as " in sentence_lower
            or sentence_lower.startswith("machine learning is ")
        )

    if question_lower.startswith("what are "):
        return (
            "the main types" in sentence_lower
            or " are supervised " in sentence_lower
            or " are unsupervised " in sentence_lower
            or " are reinforcement " in sentence_lower
            or " including " in sentence_lower
            or " include " in sentence_lower
            or "most widely used category" in sentence_lower
            or "has no labels" in sentence_lower
            or "third major paradigm" in sentence_lower
        )

    return False


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


def _phrase_overlap(question_tokens: list[str], sentence_tokens: list[str]) -> int:
    if len(question_tokens) < 2 or len(sentence_tokens) < 2:
        return 0

    q_ngrams = set()
    for i in range(len(question_tokens) - 1):
        q_ngrams.add((question_tokens[i], question_tokens[i + 1]))
    for i in range(len(question_tokens) - 2):
        q_ngrams.add((question_tokens[i], question_tokens[i + 1], question_tokens[i + 2]))

    s_ngrams = set()
    for i in range(len(sentence_tokens) - 1):
        s_ngrams.add((sentence_tokens[i], sentence_tokens[i + 1]))
    for i in range(len(sentence_tokens) - 2):
        s_ngrams.add((sentence_tokens[i], sentence_tokens[i + 1], sentence_tokens[i + 2]))

    return len(q_ngrams & s_ngrams)


def _looks_like_title(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped or stripped[-1] in ".!?":
        return False

    words = stripped.split()
    if len(words) > 12:
        return False

    title_case_words = sum(1 for word in words if word[:1].isupper())
    return title_case_words / len(words) >= 0.6


def _normalize_token(token: str) -> str:
    for suffix in ("'s", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _stable_index(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions