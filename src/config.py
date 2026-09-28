"""
Centralized configuration for the Document Q&A Bot.

All tunable parameters are read from environment variables (via a .env file
loaded with python-dotenv) so that nothing important is hard-coded across the
codebase. Sensible defaults are provided for local, no-billing operation.
OpenAI-backed behavior remains opt-in via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name}='{raw}' is not a valid integer."
        ) from None


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name}='{raw}' is not a valid float."
        ) from None


def _default_collection_name() -> str:
    backend = _get_str("EMBEDDING_BACKEND", "local").lower()
    return f"document_qa_bot_{backend}"


@dataclass(frozen=True)
class Settings:
    # --- API credentials -------------------------------------------------
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # --- Runtime mode ----------------------------------------------------
    embedding_backend: str = _get_str("EMBEDDING_BACKEND", "local").lower()
    generation_backend: str = _get_str("GENERATION_BACKEND", "local").lower()

    # --- Models ------------------------------------------------------------
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Local fallback settings ------------------------------------------
    local_embedding_dimensions: int = _get_int("LOCAL_EMBEDDING_DIMENSIONS", 512)
    local_answer_sentence_limit: int = _get_int("LOCAL_ANSWER_SENTENCE_LIMIT", 2)

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)

    # --- Retrieval ------------------------------------------------------------
    top_k: int = _get_int("TOP_K", 5)
    # ChromaDB is configured with cosine distance, where smaller is more
    # similar. The default was chosen from the observed gap between the
    # indexed document questions and out-of-scope probes; override it with
    # MAX_RELEVANT_DISTANCE when the document collection changes.
    max_relevant_distance: float = _get_float("MAX_RELEVANT_DISTANCE", 0.80)

    # --- Embeddings ------------------------------------------------------------
    embedding_batch_size: int = _get_int("EMBEDDING_BATCH_SIZE", 32)

    # --- Storage ------------------------------------------------------------
    chroma_path: str = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", _default_collection_name())
    data_dir: str = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "documents"))

    def require_api_key(self) -> str:
        """Return the OpenAI API key or raise a clear, actionable error."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Create a .env file (copy "
                ".env.example to .env) and set OPENAI_API_KEY=<your key>, or "
                "export it as an environment variable before running this "
                "application."
            )
        return self.openai_api_key


settings = Settings()
