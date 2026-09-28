#!/usr/bin/env python3
"""
Interactive CLI for the Document Q&A Bot.

Usage:
    python app.py

Loads the existing persistent ChromaDB vector store (built by index.py) and
starts a question-answering loop. Type 'exit' or 'quit' to stop.
"""

from __future__ import annotations

from src.config import settings
from src.generator import GenerationError
from src.qa import QAResult, answer_question
from src.retriever import Retriever
from src.vector_store import VectorStoreError, load_existing_store

BANNER = "=" * 40
EXIT_COMMANDS = {"exit", "quit"}


def print_result(result: QAResult) -> None:
    print("\nAnswer:")
    print(result.answer)

    if result.chunks:
        print("\nSources:")
        for i, citation in enumerate(result.citations, start=1):
            print(f"{i}. {citation}")
    else:
        print("\nSources searched: none matched closely enough to cite.")


def main() -> int:
    print(BANNER)
    print("DOCUMENT Q&A BOT")
    print(BANNER)
    print()

    if settings.embedding_backend == "openai" or settings.generation_backend == "openai":
        try:
            settings.require_api_key()
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

    try:
        store = load_existing_store()
    except VectorStoreError as exc:
        print(f"ERROR: {exc}")
        return 1

    retriever = Retriever(store)

    print("Ask a question.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            print("You: ", end="", flush=True)
            question = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if question.lower().startswith("you:"):
            question = question[4:].strip()

        if not question:
            print("Please enter a non-empty question.\n")
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            return 0

        try:
            result = answer_question(retriever, question)
        except ValueError as exc:
            print(f"ERROR: {exc}\n")
            continue
        except (VectorStoreError, GenerationError) as exc:
            print(f"ERROR: {exc}\n")
            continue

        print_result(result)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
