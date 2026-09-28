"""
Optional Streamlit web UI for the Document Q&A Bot.

Usage:
    streamlit run streamlit_app.py

This is an alternative front end on top of the exact same retriever/generator
pipeline used by app.py (the CLI). It does not replace the CLI, and it does
not re-implement any RAG logic -- it only calls into src/qa.py.
"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.generator import GenerationError
from src.qa import answer_question
from src.retriever import Retriever
from src.vector_store import VectorStoreError, load_existing_store

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")
st.title("📄 Document Q&A Bot")
st.caption("Retrieval-Augmented Generation over your local document collection.")


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    store = load_existing_store()
    return Retriever(store)


if settings.embedding_backend == "openai" or settings.generation_backend == "openai":
    try:
        settings.require_api_key()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

try:
    retriever = get_retriever()
except VectorStoreError as exc:
    st.error(str(exc))
    st.info("Run `python index.py` in a terminal first, then reload this page.")
    st.stop()

question = st.text_input("Ask a question about the indexed documents:")

if st.button("Ask") or question:
    if not question or not question.strip():
        st.warning("Please enter a non-empty question.")
    else:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                result = answer_question(retriever, question)
            except (VectorStoreError, GenerationError) as exc:
                st.error(str(exc))
                result = None

        if result is not None:
            st.subheader("Answer")
            st.write(result.answer)

            if result.chunks:
                st.subheader("Sources")
                for i, citation in enumerate(result.citations, start=1):
                    st.write(f"{i}. {citation}")
            else:
                st.info("No closely matching sources were found.")
