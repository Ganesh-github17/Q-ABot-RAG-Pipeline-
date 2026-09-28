import pytest

from src.local_models import extractive_answer, split_sentences
from src.qa import answer_question
from src.retriever import Retriever
from src.vector_store import RetrievedChunk


class FakeVectorStore:
    """A minimal stand-in for VectorStore that requires no real ChromaDB."""

    def __init__(self, chunks_to_return):
        self._chunks_to_return = chunks_to_return
        self.last_query_embedding = None
        self.last_top_k = None

    def query(self, query_embedding, top_k):
        self.last_query_embedding = query_embedding
        self.last_top_k = top_k
        return self._chunks_to_return[:top_k]


def fake_embed_fn(text: str):
    """Deterministic fake embedding function, avoids any real API call."""
    return [float(len(text))]


def test_retriever_returns_chunks_from_store():
    chunks = [
        RetrievedChunk(text="ML is...", source="machine_learning.pdf", page=2, chunk_id="c1", distance=0.1),
        RetrievedChunk(text="DL is...", source="deep_learning.pdf", page=1, chunk_id="c2", distance=0.2),
    ]
    store = FakeVectorStore(chunks)
    retriever = Retriever(store, embed_fn=fake_embed_fn)

    result = retriever.retrieve("What is machine learning?", top_k=2)

    assert result.chunks == chunks
    assert result.is_weak is False
    assert store.last_top_k == 2
    assert store.last_query_embedding == fake_embed_fn("What is machine learning?")


def test_retriever_uses_configured_top_k_by_default():
    from src.config import settings

    chunks = [
        RetrievedChunk(text=f"chunk {i}", source="doc.pdf", page=i, chunk_id=str(i), distance=0.1)
        for i in range(10)
    ]
    store = FakeVectorStore(chunks)
    retriever = Retriever(store, embed_fn=fake_embed_fn)

    retriever.retrieve("some question")
    assert store.last_top_k == settings.top_k


def test_retriever_rejects_empty_question():
    store = FakeVectorStore([])
    retriever = Retriever(store, embed_fn=fake_embed_fn)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_weak_retrieval_flagged_when_no_chunks_found():
    store = FakeVectorStore([])
    retriever = Retriever(store, embed_fn=fake_embed_fn)
    result = retriever.retrieve("What is the population of Japan?")
    assert result.is_weak is True
    assert result.chunks == []


def test_weak_retrieval_flagged_when_distance_above_threshold():
    from src.config import settings

    far_chunk = RetrievedChunk(
        text="irrelevant",
        source="doc.pdf",
        page=1,
        chunk_id="x",
        distance=settings.max_relevant_distance + 1.0,
    )
    store = FakeVectorStore([far_chunk])
    retriever = Retriever(store, embed_fn=fake_embed_fn)
    result = retriever.retrieve("unrelated question")
    assert result.is_weak is True


def test_answer_question_uses_only_the_single_best_source():
    class FakeRetriever:
        def retrieve(self, question, top_k=None):
            return type(
                "Result",
                (),
                {
                    "chunks": [
                        RetrievedChunk(
                            text="AI was founded in 1956.",
                            source="ai_intro.pdf",
                            page=1,
                            chunk_id="a1",
                            distance=0.05,
                        ),
                        RetrievedChunk(
                            text="This is unrelated.",
                            source="other.pdf",
                            page=2,
                            chunk_id="o1",
                            distance=0.2,
                        ),
                    ],
                    "is_weak": False,
                },
            )()

    result = answer_question(FakeRetriever(), "What is AI?")

    assert len(result.chunks) == 1
    assert result.chunks[0].source == "ai_intro.pdf"
    assert result.citations == ["ai_intro.pdf - Page 1"]


def test_no_answer_result_hides_sources():
    class FakeRetriever:
        def retrieve(self, question, top_k=None):
            return type(
                "Result",
                (),
                {
                    "chunks": [
                        RetrievedChunk(
                            text="This chapter explains deep learning systems.",
                            source="ai_intro.pdf",
                            page=1,
                            chunk_id="a1",
                            distance=0.08,
                        )
                    ],
                    "is_weak": False,
                },
            )()

    result = answer_question(FakeRetriever(), "what is my name?")

    assert result.answer == "I couldn't find this information in the provided documents."
    assert result.chunks == []
    assert result.citations == []


def test_extractive_answer_ignores_generic_headings_for_specific_questions():
    chunks = [
        RetrievedChunk(
            text="What Is Machine Learning? Machine learning is a subfield of artificial intelligence.",
            source="machine_learning.pdf",
            page=1,
            chunk_id="c1",
            distance=0.05,
        ),
        RetrievedChunk(
            text="The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
            source="machine_learning.pdf",
            page=2,
            chunk_id="c2",
            distance=0.06,
        ),
    ]

    answer = extractive_answer("What are the main types of machine learning?", chunks)

    assert "supervised" in answer.lower()
    assert "unsupervised" in answer.lower()
    assert "reinforcement" in answer.lower()
    assert "What Is Machine Learning" not in answer


def test_retriever_reranks_question_specific_chunks():
    chunks = [
        RetrievedChunk(
            text="Machine learning provides the theoretical and mathematical foundation for both deep learning and modern natural language processing.",
            source="machine_learning.pdf",
            page=3,
            chunk_id="c3",
            distance=0.34,
        ),
        RetrievedChunk(
            text="The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
            source="machine_learning.pdf",
            page=1,
            chunk_id="c1",
            distance=0.55,
        ),
    ]
    store = FakeVectorStore(chunks)
    retriever = Retriever(store, embed_fn=fake_embed_fn)

    result = retriever.retrieve("What are the main types of machine learning?", top_k=2)

    assert result.chunks[0].chunk_id == "c1"


def test_extractive_answer_prefers_definition_sentence_for_what_is_questions():
    chunks = [
        RetrievedChunk(
            text="Machine learning provides the theoretical and mathematical foundation for both deep learning and modern natural language processing.",
            source="machine_learning.pdf",
            page=3,
            chunk_id="c3",
            distance=0.34,
        ),
        RetrievedChunk(
            text="What Is Machine Learning? Machine learning is a subfield of artificial intelligence in which computer systems learn to perform a task by finding patterns in data.",
            source="machine_learning.pdf",
            page=1,
            chunk_id="c1",
            distance=0.39,
        ),
    ]

    answer = extractive_answer("What is machine learning?", chunks)

    assert "subfield of artificial intelligence" in answer.lower()
    assert "theoretical and mathematical foundation" not in answer.lower()


def test_strong_retrieval_not_flagged_as_weak():
    from src.config import settings

    close_chunk = RetrievedChunk(
        text="relevant",
        source="doc.pdf",
        page=1,
        chunk_id="x",
        distance=max(settings.max_relevant_distance - 0.5, 0.0),
    )
    store = FakeVectorStore([close_chunk])
    retriever = Retriever(store, embed_fn=fake_embed_fn)
    result = retriever.retrieve("relevant question")
    assert result.is_weak is False


def test_sentence_splitter_preserves_full_definition_sentence():
    text = (
        "Machine Learning Fundamentals What Is Machine Learning? "
        "Machine learning is a subfield of artificial intelligence in which computer systems learn to perform a task by finding patterns in data, "
        "rather than by following a fixed set of rules written explicitly by a programmer."
    )

    sentences = split_sentences(text)

    assert any("learn to perform a task by finding patterns in data" in sentence.lower() for sentence in sentences)
    assert all(not sentence.endswith("\n") for sentence in sentences)


def test_retriever_prefers_topic_answers_for_category_questions():
    chunks = [
        RetrievedChunk(
            text="Machine learning is a field of artificial intelligence where systems learn patterns from data.",
            source="machine_learning.pdf",
            page=1,
            chunk_id="generic",
            distance=0.18,
        ),
        RetrievedChunk(
            text="The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
            source="machine_learning.pdf",
            page=2,
            chunk_id="types",
            distance=0.32,
        ),
    ]
    store = FakeVectorStore(chunks)
    retriever = Retriever(store, embed_fn=fake_embed_fn)

    result = retriever.retrieve("What are the main types of machine learning?", top_k=2)

    assert result.chunks[0].chunk_id == "types"


def test_retriever_expands_candidate_pool_for_list_style_questions():
    chunks = [
        RetrievedChunk(
            text=f"generic chunk {i}",
            source="other.pdf",
            page=1,
            chunk_id=f"g{i}",
            distance=0.1 + i * 0.01,
        )
        for i in range(10)
    ]
    chunks.append(
        RetrievedChunk(
            text="Machine learning itself contains many subfields, including supervised learning, unsupervised learning, and reinforcement learning.",
            source="machine_learning.pdf",
            page=2,
            chunk_id="types",
            distance=0.5,
        )
    )
    store = FakeVectorStore(chunks)
    retriever = Retriever(store, embed_fn=fake_embed_fn)

    result = retriever.retrieve("What are the main types of machine learning?", top_k=5)

    assert store.last_top_k > 5
    assert result.chunks[0].chunk_id == "types"


def test_category_question_answer_includes_all_supported_types():
    chunks = [
        RetrievedChunk(
            text="The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
            source="machine_learning.pdf",
            page=2,
            chunk_id="types",
            distance=0.2,
        )
    ]

    answer = extractive_answer("What are the main types of machine learning?", chunks)

    assert "supervised" in answer.lower()
    assert "unsupervised" in answer.lower()
    assert "reinforcement" in answer.lower()


def test_real_machine_learning_chunks_answer_all_three_types():
    chunks = [
        RetrievedChunk(
            text="In practice, this experience usually takes the form of a dataset, and the improvement is measured using a numerical metric such as accuracy or error rate. Supervised Learning Supervised learning is the most widely used category of machine learning. In supervised learning, the training data consists of labeled examples, meaning that each input example is paired with the correct output, often called the label or target.",
            source="machine_learning.pdf",
            page=1,
            chunk_id="machine_learning.pdf::p1::c2",
            distance=0.66,
        ),
        RetrievedChunk(
            text="Common supervised learning algorithms include linear regression and logistic regression, decision trees and random forests, support vector machines, and neural networks. Each algorithm makes different assumptions about the data and has different strengths; for example, decision trees are easy to interpret, while neural networks can capture very complex nonlinear relationships but require more data and computation. Unsupervised Learning Unsupervised learning deals with data that has no labels at all.",
            source="machine_learning.pdf",
            page=1,
            chunk_id="machine_learning.pdf::p1::c4",
            distance=0.76,
        ),
        RetrievedChunk(
            text="Principal component analysis and k-means clustering are two of the most widely taught unsupervised learning algorithms. Reinforcement Learning Reinforcement learning is a third major paradigm, distinct from supervised and unsupervised learning. In reinforcement learning, an agent interacts with an environment over a sequence of time steps, taking actions and receiving feedback in the form of rewards or penalties.",
            source="machine_learning.pdf",
            page=2,
            chunk_id="machine_learning.pdf::p2::c1",
            distance=0.72,
        ),
    ]

    answer = extractive_answer("What are the main types of machine learning?", chunks)

    assert "supervised" in answer.lower()
    assert "unsupervised" in answer.lower()
    assert "reinforcement" in answer.lower()
