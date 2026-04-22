"""Tests for SAYT retrieval and ranking behavior."""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,redefined-outer-name,too-few-public-methods,C0116,W0613

import numpy as np
import pytest
from classifai.vectorisers import VectoriserBase

from industrial_classification_utils.sayt.sayt import SAYTSuggester
from industrial_classification_utils.sayt.sayt_common import CleanCorpus
from industrial_classification_utils.sayt.sayt_retrievers import (
    NgramRetriever,
    PrefixRetriever,
    SemanticRetriever,
    _CharNgramVectoriser,
    _DenseVectorIndex,
    _L2NormalisingVectoriser,
    _PrefixIndex,
)


def test_prefix_full_string_match_ranks_expected_terms(small_corpus):
    """Return relevant prefix matches and exclude unrelated terms."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car")

    assert "Car Wash" in results
    assert "Car Waxing" in results
    assert "Dog grooming" not in results


def test_duplicate_terms_increase_rank_via_counts(small_corpus):
    """Prefer duplicated underlying search terms when scores tie."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car w")
    assert results[0] == "Car Waxing"


def test_duplicate_display_variants_are_returned_shorter_first(small_corpus):
    """Rank duplicate display variants using the configured tie-breakers."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car wa")
    ind1 = results.index("Car Wash")
    ind2 = results.index("CAR WASH (duplicate)")
    assert ind1 < ind2


def test_fuzzy_prefix_can_recover_from_simple_typo(small_corpus):
    """Recover expected results when the query has a small typo."""
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("carpentey")
    assert "Carpentry services" in results


def test_ngram_recovers_from_typo_when_prefix_does_not_match(small_corpus):
    """Use n-gram retrieval when typoed input misses prefix matching."""
    s = SAYTSuggester(
        small_corpus,
        min_chars=3,
        ngram_enable=True,
        ngram_n=3,
        ngram_max_df=1.0,
        max_suggestions=5,
        semantic_enable=False,
    )
    results = s.suggest("groming")
    assert results[0] == "Dog grooming"


class _StubVectoriser(VectoriserBase):
    def __init__(self, output):
        self._output = output

    def transform(self, texts):
        return self._output


class _StubSearchResults:
    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient="dict"):
        assert orient == "records"
        return self._rows


class _StubVectorStore:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def search(self, query, n_results=10):
        self.calls.append((query, n_results))
        return _StubSearchResults(self._rows)


def test_prefix_retriever_returns_empty_for_short_queries(small_corpus):
    """Skip prefix work when the query is shorter than the minimum."""
    corpus = CleanCorpus.model_validate(small_corpus)

    assert PrefixRetriever(corpus, min_chars=4).suggest("car", num_suggestions=5) == []


def test_prefix_retriever_handles_empty_prefix_candidates(small_corpus):
    """Skip fuzzy scoring when the query prefix is empty."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = PrefixRetriever.__new__(PrefixRetriever)
    retriever._corpus = corpus
    retriever._min_chars = 0
    retriever._index = _PrefixIndex(
        sorted_terms=[("", corpus.rows[0][0])],
        prefix_terms=[""],
        token_index={},
    )

    results = retriever.suggest("", num_suggestions=5)

    assert [result.display_text for result in results] == [corpus.rows[0][2]]


def test_l2_normalising_vectoriser_handles_one_dimensional_output():
    """Normalise a single returned embedding into a 2D unit vector."""
    vectoriser = _L2NormalisingVectoriser(_StubVectoriser(np.array([3.0, 4.0])))

    vectors = vectoriser.transform(["query"])

    assert vectors.shape == (1, 2)
    assert vectors[0] == pytest.approx(np.array([0.6, 0.8]))


def test_l2_normalising_vectoriser_preserves_two_dimensional_output():
    """Normalise batched vectors without reshaping an existing matrix."""
    vectoriser = _L2NormalisingVectoriser(
        _StubVectoriser(np.array([[3.0, 4.0], [5.0, 12.0]]))
    )

    vectors = vectoriser.transform(["first", "second"])

    assert vectors.shape == (2, 2)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.array([1.0, 1.0]))


def test_char_ngram_vectoriser_accepts_single_string_input():
    """Transform a scalar string query into a single-row dense matrix."""
    vectoriser = _CharNgramVectoriser(["car wash", "dog grooming"], n=3, max_df=1.0)

    vectors = vectoriser.transform("car")

    assert vectors.shape[0] == 1


def test_ngram_retriever_returns_empty_for_empty_query_vector(
    monkeypatch, small_corpus
):
    """Return no suggestions when the vector store yields no matches."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._corpus = corpus
    retriever._min_chars = 3
    retriever._index = _DenseVectorIndex(
        vector_store=_StubVectorStore([]),
        num_vectors=1,
    )

    assert not retriever.suggest("car", num_suggestions=3)


def test_ngram_retriever_returns_empty_for_short_queries():
    """Stop before vectorisation when the query is shorter than min_chars."""
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._min_chars = 4

    assert not retriever.suggest("car", num_suggestions=3)


def test_ngram_retriever_returns_empty_for_empty_similarity_matrix():
    """Stop early when the dense index contains no stored vectors."""
    retriever = NgramRetriever.__new__(NgramRetriever)
    retriever._corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    retriever._min_chars = 3
    retriever._index = _DenseVectorIndex(
        vector_store=_StubVectorStore([]),
        num_vectors=0,
    )

    assert not retriever.suggest("car", num_suggestions=3)


def test_semantic_retriever_builds_index_with_wrapped_vectoriser(
    monkeypatch, small_corpus
):
    """Wrap the base embedding vectoriser before building the dense index."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)

    class _StubHFVectoriser:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def transform(self, texts):
            return np.array([[1.0, 0.0]])

    def _fake_build_dense_vector_index(*, corpus, vectoriser):
        captured["vectoriser_type"] = type(vectoriser).__name__
        return _DenseVectorIndex(
            vector_store=_StubVectorStore([]),
            num_vectors=1,
        )

    monkeypatch.setattr(
        "industrial_classification_utils.sayt.sayt_retrievers.HuggingFaceVectoriser",
        _StubHFVectoriser,
    )
    monkeypatch.setattr(
        "industrial_classification_utils.sayt.sayt_retrievers._DenseVectorIndex.from_corpus",
        _fake_build_dense_vector_index,
    )

    retriever = SemanticRetriever(corpus, model="all-MiniLM-L6-v2", min_chars=3)

    assert captured == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "vectoriser_type": "_L2NormalisingVectoriser",
    }
    assert retriever._min_chars == 3


def test_semantic_retriever_returns_empty_for_short_queries():
    """Stop before vectorisation when the semantic query is too short."""
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever._min_chars = 4

    assert not retriever.suggest("car", num_suggestions=3)


def test_semantic_retriever_returns_empty_for_empty_query_vector(small_corpus):
    """Return no suggestions when semantic vector search yields no matches."""
    corpus = CleanCorpus.model_validate(small_corpus)
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever._corpus = corpus
    retriever._min_chars = 3
    retriever._index = _DenseVectorIndex(
        vector_store=_StubVectorStore([]),
        num_vectors=1,
    )

    assert not retriever.suggest("car", num_suggestions=3)


def test_semantic_retriever_returns_empty_for_empty_similarity_matrix():
    """Stop early when semantic retrieval has no stored vectors."""
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever._corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    retriever._min_chars = 3
    retriever._index = _DenseVectorIndex(
        vector_store=_StubVectorStore([]),
        num_vectors=0,
    )

    assert not retriever.suggest("car", num_suggestions=3)
