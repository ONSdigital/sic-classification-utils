"""Tests for SAYT configuration validation."""

import pytest
from pydantic import ValidationError

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
    default_retriever_specs,
)
from industrial_classification_utils.sayt.sayt_core import CleanCorpus, SaytConfig


@pytest.mark.parametrize(
    "kwargs, exc_type",
    [
        ({"min_chars": 2}, ValidationError),
        ({"min_chars": True}, ValidationError),
        ({"max_suggestions": 0}, ValidationError),
        ({"max_suggestions": 101}, ValidationError),
        ({"ngram_enable": True}, ValidationError),
    ],
)
def test_config_validation(kwargs, exc_type):
    """Reject unsupported SAYT config values and types."""
    with pytest.raises(exc_type):
        SaytConfig.model_validate(kwargs)


def test_default_retriever_specs_returns_standard_set():
    """Provide the standard prefix, n-gram, and semantic specs."""
    specs = default_retriever_specs()

    assert [type(spec).__name__ for spec in specs] == [
        "PrefixRetrieverSpec",
        "NgramRetrieverSpec",
        "SemanticRetrieverSpec",
    ]


@pytest.mark.parametrize(
    "factory, kwargs, match",
    [
        (PrefixRetrieverSpec, {"weight": 0.0}, "retriever weight must be > 0"),
        (NgramRetrieverSpec, {"weight": 0.0}, "retriever weight must be > 0"),
        (NgramRetrieverSpec, {"n": 1}, "ngram n must be between 2 and 5"),
        (NgramRetrieverSpec, {"n": 6}, "ngram n must be between 2 and 5"),
        (NgramRetrieverSpec, {"max_df": 0.0}, "ngram max_df must be in"),
        (NgramRetrieverSpec, {"max_df": 1.1}, "ngram max_df must be in"),
        (SemanticRetrieverSpec, {"weight": 0.0}, "retriever weight must be > 0"),
        (SemanticRetrieverSpec, {"model": "   "}, "semantic model must be"),
    ],
)
def test_retriever_spec_validation(factory, kwargs, match):
    """Reject invalid retriever-spec settings."""
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


def test_ngram_retriever_spec_validates_against_corpus_size():
    """Reject n-gram configs that would filter every feature from a corpus."""
    corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])

    with pytest.raises(ValueError, match="ngram max_df is too low"):
        NgramRetrieverSpec(max_df=0.2).build(corpus, min_chars=3)


def test_retriever_specs_keep_their_config():
    """Expose per-retriever settings on the spec object rather than SaytConfig."""
    n = 4
    max_df = 0.8
    spec = NgramRetrieverSpec(weight=2.0, n=n, max_df=max_df)

    assert spec.weight == pytest.approx(2.0)
    assert spec.n == n
    assert spec.max_df == pytest.approx(max_df)
