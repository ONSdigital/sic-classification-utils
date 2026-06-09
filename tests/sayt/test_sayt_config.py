"""Tests for SAYT configuration validation."""

# pylint: disable=too-few-public-methods, R0801

import pytest

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
    SAYTSuggester,
    SemanticRetrieverSpec,
    default_retriever_specs,
)
from industrial_classification_utils.sayt.sayt_core import CleanCorpus


@pytest.mark.parametrize(
    "factory, kwargs, exc_type, match",
    [
        (SAYTSuggester, {"min_chars": 2}, ValueError, "min_chars must be >= 3"),
        (
            SAYTSuggester,
            {"min_chars": True},
            TypeError,
            "min_chars must be an integer",
        ),
        (
            SAYTSuggester,
            {"max_suggestions": 0},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (
            SAYTSuggester,
            {"max_suggestions": 101},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (SAYTBuilder, {"min_chars": 2}, ValueError, "min_chars must be >= 3"),
        (
            SAYTBuilder,
            {"min_chars": True},
            TypeError,
            "min_chars must be an integer",
        ),
        (
            SAYTBuilder,
            {"max_suggestions": 0},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
        (
            SAYTBuilder,
            {"max_suggestions": 101},
            ValueError,
            "max_suggestions must be between 1 and 100",
        ),
    ],
)
def test_runtime_setting_validation(factory, kwargs, exc_type, match):
    """Reject unsupported global SAYT settings on public entry points."""
    with pytest.raises(exc_type, match=match):
        factory([("car wash", "Car Wash")], **kwargs)


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
        (
            PrefixRetrieverSpec,
            {"weight": 0.0},
            "retriever weight must be a finite value > 0",
        ),
        (
            PrefixRetrieverSpec,
            {"weight": float("nan")},
            "retriever weight must be a finite value > 0",
        ),
        (
            NgramRetrieverSpec,
            {"weight": 0.0},
            "retriever weight must be a finite value > 0",
        ),
        (NgramRetrieverSpec, {"n": 1}, "ngram n must be between 2 and 5"),
        (NgramRetrieverSpec, {"n": 6}, "ngram n must be between 2 and 5"),
        (NgramRetrieverSpec, {"max_df": 0.0}, "ngram max_df must be in"),
        (NgramRetrieverSpec, {"max_df": 1.1}, "ngram max_df must be in"),
        (
            SemanticRetrieverSpec,
            {"weight": float("inf")},
            "retriever weight must be a finite value > 0",
        ),
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
    """Expose per-retriever settings on the spec object."""
    n = 4
    max_df = 0.8
    spec = NgramRetrieverSpec(weight=2.0, n=n, max_df=max_df)

    assert spec.weight == pytest.approx(2.0)
    assert spec.n == n
    assert spec.max_df == pytest.approx(max_df)


def test_semantic_retriever_spec_builds_semantic_retriever(monkeypatch):
    """Delegate semantic retriever construction with the configured settings."""
    corpus = CleanCorpus.model_validate([("car wash", "Car Wash")])
    captured = {}

    class _StubSemanticRetriever:
        def __init__(self, corpus_arg, *, model, min_chars):
            captured["corpus"] = corpus_arg
            captured["model"] = model
            captured["min_chars"] = min_chars

    monkeypatch.setattr(
        "industrial_classification_utils.sayt.sayt_retriever_specs.SemanticRetriever",
        _StubSemanticRetriever,
    )

    spec = SemanticRetrieverSpec(model="custom-model", weight=2.0)

    retriever = spec.build(corpus, min_chars=4)

    assert isinstance(retriever, _StubSemanticRetriever)
    assert captured == {
        "corpus": corpus,
        "model": "custom-model",
        "min_chars": 4,
    }
