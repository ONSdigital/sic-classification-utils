"""Tests for SAYTSuggester, covering constructor validation and suggestion logic."""

# pylint: disable=protected-access,missing-function-docstring,redefined-outer-name

import pandas as pd
import pytest
from pydantic import ValidationError

from industrial_classification_utils.sayt.sayt import SAYTSuggester


@pytest.fixture
def small_corpus():
    # Use mixed case + duplicates to verify normalisation and tie-breaking.
    return [
        ("Car wash", "Car Wash"),
        ("Car wash", "CAR WASH (duplicate)"),
        ("Car waxing", "Car Waxing"),
        ("Waxing car", "Car Waxing"),
        ("Carpentry services", "Carpentry services"),
        ("Dog grooming", "Dog grooming"),
    ]


def test_constructor_rejects_unknown_kwargs(small_corpus):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SAYTSuggester(small_corpus, does_not_exist=True)


@pytest.mark.parametrize(
    "kwargs, exc_type",
    [
        ({"min_chars": 2}, ValidationError),
        ({"min_chars": True}, ValidationError),
        ({"max_suggestions": 0}, ValidationError),
        ({"max_suggestions": 101}, ValidationError),
        ({"ngram_enable": True, "ngram_n": 1}, ValidationError),
        ({"ngram_enable": True, "ngram_n": 6}, ValidationError),
        ({"ngram_enable": True, "ngram_max_df": 0.0}, ValidationError),
        ({"ngram_enable": True, "ngram_max_df": 1.1}, ValidationError),
        ({"ngram_enable": True, "ngram_weight": 0.0}, ValueError),
    ],
)
def test_config_validation(small_corpus, kwargs, exc_type):
    with pytest.raises(exc_type):
        SAYTSuggester(small_corpus, **kwargs)


def test_empty_corpus_after_filtering_raises():
    # Normalisation turns non-strings into empty, and "-9" is explicitly filtered.
    corpus = [None, " ", "-9", ("-9", "ignored")]
    with pytest.raises(ValueError, match="corpus is empty"):
        SAYTSuggester(corpus)


def test_from_csv_builds_and_suggests(tmp_path, small_corpus):
    csv_path = tmp_path / "responses.csv"
    df = pd.DataFrame(
        {
            "search": [x[0] for x in small_corpus],
            "display": [x[1] for x in small_corpus],
        }
    )
    df.to_csv(csv_path, index=False)

    s = SAYTSuggester.from_csv(
        str(csv_path),
        search_text_col="search",
        display_text_col="display",
        semantic_enable=False,
        ngram_enable=False,
        min_chars=3,
        max_suggestions=10,
    )

    assert s.suggest("car")[0].startswith("Car")


def test_suggest_returns_empty_for_short_or_non_string_query(small_corpus):
    s = SAYTSuggester(
        small_corpus, min_chars=4, ngram_enable=False, semantic_enable=False
    )
    assert not s.suggest("car")
    assert not s.suggest(None)


def test_prefix_full_string_match_ranks_expected_terms(small_corpus):
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car")

    # Should contain only the car-related entries (order depends on tie-breakers).
    assert "Car Wash" in results
    assert "Car Waxing" in results
    assert "Dog grooming" not in results


def test_duplicate_terms_increase_rank_via_counts(small_corpus):
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car w")
    # Both "car wash" and "car waxing" start with "car w"; duplicates should win.
    assert results[0] == "Car Waxing"


def test_duplicate_display_variants_are_returned_shorter_first(small_corpus):
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    results = s.suggest("car wa")
    # Both display variants should be present, alphabetically.
    ind1 = results.index("Car Wash")
    ind2 = results.index("CAR WASH (duplicate)")
    assert ind1 < ind2


def test_fuzzy_prefix_can_recover_from_simple_typo(small_corpus):
    s = SAYTSuggester(
        small_corpus, min_chars=3, ngram_enable=False, semantic_enable=False
    )
    # Typo in "carpentry" should still surface "Carpentry services".
    results = s.suggest("carpentey")
    assert "Carpentry services" in results


def test_ngram_recovers_from_typo_when_prefix_does_not_match(small_corpus):
    s = SAYTSuggester(
        small_corpus,
        min_chars=3,
        ngram_enable=True,
        ngram_n=3,
        ngram_max_df=1.0,
        max_suggestions=5,
        semantic_enable=False,
    )
    # Not a prefix of anything, but close in character n-grams.
    results = s.suggest("groming")
    assert results[0] == "Dog grooming"
