"""Tests for the SAYTSuggester public API."""

# pylint: disable=protected-access,redefined-outer-name

import pandas as pd
import pytest
from pydantic import ValidationError

from industrial_classification_utils.sayt.sayt import SAYTSuggester


def test_constructor_rejects_unknown_kwargs(small_corpus):
    """Reject unknown constructor kwargs during config validation."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SAYTSuggester(small_corpus, does_not_exist=True)


def test_empty_corpus_after_filtering_raises():
    """Raise when corpus normalisation removes every input row."""
    # Normalisation turns non-strings into empty, and "-9" is explicitly filtered.
    corpus = [None, " ", "-9", ("-9", "ignored")]
    with pytest.raises(ValueError, match="corpus is empty"):
        SAYTSuggester(corpus)


def test_from_csv_builds_and_suggests(tmp_path, small_corpus):
    """Build a suggester from CSV input and return matching suggestions."""
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
    """Return no suggestions for short or non-string queries."""
    s = SAYTSuggester(
        small_corpus, min_chars=4, ngram_enable=False, semantic_enable=False
    )
    assert not s.suggest("car")
    assert not s.suggest(None)
