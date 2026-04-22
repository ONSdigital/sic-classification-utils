"""Tests for the SAYTSuggester public API."""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,redefined-outer-name,too-few-public-methods,C0116,W0613

from uuid import UUID

import pandas as pd
import pytest
from pydantic import ValidationError

from industrial_classification_utils.sayt.sayt import SAYTSuggester
from industrial_classification_utils.sayt.sayt_common import CleanCorpus
from industrial_classification_utils.sayt.sayt_retrievers import _Suggestion


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


def test_clean_corpus_assigns_uuid_row_ids(small_corpus):
    """Assign deterministic UUID row identifiers to cleaned corpus rows."""
    corpus = CleanCorpus.model_validate(small_corpus)

    assert len({row_id for row_id, _, _ in corpus.rows}) == len(corpus.rows)
    for row_id, _, _ in corpus.rows:
        assert str(UUID(row_id)) == row_id


def test_clean_corpus_accepts_existing_instance_and_dict_input(small_corpus):
    """Preserve existing validated input forms through pydantic coercion."""
    corpus = CleanCorpus.model_validate(small_corpus)

    same_corpus = CleanCorpus.model_validate(corpus)
    dict_corpus = CleanCorpus.model_validate({"corpus": small_corpus})

    assert same_corpus.rows == corpus.rows
    assert dict_corpus.rows == corpus.rows


def test_clean_corpus_rejects_non_iterable_input():
    """Reject scalar corpus values before attempting to clean them."""
    with pytest.raises(TypeError, match="corpus must be an iterable"):
        CleanCorpus._clean_corpus(123)


def test_clean_corpus_warns_and_falls_back_when_display_is_missing():
    """Use the search text when the display value is empty or missing."""
    with pytest.warns(UserWarning, match="using search text as display"):
        corpus = CleanCorpus.model_validate([("Car wash", "")])

    assert corpus.rows[0][2] == "Car wash"


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


def test_from_csv_uses_search_column_as_default_display(tmp_path, small_corpus):
    """Reuse the search column as display when none is configured."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame({"search": [x[0] for x in small_corpus]}).to_csv(csv_path, index=False)

    suggester = SAYTSuggester.from_csv(
        str(csv_path),
        search_text_col="search",
        semantic_enable=False,
        ngram_enable=False,
        min_chars=3,
    )

    assert suggester.suggest("car")[0] == "Car wash"


def test_from_csv_rejects_missing_search_column(tmp_path, small_corpus):
    """Raise when the configured search column is absent from the CSV."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "display": [x[1] for x in small_corpus],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Column 'search' not found"):
        SAYTSuggester.from_csv(str(csv_path), search_text_col="search")


def test_from_csv_rejects_missing_display_column(tmp_path, small_corpus):
    """Raise when the configured display column is absent from the CSV."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "search": [x[0] for x in small_corpus],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Column 'display' not found"):
        SAYTSuggester.from_csv(
            str(csv_path),
            search_text_col="search",
            display_text_col="display",
        )


def test_suggest_returns_empty_for_short_or_non_string_query(small_corpus):
    """Return no suggestions for short or non-string queries."""
    s = SAYTSuggester(
        small_corpus, min_chars=4, ngram_enable=False, semantic_enable=False
    )
    assert not s.suggest("car")
    assert not s.suggest(None)


def test_suggest_with_scores_defaults_to_config_max_suggestions(small_corpus):
    """Use the configured max_suggestions when no override is supplied."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=2,
        ngram_enable=False,
        semantic_enable=False,
    )

    results = suggester.suggest_with_scores("car")

    assert len(results) == 2


def test_suggest_respects_explicit_num_suggestions(small_corpus):
    """Allow callers to override the configured suggestion limit."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=5,
        ngram_enable=False,
        semantic_enable=False,
    )

    assert len(suggester.suggest("car", num_suggestions=1)) == 1


def test_suggest_with_scores_skips_disabled_retrievers(monkeypatch, small_corpus):
    """Use only enabled retrievers when combining scores."""
    semantic_calls = []

    class _StubSemanticRetriever:
        def __init__(self, corpus, *, model, min_chars):
            self._row = corpus.rows[0]

        def suggest(self, q_norm, num_suggestions):
            semantic_calls.append((q_norm, num_suggestions))
            return [
                _Suggestion(
                    display_text=self._row[2],
                    score=3.0,
                    search_text=self._row[1],
                    row_id=self._row[0],
                )
            ]

    monkeypatch.setattr(
        "industrial_classification_utils.sayt.sayt.SemanticRetriever",
        _StubSemanticRetriever,
    )

    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        prefix_enable=False,
        ngram_enable=False,
        semantic_enable=True,
        semantic_model="stub-model",
    )

    results = suggester.suggest_with_scores("car")

    assert semantic_calls == [("car", 100)]
    assert [result.display_text for result in results] == [suggester._corpus.rows[0][2]]


def test_combine_suggestions_ignores_non_positive_score_groups(small_corpus):
    """Drop a retriever group entirely when its max score is not positive."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        ngram_enable=False,
        semantic_enable=False,
    )
    first_row_id, first_search, first_display = suggester._corpus.rows[0]

    combined = suggester._combine_suggestions(
        prefix_results=[
            _Suggestion(
                display_text=first_display,
                score=0.0,
                search_text=first_search,
                row_id=first_row_id,
            )
        ],
        ngram_results=[],
        semantic_results=[],
    )

    assert combined == []


def test_combine_and_dedup_ignore_invalid_scores_and_duplicate_display(small_corpus):
    """Ignore missing row ids and keep the highest-scoring display variant."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        ngram_enable=False,
        semantic_enable=False,
    )
    target_rows = [row for row in suggester._corpus.rows if row[2] == "Car Waxing"]
    first_row_id, first_search, first_display = target_rows[0]
    second_row_id, second_search, _ = target_rows[1]

    combined = suggester._combine_suggestions(
        prefix_results=[
            _Suggestion(
                display_text=first_display,
                score=0.0,
                search_text=first_search,
                row_id=first_row_id,
            ),
            _Suggestion(
                display_text="ignored", score=5.0, search_text="ignored", row_id=""
            ),
        ],
        ngram_results=[
            _Suggestion(
                display_text=first_display,
                score=2.0,
                search_text=first_search,
                row_id=first_row_id,
            ),
            _Suggestion(
                display_text=first_display,
                score=1.0,
                search_text=second_search,
                row_id=second_row_id,
            ),
        ],
        semantic_results=[],
    )

    deduped = suggester._dedup_suggestions(combined)

    assert [suggestion.display_text for suggestion in deduped] == [first_display]
