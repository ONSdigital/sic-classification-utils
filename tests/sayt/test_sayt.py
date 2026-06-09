"""Tests for the SAYTSuggester public API."""

# pylint: disable=protected-access,redefined-outer-name,too-few-public-methods,C0116,W0613

from dataclasses import dataclass
from uuid import UUID

import pandas as pd
import pytest

from industrial_classification_utils.sayt import (
    PrefixRetrieverSpec,
    SAYTBuilder,
)
from industrial_classification_utils.sayt.sayt import SAYTSuggester
from industrial_classification_utils.sayt.sayt_core import (
    CleanCorpus,
    PersistedCorpusRow,
    Suggestion,
)


def test_constructor_rejects_unknown_kwargs(small_corpus):
    """Reject unknown constructor kwargs during config validation."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SAYTSuggester(small_corpus, does_not_exist=True)  # pylint: disable=E1123


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


def test_clean_corpus_restores_persisted_rows(small_corpus):
    """Restore cleaned corpus rows without regenerating row identifiers."""
    corpus = CleanCorpus.model_validate(small_corpus)

    restored = CleanCorpus.from_persisted_rows(
        [PersistedCorpusRow(*row) for row in corpus.rows]
    )

    assert restored.rows == corpus.rows
    assert restored.id_to_search == corpus.id_to_search
    assert restored.id_to_display == corpus.id_to_display
    assert restored.display_text_count == corpus.display_text_count


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
        retrievers=[PrefixRetrieverSpec()],
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
        retrievers=[PrefixRetrieverSpec()],
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


def test_from_artifact_restores_prefix_suggester(tmp_path, small_corpus):
    """Round-trip a prefix-only artifact into a working suggester."""
    artifact_dir = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
    ).build_artifact(tmp_path / "artifact")

    restored = SAYTSuggester.from_artifact(artifact_dir)
    expected = SAYTSuggester(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
    )

    assert restored.suggest("car") == expected.suggest("car")
    assert restored.get_config() == expected.get_config()


def test_suggest_returns_empty_for_short_or_non_string_query(small_corpus):
    """Return no suggestions for short or non-string queries."""
    s = SAYTSuggester(small_corpus, min_chars=4, retrievers=[PrefixRetrieverSpec()])
    assert not s.suggest("car")
    assert not s.suggest(None)


def test_suggest_with_scores_defaults_to_config_max_suggestions(small_corpus):
    """Use the configured max_suggestions, but keep ties at the cutoff."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=2,
        retrievers=[PrefixRetrieverSpec()],
    )

    results = suggester.suggest_with_scores("car")

    assert {result.display_text for result in results} == {
        "Car Waxing",
        "Car Wash",
        "CAR WASH (duplicate)",
        "Carpentry services",
    }


def test_suggest_respects_explicit_num_suggestions(small_corpus):
    """Allow callers to override the configured limit, while keeping ties."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=5,
        retrievers=[PrefixRetrieverSpec()],
    )

    assert suggester.suggest("car", num_suggestions=1) == [
        "Car Waxing",
        "Car Wash",
        "CAR WASH (duplicate)",
        "Carpentry services",
    ]


def test_suggest_with_scores_keeps_ties_at_cutoff(small_corpus):
    """Keep all tied scored suggestions at the public cutoff."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
    )

    results = suggester.suggest_with_scores("car", num_suggestions=1)

    assert {result.display_text for result in results} == {
        "Car Waxing",
        "Car Wash",
        "CAR WASH (duplicate)",
        "Carpentry services",
    }


def test_suggest_keeps_ties_at_cutoff(small_corpus):
    """Keep all tied display suggestions at the public cutoff."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
    )

    results = suggester.suggest("car", num_suggestions=1)

    assert results == [
        "Car Waxing",
        "Car Wash",
        "CAR WASH (duplicate)",
        "Carpentry services",
    ]


def test_suggest_with_scores_uses_only_supplied_retrievers(small_corpus):
    """Delegate only to the configured retriever specs."""
    semantic_calls = []

    class _StubRetriever:
        def __init__(self, row):
            self._row = row

        def suggest_with_scores(self, q_norm, num_suggestions):
            semantic_calls.append((q_norm, num_suggestions))
            return [
                Suggestion(
                    display_text=self._row[2],
                    score=3.0,
                    search_text=self._row[1],
                    row_id=self._row[0],
                )
            ]

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        weight: float = 1.0
        name: str = "stub"

        def build(self, corpus, *, min_chars):
            return _StubRetriever(corpus.rows[0])

    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_StubRetrieverSpec()],
    )

    results = suggester.suggest_with_scores("car")

    assert semantic_calls == [("car", 100)]
    assert [result.display_text for result in results] == [suggester._corpus.rows[0][2]]


def test_combine_suggestions_ignores_non_positive_score_groups(small_corpus):
    """Drop a retriever group entirely when its max score is not positive."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
    )
    first_row_id, first_search, first_display = suggester._corpus.rows[0]

    combined = suggester._combine_suggestions(
        [
            (
                1.0,
                [
                    Suggestion(
                        display_text=first_display,
                        score=0.0,
                        search_text=first_search,
                        row_id=first_row_id,
                    )
                ],
            ),
            (1.0, []),
            (1.0, []),
        ]
    )

    assert combined == []


def test_combine_suggestions_ignores_invalid_scores(small_corpus):
    """Ignore missing row ids and keep distinct row ids in combined scores."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[PrefixRetrieverSpec()],
    )
    target_rows = [row for row in suggester._corpus.rows if row[2] == "Car Waxing"]
    first_row_id, first_search, first_display = target_rows[0]
    second_row_id, second_search, _ = target_rows[1]

    combined = suggester._combine_suggestions(
        [
            (
                1.0,
                [
                    Suggestion(
                        display_text=first_display,
                        score=0.0,
                        search_text=first_search,
                        row_id=first_row_id,
                    ),
                    Suggestion(
                        display_text="ignored",
                        score=5.0,
                        search_text="ignored",
                        row_id="",
                    ),
                ],
            ),
            (
                1.0,
                [
                    Suggestion(
                        display_text=first_display,
                        score=2.0,
                        search_text=first_search,
                        row_id=first_row_id,
                    ),
                    Suggestion(
                        display_text=first_display,
                        score=1.0,
                        search_text=second_search,
                        row_id=second_row_id,
                    ),
                ],
            ),
        ]
    )

    assert combined == [(first_row_id, 1.0), (second_row_id, 0.5)]


def test_suggester_defaults_to_standard_retriever_specs(monkeypatch, small_corpus):
    """Use the standard prefix, n-gram, and semantic specs when none are supplied."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            return []

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str
        weight: float = 1.0

        def build(self, corpus, *, min_chars):
            return _StubRetriever()

    monkeypatch.setattr(
        "industrial_classification_utils.sayt.sayt.default_retriever_specs",
        lambda: [
            _StubRetrieverSpec(name="prefix"),
            _StubRetrieverSpec(name="ngram"),
            _StubRetrieverSpec(name="semantic"),
        ],
    )

    suggester = SAYTSuggester(small_corpus, min_chars=3)

    assert [configured.name for configured in suggester._retrievers] == [
        "prefix",
        "ngram",
        "semantic",
    ]


def test_constructor_rejects_empty_retriever_list(small_corpus):
    """Reject suggester construction without any retriever specs."""
    with pytest.raises(ValueError, match="At least one retriever"):
        SAYTSuggester(small_corpus, retrievers=[])


def test_constructor_rejects_invalid_custom_retriever_weight(small_corpus):
    """Reject custom retriever specs whose own weight is invalid."""
    build_calls = []

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            return []

    @dataclass(frozen=True, slots=True)
    class _StubRetrieverSpec:
        name: str = "stub"
        weight: float = 1.0

        def build(self, corpus, *, min_chars):
            build_calls.append((corpus, min_chars))
            return _StubRetriever()

    @dataclass(frozen=True, slots=True)
    class _NegativeStubRetrieverSpec:
        name: str = "negative"
        weight: float = -0.5

        def build(self, corpus, *, min_chars):
            build_calls.append((corpus, min_chars))
            return _StubRetriever()

    @dataclass(frozen=True, slots=True)
    class _NanStubRetrieverSpec:
        name: str = "nan"
        weight: float = float("nan")

        def build(self, corpus, *, min_chars):
            build_calls.append((corpus, min_chars))
            return _StubRetriever()

    with pytest.raises(
        ValueError,
        match="Retriever 'negative' weight must be a finite value > 0",
    ):
        SAYTSuggester(
            small_corpus,
            retrievers=[_StubRetrieverSpec(), _NegativeStubRetrieverSpec()],
        )

    with pytest.raises(
        ValueError,
        match="Retriever 'nan' weight must be a finite value > 0",
    ):
        SAYTSuggester(
            small_corpus,
            retrievers=[_StubRetrieverSpec(), _NanStubRetrieverSpec()],
        )

    assert not build_calls
