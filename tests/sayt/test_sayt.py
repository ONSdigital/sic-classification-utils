"""Tests for the SAYTSuggester public API."""

# pylint: disable=protected-access,redefined-outer-name,too-few-public-methods,C0116,W0613

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
    SaytConfiguration,
)
from industrial_classification_utils.sayt.core import (
    CleanCorpus,
    PersistedCorpusRow,
    Suggestion,
)
from industrial_classification_utils.sayt.suggester import SAYTSuggester


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


def test_clean_corpus_model_dump_excludes_derived_lookup_dicts(small_corpus):
    """Keep derived lookup dictionaries out of the public model fields."""
    corpus = CleanCorpus.model_validate(small_corpus)

    dumped = corpus.model_dump()

    assert "id_to_search" not in dumped
    assert "id_to_display" not in dumped
    assert "display_text_count" not in dumped
    assert dumped["rows"] == corpus.rows


def test_clean_corpus_restores_persisted_rows(small_corpus):
    """Restore cleaned corpus rows without regenerating row identifiers."""
    corpus = CleanCorpus.model_validate(small_corpus)

    restored = CleanCorpus.from_persisted_rows(
        [PersistedCorpusRow(*row) for row in corpus.rows]
    )

    assert restored.rows == corpus.rows
    assert restored.model_dump() == corpus.model_dump()


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
    restored_config = restored.get_config()
    expected_config = expected.get_config()

    assert restored.suggest("car") == expected.suggest("car")
    assert restored_config.settings == expected_config.settings
    assert restored_config.corpus == expected_config.corpus
    assert [
        retriever.model_dump(exclude={"artifact_provenance"})
        for retriever in restored_config.retrievers
    ] == [
        retriever.model_dump(exclude={"artifact_provenance"})
        for retriever in expected_config.retrievers
    ]
    assert restored_config.artifact_provenance is not None
    assert restored_config.artifact_provenance.artifact_dir == str(artifact_dir)
    assert restored_config.retrievers[0].artifact_provenance is not None
    assert expected_config.artifact_provenance is None


def test_from_artifact_rejects_manifest_corpus_size_mismatch(tmp_path, small_corpus):
    """Reject artifacts whose manifest corpus size disagrees with stored rows."""
    artifact_dir = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
    ).build_artifact(tmp_path / "artifact")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="Artifact corpus size does not match manifest"
    ):
        SAYTSuggester.from_artifact(artifact_dir)


def test_get_config_returns_rich_runtime_summary(small_corpus):
    """Expose runtime settings, corpus stats, and retriever summaries."""
    suggester = SAYTSuggester(
        small_corpus,
        min_chars=3,
        max_suggestions=5,
        retrievers=[
            PrefixRetrieverSpec(weight=2.0),
            NgramRetrieverSpec(weight=1.0, n=4, max_df=1.0),
        ],
    )

    config = suggester.get_config()
    display_counts = Counter(display for _, _, display in suggester._corpus.rows)

    assert isinstance(config, SaytConfiguration)
    assert config.settings.model_dump() == {
        "min_chars": 3,
        "max_suggestions": 5,
    }
    assert config.corpus.model_dump() == {
        "size": suggester._corpus.size,
        "unique_display_texts": len(display_counts),
        "max_duplication": max(display_counts.values(), default=0),
    }
    assert [retriever.name for retriever in config.retrievers] == ["prefix", "ngram"]
    assert config.retrievers[0].config == {}
    assert config.retrievers[1].config == {"n": 4, "max_df": 1.0}
    assert config.retrievers[0].configured_weight == pytest.approx(2.0)
    assert config.retrievers[0].normalised_weight == pytest.approx(2.0 / 3.0)
    assert config.retrievers[1].normalised_weight == pytest.approx(1.0 / 3.0)
    assert config.retrievers[1].retriever_type == "NgramRetriever"
    assert config.artifact_provenance is None


def test_get_config_supports_custom_specs_without_artifact_handlers(small_corpus):
    """Summarise custom runtime-only specs without requiring persistence hooks."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return []

    class _CustomSpec:
        def __init__(self, *, trigger: str, weight: float = 1.0):
            self.trigger = trigger
            self.weight = weight
            self.name = "custom"

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever()

    config = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_CustomSpec(trigger="groom")],
    ).get_config()

    assert config.retrievers[0].config == {"trigger": "groom"}
    assert config.retrievers[0].artifact_provenance is None


def test_get_config_serialises_nested_custom_spec_values(small_corpus):
    """Convert non-JSON-native custom spec config values into JSON-safe forms."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return []

    class _Marker:
        def __str__(self):
            return "marker-object"

    class _CustomSpec:
        def __init__(self):
            self.name = "custom"
            self.weight = 1.0
            self.folder = Path("artifacts/model")
            self.options = {
                "labels": ["car", Path("cache/index"), _Marker()],
                "metadata": {"marker": _Marker()},
            }
            self.values = (Path("weights.bin"), _Marker())

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever()

    config = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_CustomSpec()],
    ).get_config()

    assert config.retrievers[0].config == {
        "folder": "artifacts/model",
        "options": {
            "labels": ["car", "cache/index", "marker-object"],
            "metadata": {"marker": "marker-object"},
        },
        "values": ["weights.bin", "marker-object"],
    }


def test_get_config_returns_empty_config_for_slots_only_custom_spec(small_corpus):
    """Return an empty config summary when a custom spec exposes no __dict__."""

    class _StubRetriever:
        def suggest_with_scores(self, q_norm, num_suggestions):
            _ = (q_norm, num_suggestions)
            return []

    class _SlotsOnlySpec:
        __slots__ = ("name", "trigger", "weight")

        def __init__(self, *, trigger: str, weight: float = 1.0):
            self.name = "slots-only"
            self.weight = weight
            self.trigger = trigger

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)
            return _StubRetriever()

    config = SAYTSuggester(
        small_corpus,
        min_chars=3,
        retrievers=[_SlotsOnlySpec(trigger="groom")],
    ).get_config()

    assert config.retrievers[0].config == {}


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
        "industrial_classification_utils.sayt.suggester.default_retriever_specs",
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


def test_clean_corpus_rejects_empty_persisted_rows():
    """Reject empty persisted row collections during artifact restore."""
    with pytest.raises(ValueError, match="corpus is empty after filtering"):
        CleanCorpus.from_persisted_rows([])


def test_clean_corpus_coerces_persisted_tuple_values_to_strings():
    """Coerce tuple-based persisted rows to strings before rebuilding indexes."""
    restored = CleanCorpus.from_persisted_rows([(123, 456, 789)])

    assert restored.rows == [("123", "456", "789")]
