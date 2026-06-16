"""Search-as-you-type (SAYT) orchestration.

This module provides the public suggester API that coordinates configured
retrievers and combines their scores into ranked suggestions.
"""

# pylint: disable=duplicate-code

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from survey_assist_utils.logging import get_logger

from .core import (
    CleanCorpus,
    SaytArtifactProvenance,
    SaytConfiguration,
    SaytCorpusSummary,
    SaytGlobalSettings,
    SaytRetrieverArtifactProvenance,
    SaytRetrieverSummary,
    Suggestion,
    _normalise,
    take_with_ties,
    validate_max_suggestions,
    validate_min_chars,
)
from .retriever_specs import (
    Retriever,
    RetrieverSpec,
    default_retriever_specs,
)
from .storage import (
    SAYT_ARTIFACT_TYPE,
    SAYT_ARTIFACT_VERSION,
    StoredRetrieverSpec,
    load_corpus_from_csv,
    load_retriever_from_artifact,
    read_artifact_corpus,
    read_artifact_manifest,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _ConfiguredRetriever:
    """Runtime retriever binding with its configured contribution weight."""

    name: str
    weight: float
    retriever: Retriever


class SAYTSuggester:  # pylint: disable=too-many-instance-attributes
    """Suggest free-text responses as a user types.

    The suggester:
    - validates and cleans the supplied corpus
    - builds the configured retrievers for that corpus
    - combines retriever-local scores into a shared weighted ranking

    By default it uses the standard prefix, n-gram, and semantic retriever
    specifications. Use ``retrievers=`` to override that mix.

    Suggester-wide settings are configured directly on the suggester. At
    present these include:
    - ``min_chars``: minimum query length before retrieval runs
    - ``max_suggestions``: default maximum number of ranked suggestions to return

    Examples:
        Basic usage with an in-memory corpus:

            ```python
            from industrial_classification_utils.sayt import SAYTSuggester

            suggester = SAYTSuggester(
                corpus=[
                    ("Car wash", "Car Wash"),
                    ("Dog grooming", "Dog grooming"),
                ],
                min_chars=3,
                max_suggestions=5,
            )

            results = suggester.suggest("car")
            ```

        Usage with custom retriever specifications:

            ```python
            from industrial_classification_utils.sayt import (
                PrefixRetrieverSpec,
                SAYTSuggester,
            )

            suggester = SAYTSuggester(
                corpus=[("Car wash", "Car Wash")],
                retrievers=[PrefixRetrieverSpec()],
                min_chars=3,
            )
            ```
    """

    def __init__(
        self,
        corpus: Iterable[tuple[object, object]] | Iterable[str],
        *,
        retrievers: Sequence[RetrieverSpec] | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> None:
        """Initialise a suggester for a cleaned response corpus.

        Args:
            corpus: Iterable of search strings or ``(search_text, display_text)``
                pairs.
            retrievers: Optional retriever specifications. When omitted, the
                standard prefix, n-gram, and semantic spec set is used.
            min_chars: Minimum query length before retrieval runs.
            max_suggestions: Default maximum number of ranked suggestions to
                return.
        """
        self._corpus = CleanCorpus.model_validate(corpus)
        self._min_chars = validate_min_chars(min_chars)
        self._max_suggestions = validate_max_suggestions(max_suggestions)

        self._retriever_specs = tuple(
            default_retriever_specs() if retrievers is None else retrievers
        )
        self._retrievers = self._build_retrievers(self._retriever_specs)
        self._max_duplication = max(self._corpus.display_text_count.values(), default=0)
        self._stored_retrievers: tuple[StoredRetrieverSpec, ...] | None = None
        self._artifact_provenance: SaytArtifactProvenance | None = None
        logger.info(
            "SAYT suggester initialized",
            config=self.get_config().model_dump(mode="json"),
        )

    @classmethod
    def _from_state(  # pylint: disable=too-many-arguments  # noqa: PLR0913
        cls,
        *,
        corpus: CleanCorpus,
        min_chars: int,
        max_suggestions: int,
        retriever_specs: Sequence[RetrieverSpec],
        retrievers: list[_ConfiguredRetriever],
        stored_retrievers: Sequence[StoredRetrieverSpec] | None = None,
        artifact_provenance: SaytArtifactProvenance | None = None,
    ) -> "SAYTSuggester":
        """Construct a suggester from already-validated runtime state."""
        suggester = cls.__new__(cls)
        suggester._corpus = corpus
        suggester._min_chars = min_chars
        suggester._max_suggestions = max_suggestions
        suggester._max_duplication = max(corpus.display_text_count.values(), default=0)
        suggester._retriever_specs = tuple(retriever_specs)
        suggester._retrievers = retrievers
        suggester._stored_retrievers = (
            tuple(stored_retrievers) if stored_retrievers is not None else None
        )
        suggester._artifact_provenance = artifact_provenance
        logger.info(
            "SAYT suggester initialized",
            config=suggester.get_config().model_dump(mode="json"),
        )
        return suggester

    @classmethod
    def from_csv(  # pylint: disable=too-many-arguments  # noqa: PLR0913
        cls,
        file_path: str | os.PathLike,
        *,
        search_text_col: str = "title",
        display_text_col: str | None = None,
        retrievers: Sequence[RetrieverSpec] | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> "SAYTSuggester":
        """Build a suggester from CSV input.

        Args:
            file_path: Path to the CSV file containing suggestion rows.
            search_text_col: Column containing the searchable text.
            display_text_col: Optional column containing display text. When
                omitted, the search column is reused for display values.
            retrievers: Optional retriever specifications. When omitted, the
                standard retriever set is used.
            min_chars: Minimum query length before retrieval runs.
            max_suggestions: Default maximum number of ranked suggestions to
                return.

        Returns:
            A configured ``SAYTSuggester`` instance.

        Raises:
            ValueError: If the requested search or display column is missing.
        """
        corpus_rows = load_corpus_from_csv(
            file_path,
            search_text_col=search_text_col,
            display_text_col=display_text_col,
        )
        return cls(
            corpus_rows,
            retrievers=retrievers,
            min_chars=min_chars,
            max_suggestions=max_suggestions,
        )

    @classmethod
    def from_artifact(cls, artifact_dir: str | os.PathLike) -> "SAYTSuggester":
        """Load a suggester from a persisted SAYT artifact directory."""
        artifact_path = Path(artifact_dir)
        manifest = read_artifact_manifest(artifact_dir=artifact_path)
        persisted_rows = read_artifact_corpus(
            artifact_dir=artifact_path,
            corpus_file=manifest.corpus_file,
        )
        corpus = CleanCorpus.from_persisted_rows(persisted_rows)
        if corpus.size != manifest.corpus_size:
            raise ValueError("Artifact corpus size does not match manifest")

        retrievers = _load_retrievers_from_artifact(
            corpus=corpus,
            min_chars=manifest.min_chars,
            stored_retrievers=manifest.retrievers,
            artifact_dir=artifact_path,
        )
        artifact_provenance = SaytArtifactProvenance(
            artifact_dir=str(artifact_path),
            artifact_type=SAYT_ARTIFACT_TYPE,
            artifact_version=SAYT_ARTIFACT_VERSION,
            corpus_file=manifest.corpus_file,
            corpus_size=manifest.corpus_size,
        )
        return cls._from_state(
            corpus=corpus,
            min_chars=manifest.min_chars,
            max_suggestions=manifest.max_suggestions,
            retriever_specs=[
                stored_retriever.spec for stored_retriever in manifest.retrievers
            ],
            retrievers=retrievers,
            stored_retrievers=manifest.retrievers,
            artifact_provenance=artifact_provenance,
        )

    def _build_retrievers(
        self, retriever_specs: Sequence[RetrieverSpec]
    ) -> list[_ConfiguredRetriever]:
        return [
            _ConfiguredRetriever(
                name=spec.name,
                weight=weight,
                retriever=spec.build(
                    self._corpus,
                    min_chars=self._min_chars,
                ),
            )
            for spec, weight in _normalised_retriever_specs(retriever_specs)
        ]

    def _dedup_suggestions(
        self, suggestions: list[Suggestion]
    ) -> list[tuple[str, float]]:
        # sort by score and deduplicate by display text, keeping the highest-scoring variant.
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: (
                -s.score,
                -self._corpus.display_text_count.get(s.display_text, 0),
                s.display_text.lower(),
                s.row_id,
            ),
        )
        seen: set[str] = set()
        deduped: list[tuple[str, float]] = []
        for s in sorted_suggestions:
            display_text = s.display_text  # lower? normalised?
            if display_text not in seen:
                deduped.append((display_text, s.score))
                seen.add(display_text)
        return deduped

    def _combine_suggestions(
        self,
        result_groups: Iterable[tuple[float, list[Suggestion]]],
    ) -> list[tuple[str, float]]:
        def normalise_scores(
            items: list[Suggestion], weight: float
        ) -> dict[str, float]:
            if not items:
                return {}
            max_score = max((float(s.score) for s in items), default=0.0)
            if max_score <= 0:
                return {}
            out: dict[str, float] = {}
            for s in items:
                if not s.row_id:
                    continue
                out[s.row_id] = max(
                    out.get(s.row_id, 0.0), float(s.score) / max_score * weight
                )
            return out

        combined_scores: dict[str, float] = {}
        for weight, suggestions in result_groups:
            d = normalise_scores(suggestions, weight)
            for k, v in d.items():
                combined_scores[k] = combined_scores.get(k, 0.0) + v

        return [(row_id, float(score)) for row_id, score in combined_scores.items()]

    def _collect_retriever_results(
        self, q_norm: str, num_suggestions: int
    ) -> list[tuple[float, list[Suggestion]]]:
        return [
            (
                configured_retriever.weight,
                configured_retriever.retriever.suggest_with_scores(
                    q_norm,
                    num_suggestions=num_suggestions,
                ),
            )
            for configured_retriever in self._retrievers
        ]

    def suggest_with_scores(
        self, query: str | None, num_suggestions: int | None = None
    ) -> list[Suggestion]:
        """Return ranked suggestions and their combined scores.

        Args:
            query: Raw user query text.
            num_suggestions: Optional maximum number of ranked suggestions to
                return. When omitted, the configured default is used.

        Returns:
            A list of combined suggestions ordered by descending score. Returns
            an empty list when the normalised query is shorter than
            ``min_chars``.
        """
        if num_suggestions is None:
            num_suggestions = self._max_suggestions
        q_norm = _normalise(query)
        if len(q_norm) < self._min_chars:
            return []

        # Ask for more suggestions, as some may be filtered out after deduplication
        results_by_kind = self._collect_retriever_results(
            q_norm,
            num_suggestions=10 * num_suggestions,
        )

        combined_result = self._combine_suggestions(results_by_kind)
        ranked_results = take_with_ties(combined_result, num_suggestions)
        out = [
            Suggestion(
                row_id=row_id,
                display_text=self._corpus.id_to_display.get(row_id, ""),
                score=score,
                search_text=self._corpus.id_to_search.get(row_id, ""),
            )
            for row_id, score in ranked_results
        ]

        return out

    def suggest(
        self, query: str | None, num_suggestions: int | None = None
    ) -> list[str]:
        """Return deduplicated display-text suggestions.

        Args:
            query: Raw user query text.
            num_suggestions: Optional maximum number of display values to
                return. When omitted, the configured default is used.

        Returns:
            A list of display-text suggestions ordered by descending combined
            score, while preserving ties at the cutoff.
        """
        if num_suggestions is None:
            num_suggestions = self._max_suggestions
        results = self.suggest_with_scores(
            query, num_suggestions=num_suggestions * self._max_duplication
        )
        dedup_results = self._dedup_suggestions(results)
        ranked_results = take_with_ties(dedup_results, num_suggestions)
        return [result[0] for result in ranked_results]

    def get_config(self) -> SaytConfiguration:
        """Return a rich runtime summary of this suggester.

        Returns:
            A summary of global settings, corpus details, retriever
            configuration, and any artifact provenance available for this
            suggester.
        """
        stored_retrievers: Sequence[StoredRetrieverSpec | None]
        if self._stored_retrievers is None:
            stored_retrievers = [None] * len(self._retriever_specs)
        else:
            stored_retrievers = list(self._stored_retrievers)

        retrievers = [
            _build_retriever_summary(
                spec=spec,
                configured_retriever=configured_retriever,
                stored_retriever=stored_retriever,
            )
            for spec, configured_retriever, stored_retriever in zip(
                self._retriever_specs,
                self._retrievers,
                stored_retrievers,
                strict=True,
            )
        ]

        return SaytConfiguration(
            settings=SaytGlobalSettings(
                min_chars=self._min_chars,
                max_suggestions=self._max_suggestions,
            ),
            corpus=SaytCorpusSummary(
                size=self._corpus.size,
                unique_display_texts=len(self._corpus.display_text_count),
                max_duplication=self._max_duplication,
            ),
            retrievers=retrievers,
            artifact_provenance=(
                self._artifact_provenance.model_copy(deep=True)
                if self._artifact_provenance is not None
                else None
            ),
        )


def _normalised_retriever_specs(
    retriever_specs: Sequence[RetrieverSpec],
) -> list[tuple[RetrieverSpec, float]]:
    """Validate and normalise configured retriever weights."""
    if not retriever_specs:
        raise ValueError("At least one retriever must be configured")

    validated_specs: list[tuple[RetrieverSpec, float]] = []
    for spec in retriever_specs:
        weight = float(spec.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"Retriever '{spec.name}' weight must be a finite value > 0"
            )
        validated_specs.append((spec, weight))

    total_weight = sum(weight for _, weight in validated_specs)
    return [(spec, weight / total_weight) for spec, weight in validated_specs]


def _load_retrievers_from_artifact(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    stored_retrievers: Sequence[StoredRetrieverSpec],
    artifact_dir: Path,
) -> list[_ConfiguredRetriever]:
    """Restore runtime retrievers from a persisted SAYT artifact."""
    normalised_specs = _normalised_retriever_specs(
        [stored_retriever.spec for stored_retriever in stored_retrievers]
    )
    return [
        _ConfiguredRetriever(
            name=stored_retriever.spec.name,
            weight=weight,
            retriever=load_retriever_from_artifact(
                corpus=corpus,
                min_chars=min_chars,
                stored_retriever=stored_retriever,
                artifact_dir=artifact_dir,
            ),
        )
        for (_, weight), stored_retriever in zip(
            normalised_specs,
            stored_retrievers,
            strict=True,
        )
    ]


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_value(item) for item in value]
    return str(value)


def _summarise_retriever_config(spec: RetrieverSpec) -> dict[str, Any]:
    if is_dataclass(spec):
        items = (
            (field.name, getattr(spec, field.name))
            for field in fields(spec)
            if field.name not in {"name", "weight"}
        )
        return {key: _jsonable_value(value) for key, value in items}

    raw_config = getattr(spec, "__dict__", None)
    if isinstance(raw_config, dict):
        return {
            str(key): _jsonable_value(value)
            for key, value in raw_config.items()
            if key not in {"name", "weight"}
        }
    return {}


def _build_retriever_summary(
    *,
    spec: RetrieverSpec,
    configured_retriever: _ConfiguredRetriever,
    stored_retriever: StoredRetrieverSpec | None,
) -> SaytRetrieverSummary:
    artifact_provenance = None
    config = _summarise_retriever_config(spec)
    if stored_retriever is not None:
        artifact_provenance = SaytRetrieverArtifactProvenance(
            artifact_type=stored_retriever.spec.name,
            path=stored_retriever.path,
            config=_summarise_retriever_config(stored_retriever.spec),
        )

    return SaytRetrieverSummary(
        name=spec.name,
        spec_type=type(spec).__name__,
        retriever_type=type(configured_retriever.retriever).__name__,
        configured_weight=float(spec.weight),
        normalised_weight=configured_retriever.weight,
        config=config,
        artifact_provenance=artifact_provenance,
    )
