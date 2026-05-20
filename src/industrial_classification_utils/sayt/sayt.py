"""Search-as-you-type (SAYT) utilities.

This module provides a lightweight SAYT implementation for suggesting free-text
survey responses for organisation/industry description questions.
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd
from survey_assist_utils.logging import get_logger

from .sayt_core import (
    CleanCorpus,
    SaytConfig,
    Suggestion,
    _normalise,
    take_with_ties,
)
from .sayt_retriever_specs import (
    Retriever,
    RetrieverSpec,
    default_retriever_specs,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _ConfiguredRetriever:
    """Runtime retriever binding with its configured contribution weight."""

    name: str
    weight: float
    retriever: Retriever


class SAYTSuggester:
    """Suggests industry/organisation description text as the user types.

    Matching strategies:
    - The suggester orchestrates whichever retrievers are supplied at construction time.
    - By default it uses prefix, n-gram, and semantic retriever specs.
    - Each retriever carries its own configuration and relative weight.

    Example parameters:
    - min_chars: minimum query length to trigger suggestions (default: 4)
    - max_suggestions: maximum number of suggestions to return (default: 10)
    - retrievers: optional explicit retriever specs; defaults to the standard set
    """

    def __init__(
        self,
        corpus: Iterable[tuple[str, str]] | Iterable[str],
        *,
        retrievers: Sequence[RetrieverSpec] | None = None,
        **kwargs: object,
    ) -> None:
        self._corpus = CleanCorpus.model_validate(corpus)
        self._config = SaytConfig.model_validate(kwargs)
        self._max_duplication = max(self._corpus.display_text_count.values(), default=0)

        self._retriever_specs = tuple(
            default_retriever_specs() if retrievers is None else retrievers
        )
        self._retrievers = self._build_retrievers(self._retriever_specs)
        logger.info(f"SAYT suggester initialized with config: {self.get_config()}")

    def _build_retrievers(
        self, retriever_specs: Sequence[RetrieverSpec]
    ) -> list[_ConfiguredRetriever]:
        if not retriever_specs:
            raise ValueError("At least one retriever must be configured")

        total_weight = sum(float(spec.weight) for spec in retriever_specs)
        if total_weight <= 0:
            raise ValueError("At least one retriever must have a positive weight")

        return [
            _ConfiguredRetriever(
                name=spec.name,
                weight=float(spec.weight) / total_weight,
                retriever=spec.build(
                    self._corpus,
                    min_chars=self._config.min_chars,
                ),
            )
            for spec in retriever_specs
        ]

    @classmethod
    def from_csv(
        cls,
        file_path: str | os.PathLike,
        *,
        search_text_col: str = "title",
        display_text_col: str | None = None,
        retrievers: Sequence[RetrieverSpec] | None = None,
        **kwargs: object,
    ) -> "SAYTSuggester":
        """Alternative constructor to build a SAYTSuggester from a CSV file."""
        df = pd.read_csv(file_path)
        if search_text_col not in df.columns:
            raise ValueError(f"Column '{search_text_col}' not found in CSV")
        if display_text_col is None:
            display_text_col = search_text_col
        if display_text_col not in df.columns:
            raise ValueError(f"Column '{display_text_col}' not found in CSV")
        return cls(
            list(zip(df[search_text_col], df[display_text_col], strict=False)),
            retrievers=retrievers,
            **kwargs,
        )

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
        """Return suggestions for the given query, with relevance scores."""
        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions
        q_norm = _normalise(query)
        if len(q_norm) < self._config.min_chars:
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
        """Return list of suggestions (display text only) upto a maximum limit."""
        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions
        results = self.suggest_with_scores(
            query, num_suggestions=num_suggestions * self._max_duplication
        )
        dedup_results = self._dedup_suggestions(results)
        ranked_results = take_with_ties(dedup_results, num_suggestions)
        return [result[0] for result in ranked_results]

    def get_config(self) -> SaytConfig:
        """Return the validated configuration of this SAYT suggester instance."""
        return self._config.model_copy(deep=True)
