"""Search-as-you-type (SAYT) utilities.

This module provides a lightweight SAYT implementation for suggesting free-text
survey responses for organisation/industry description questions.
"""

import os
from collections.abc import Iterable
from typing import cast

import pandas as pd
from survey_assist_utils.logging import get_logger

from .sayt_common import CleanCorpus, _normalise
from .sayt_config import SaytConfig
from .sayt_retrievers import (
    NgramRetriever,
    PrefixRetriever,
    SemanticRetriever,
    _Suggestion,
    _take_with_ties,
)

logger = get_logger(__name__)


class SAYTSuggester:
    """Suggests industry/organisation description text as the user types.

    Matching strategies:
    - Prefix: exact prefix, token-prefix, and fuzzy prefix (typo tolerant)
    - N-gram: char n-gram cosine similarity via classifai vectorisers/indexing
    - Semantic: embedding cosine similarity via classifai vectorisers/indexing

    Configuration options allow enabling/disabling strategies, setting their relative
    weights, and tuning their parameters (e.g. n-gram size, fuzzy matching ratio).

    Example parameters:
    - min_chars: minimum query length to trigger suggestions (default: 4)
    - max_suggestions: maximum number of suggestions to return (default: 10)
    - prefix_enable: whether to use prefix matching (default: True)
    - prefix_weight: weight of prefix matching in combined scoring (default: 1.0)
    - ngram_enable: whether to use n-gram matching (default: True)
    - ngram_weight: weight of n-gram matching in combined scoring (default: 1.0)
    - ngram_n: n-gram size (default: 3)
    - ngram_max_df: max document frequency for n-grams (default: 0.2)
    - semantic_enable: whether to use semantic matching (default: True)
    - semantic_weight: weight of semantic matching in combined scoring (default: 1.0)
    - semantic_model: sentence transformer model for semantic matching (default: "all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        corpus: Iterable[tuple[str, str]] | Iterable[str],
        **kwargs: object,
    ) -> None:
        self._corpus = CleanCorpus.model_validate(corpus)
        self._config = SaytConfig.model_validate(
            {**kwargs, "corpus_size": self._corpus.size}
        )
        self._max_duplication = max(self._corpus.display_text_count.values(), default=0)

        self._prefix_retriever = (
            PrefixRetriever(self._corpus, min_chars=self._config.min_chars)
            if self._config.prefix_enable
            else None
        )
        self._ngram_retriever = (
            NgramRetriever(
                self._corpus,
                # using cast to make mypy happy (conditional validation done in SaytConfig)
                n=cast(int, self._config.ngram_n),
                max_df=cast(float, self._config.ngram_max_df),
                min_chars=self._config.min_chars,
            )
            if self._config.ngram_enable
            else None
        )
        self._semantic_retriever = (
            SemanticRetriever(
                self._corpus,
                model=cast(str, self._config.semantic_model),
                min_chars=self._config.min_chars,
            )
            if self._config.semantic_enable
            else None
        )
        logger.info(f"SAYT suggester initialized with config: {self.get_config()}")

    @classmethod
    def from_csv(
        cls,
        file_path: str | os.PathLike,
        *,
        search_text_col: str = "title",
        display_text_col: str | None = None,
        **kwargs,
    ) -> "SAYTSuggester":
        """Alternative constructor to build a SAYTSuggester from a CSV file."""
        df = pd.read_csv(file_path)
        if search_text_col not in df.columns:
            raise ValueError(f"Column '{search_text_col}' not found in CSV")
        if display_text_col is None:
            display_text_col = search_text_col
        if display_text_col not in df.columns:
            raise ValueError(f"Column '{display_text_col}' not found in CSV")
        return cls(list(zip(df[search_text_col], df[display_text_col])), **kwargs)

    def _dedup_suggestions(
        self, suggestions: list[_Suggestion]
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
        *,
        prefix_results: list[_Suggestion],
        ngram_results: list[_Suggestion],
        semantic_results: list[_Suggestion],
    ) -> list[tuple[str, float]]:
        w_prefix = self._config.prefix_weight if self._config.prefix_enable else 0.0
        w_ngram = self._config.ngram_weight if self._config.ngram_enable else 0.0
        w_sem = self._config.semantic_weight if self._config.semantic_enable else 0.0

        def normalise_scores(
            items: list[_Suggestion], weight: float
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

        prefix_norm = normalise_scores(prefix_results, w_prefix)
        ngram_norm = normalise_scores(ngram_results, w_ngram)
        sem_norm = normalise_scores(semantic_results, w_sem)

        combined_scores: dict[str, float] = {}
        for d in (prefix_norm, ngram_norm, sem_norm):
            for k, v in d.items():
                combined_scores[k] = combined_scores.get(k, 0.0) + v

        return [(row_id, float(score)) for row_id, score in combined_scores.items()]

    def suggest_with_scores(
        self, query: str | None, num_suggestions: int | None = None
    ) -> list[_Suggestion]:
        """Return suggestions for the given query, with relevance scores."""
        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions
        q_norm = _normalise(query)
        if len(q_norm) < self._config.min_chars:
            return []

        # Ask for more suggestions, as some may be filtered out after deduplication
        prefix_results = []
        if self._prefix_retriever is not None:
            prefix_results = self._prefix_retriever.suggest_with_scores(
                q_norm, num_suggestions=10 * num_suggestions
            )

        ngram_results = []
        if self._ngram_retriever is not None:
            ngram_results = self._ngram_retriever.suggest_with_scores(
                q_norm, num_suggestions=10 * num_suggestions
            )

        semantic_results = []
        if self._semantic_retriever is not None:
            semantic_results = self._semantic_retriever.suggest_with_scores(
                q_norm, num_suggestions=10 * num_suggestions
            )

        combined_result = self._combine_suggestions(
            prefix_results=prefix_results,
            ngram_results=ngram_results,
            semantic_results=semantic_results,
        )
        ranked_results = _take_with_ties(combined_result, num_suggestions)
        out = [
            _Suggestion(
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
        ranked_results = _take_with_ties(dedup_results, num_suggestions)
        return [result[0] for result in ranked_results]

    def get_config(self) -> SaytConfig:
        """Return the validated configuration of this SAYT suggester instance."""
        return self._config
