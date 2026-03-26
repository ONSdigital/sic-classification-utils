"""Search-as-you-type (SAYT) utilities.

This module provides a lightweight SAYT implementation for suggesting free-text
survey responses for organisation/industry description questions.
"""

# ruff: noqa: PLR2004
# pylint: disable=too-many-instance-attributes,too-many-locals,protected-access

import io
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from industrial_classification_utils.embed.embedding import (
    EmbeddingHandler,
)

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z ]+")
_FUZZY_PREFIX_MIN_RATIO = 0.75


class _SaytDefaults(NamedTuple):
    min_chars: int = 4
    max_suggestions: int = 10
    prefix_enable: bool = True
    prefix_weight: float = 1.0
    ngram_enable: bool = True
    ngram_weight: float = 1.0
    ngram_n: int = 3
    ngram_max_df: float = 0.2
    semantic_enable: bool = True
    semantic_weight: float = 1.0
    semantic_min_chars: int = 10
    semantic_handler: EmbeddingHandler | None = None


_DEFAULT_SAYT_KWARGS = _SaytDefaults()


def _coerce_int(value: object, *, key: str) -> int:
    # bool is a subclass of int; treat it as invalid here.
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an int, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    raise TypeError(f"{key} must be an int")


def _coerce_float(value: object, *, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{key} must be a float, not bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:  # pragma: no cover
            raise TypeError(f"{key} must be a float") from exc
    raise TypeError(f"{key} must be a float")


def _coerce_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    raise TypeError(f"{key} must be a bool")


def _normalise(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass(frozen=True, slots=True)
class _Suggestion:
    text: str
    score: float
    key: str = ""


class SAYTSuggester:
    """Suggests industry/organisation description text as the user types.

    The primary matching strategy is SAYT-style (prefix and token-prefix matches),
    with optional n-gram and semantic matching components.
    """

    def __init__(
        self,
        corpus: Iterable[tuple[str, str]] | Iterable[str],
        **kwargs: object,
    ) -> None:
        """Create a SAYT suggester.

        Args:
                corpus: Iterable of reference terms. Either one, or a tuple of (term, original)
                    where 'term' is used for matching and 'original' is the un-normalised
                    version to return in suggestions.
                **kwargs: Configuration options.

                    Supported keys: `min_chars`, `max_suggestions`,  `prefix_enable`,
                        `prefix_weight`, `ngram_enable`, `ngram_weight`, `ngram_n`,
                        `ngram_max_df`, `semantic_enable`,  `semantic_weight`,
                        `semantic_min_chars`, `semantic_handler`.
        """
        self._build_config(kwargs)
        self._validate_weights()

        cleaned = self._clean_corpus(corpus)
        self._init_prefix_index(cleaned)

        if self._semantic_enable:
            self._init_semantic_index(cleaned)

        if self._ngram_enable:
            self._init_ngram_index(cleaned)

    def _build_config(self, kwargs: dict[str, object]) -> None:
        defaults = _DEFAULT_SAYT_KWARGS._asdict()
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError(
                f"Unexpected keyword argument(s): {', '.join(sorted(unknown))}"
            )
        cfg = {**defaults, **kwargs}
        self._min_chars = _coerce_int(cfg["min_chars"], key="min_chars")
        if self._min_chars < 3:
            raise ValueError("min_chars must be >= 3")
        self._max_suggestions = _coerce_int(
            cfg["max_suggestions"], key="max_suggestions"
        )
        if (self._max_suggestions < 1) | (self._max_suggestions > 100):
            raise ValueError("max_suggestions must be between 1 and 100")

        self._prefix_enable = _coerce_bool(cfg["prefix_enable"], key="prefix_enable")
        self._prefix_weight = (
            _coerce_float(cfg["prefix_weight"], key="prefix_weight")
            if self._prefix_enable
            else 0.0
        )
        self._ngram_enable = _coerce_bool(cfg["ngram_enable"], key="ngram_enable")
        self._ngram_weight = (
            _coerce_float(cfg["ngram_weight"], key="ngram_weight")
            if self._ngram_enable
            else 0.0
        )
        self._semantic_enable = _coerce_bool(
            cfg["semantic_enable"], key="semantic_enable"
        )
        self._semantic_weight = (
            _coerce_float(cfg["semantic_weight"], key="semantic_weight")
            if self._semantic_enable
            else 0.0
        )

        if self._ngram_enable:
            self._ngram_n = _coerce_int(cfg["ngram_n"], key="ngram_n")
            if not 2 <= self._ngram_n <= 5:
                raise ValueError("ngram_n must be between 2 and 5")
            self._ngram_max_df = _coerce_float(cfg["ngram_max_df"], key="ngram_max_df")
            if not 0.0 < self._ngram_max_df <= 1.0:
                raise ValueError("ngram_max_df must be in (0, 1]")

        if self._semantic_enable:
            self._semantic_min_chars = _coerce_int(
                cfg["semantic_min_chars"], key="semantic_min_chars"
            )
            self._semantic_handler = cfg.get("semantic_handler")

    def _validate_weights(self) -> None:
        if self._prefix_enable and self._prefix_weight <= 0:
            raise ValueError("prefix_weight must be > 0")
        if not self._prefix_enable:
            self._prefix_weight = 0.0

        if self._ngram_enable and self._ngram_weight <= 0:
            raise ValueError("ngram_weight must be > 0")
        if not self._ngram_enable:
            self._ngram_weight = 0.0

        if self._semantic_enable and self._semantic_weight <= 0:
            raise ValueError("semantic_weight must be > 0")
        if not self._semantic_enable:
            self._semantic_weight = 0.0

        total_weight = self._prefix_weight + self._ngram_weight + self._semantic_weight
        if total_weight == 0.0:
            raise ValueError(
                "At least one of prefix, ngram, or semantic must be enabled with a positive weight"
            )
        self._prefix_weight /= total_weight
        self._ngram_weight /= total_weight
        self._semantic_weight /= total_weight

    def _clean_corpus(
        self, corpus: Iterable[str] | Iterable[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        if not isinstance(corpus, Iterable):
            raise TypeError(
                "corpus must be an iterable of strings or (string, original) tuples"
            )
        cleaned: list[tuple[str, str]] = []
        for item in corpus:
            item_tuple = item if isinstance(item, tuple) else (item, item)
            text = _normalise(item_tuple[0])
            if not text or text == "-9":
                continue
            cleaned.append((text, item_tuple[1]))
        if not cleaned:
            raise ValueError("corpus is empty after filtering")
        return cleaned

    def _init_prefix_index(self, corpus: list[tuple[str, str]]) -> None:
        self._norm_to_original: dict[str, str] = {}
        self._norm_sorted: list[str] = []
        self._token_prefix_index: dict[str, set[str]] = {}
        self._counts: dict[str, int] = {}

        for norm, original in corpus:
            if norm in self._norm_to_original:
                self._counts[norm] += 1
            else:
                self._norm_to_original[norm] = original
                self._norm_sorted.append(norm)
                self._counts[norm] = 1
                for token in norm.split():
                    for i in range(1, min(len(token), len(norm)) + 1):
                        prefix = token[:i]
                        if prefix not in self._token_prefix_index:
                            self._token_prefix_index[prefix] = set()
                        self._token_prefix_index[prefix].add(norm)

        self._norm_sorted.sort()

    def _init_ngram_index(self, corpus) -> None:
        self._ngram_vectoriser = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(self._ngram_n, self._ngram_n),
            max_df=self._ngram_max_df,
        )
        self._ngram_terms = [t[0] for t in corpus]
        self._ngram_matrix = self._ngram_vectoriser.fit_transform(self._ngram_terms)
        # Precompute row norms for cosine similarity.
        self._ngram_row_norms = np.sqrt(
            self._ngram_matrix.multiply(self._ngram_matrix).sum(axis=1)
        ).A1

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        *,
        search_text_col: str = "sic2007_employee",
        display_text_col: str | None = None,
        **kwargs,
    ) -> "SAYTSuggester":
        """Build the suggester from a CSV file column.

        Args:
                file_path: Path to CSV.
                search_text_col: Column name containing historic survey responses.
                display_text_col: Column name for display text.
                    If None, defaults to `search_text_col`.
                **kwargs: Forwarded to the constructor.
        """
        df = pd.read_csv(file_path)
        if search_text_col not in df.columns:
            raise ValueError(f"Column '{search_text_col}' not found in CSV")
        if display_text_col is None:
            display_text_col = search_text_col
        if display_text_col not in df.columns:
            raise ValueError(f"Column '{display_text_col}' not found in CSV")
        return cls(list(zip(df[search_text_col], df[display_text_col])), **kwargs)

    def _get_prefix_suggestions(self, query: str) -> list[_Suggestion]:
        if not isinstance(query, str):
            return []
        q = _normalise(query)
        if len(q) < self._min_chars:
            return []

        scores: dict[str, float] = {}

        # 1) Full string prefix match (fast range query).
        left = bisect_left(self._norm_sorted, q)
        right = bisect_right(self._norm_sorted, q + "\uffff")
        for norm in self._norm_sorted[left:right]:
            scores[norm] = scores.get(norm, 0.0) + 3.0

        # 2) Token prefix match.
        for norm in self._token_prefix_index.get(q, set()):
            scores[norm] = scores.get(norm, 0.0) + 2.5

        # 3) Fuzzy prefix (typo-tolerant): compare query to the start of the term.
        for norm in self._norm_to_original:
            prefix = norm[: len(q)]
            if not prefix:
                continue
            ratio = SequenceMatcher(a=q, b=prefix).ratio()
            if ratio >= _FUZZY_PREFIX_MIN_RATIO:
                scores[norm] = scores.get(norm, 0.0) + (2.4 * ratio)

        ranked = sorted(
            scores.items(),
            key=lambda kv: (
                -kv[1],
                -self._counts.get(kv[0], 0),
                self._norm_to_original[kv[0]],
            ),
        )

        out: list[_Suggestion] = []
        for norm, score in ranked[: self._max_suggestions]:
            out.append(
                _Suggestion(
                    text=self._norm_to_original[norm],
                    score=float(score),
                    key=norm,
                )
            )
        return out

    def _get_ngram_suggestions(self, q_norm: str) -> list[_Suggestion]:
        if not self._ngram_enable:
            return []

        q_vec = self._ngram_vectoriser.transform([q_norm])
        if q_vec.nnz == 0:
            return []

        q_size = float(np.sqrt(q_vec.multiply(q_vec).sum()))
        if q_size == 0.0:
            return []

        dots = (self._ngram_matrix @ q_vec.T).toarray().ravel()
        denom = (self._ngram_row_norms * q_size) + 1e-12
        sims = dots / denom

        k = min(self._max_suggestions, sims.size)
        if k <= 0:
            return []

        # Get top-k indices.
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        weight = float(getattr(self, "_ngram_weight", 1.0))
        out: list[_Suggestion] = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim <= 0:
                continue
            norm = self._ngram_terms[int(idx)]
            # norm is already normalised.
            display = self._norm_to_original.get(norm, norm)
            out.append(_Suggestion(text=display, score=weight * sim, key=norm))
        return out

    def _init_semantic_index(self, corpus: list[tuple[str, str]]) -> None:
        if self._semantic_handler is None:
            self._semantic_handler = EmbeddingHandler(
                k_matches=self._max_suggestions, db_dir=None
            )

        # Rebuild from empty so results correspond to the provided corpus.
        with io.StringIO() as file_object:
            for i, norm in enumerate(corpus):
                # EmbeddingHandler expects "code: description" per line.
                code = f"{i:05d}"
                file_object.write(f"{code}:{norm[0]}\n")
            file_object.seek(0)
            self._semantic_handler.embed_index(from_empty=True, file_object=file_object)
        if self._semantic_handler._index_size == 0:
            raise ValueError("Semantic embedding index is empty after initialization")

    def _get_semantic_suggestions(self, query: str) -> list[_Suggestion]:
        """Semantic suggestions using EmbeddingHandler."""
        # EmbeddingHandler.search_index returns dicts with a "title" key holding
        # the embedded page_content.
        if not self._semantic_enable or not self._semantic_handler:
            raise RuntimeError(
                "Semantic search is not enabled or handler is not initialized"
            )
        results = self._semantic_handler.search_index(query, return_dicts=True)
        suggestions: list[_Suggestion] = []
        for r in results:
            title = r.get("title")
            if not title:
                continue
            norm_key = _normalise(str(title))
            if not norm_key:
                continue
            distance = float(r.get("distance", 0.0))
            score = 0.5 + (1.0 / (1.0 + max(distance, 0.0)))
            suggestions.append(
                _Suggestion(
                    text=self._norm_to_original.get(norm_key, str(title).strip()),
                    score=score,
                    key=norm_key,
                )
            )
        return suggestions

    def _rank_suggestions(
        self,
        prefix_results: list[_Suggestion],
        ngram_results: list[_Suggestion],
        semantic_results: list[_Suggestion],
    ) -> list[_Suggestion]:
        # Relative weights: prefix is baseline=1.0, others configurable.
        w_prefix = 1.0
        w_ngram = (
            float(getattr(self, "_ngram_weight", 1.0)) if self._ngram_enable else 0.0
        )
        w_sem = (
            float(getattr(self, "_semantic_weight", 1.0))
            if self._semantic_enable
            else 0.0
        )

        def normalise_scores(items: list[_Suggestion]) -> dict[str, float]:
            if not items:
                return {}
            max_score = max((float(s.score) for s in items), default=0.0)
            if max_score <= 0:
                return {}
            out: dict[str, float] = {}
            for s in items:
                key = s.key or _normalise(s.text)
                if not key:
                    continue
                out[key] = max(out.get(key, 0.0), float(s.score) / max_score)
            return out

        prefix_norm = normalise_scores(prefix_results)
        ngram_norm = normalise_scores(ngram_results)
        sem_norm = normalise_scores(semantic_results)

        combined: dict[str, float] = {}
        for key, score in prefix_norm.items():
            combined[key] = combined.get(key, 0.0) + w_prefix * score
        for key, score in ngram_norm.items():
            combined[key] = combined.get(key, 0.0) + w_ngram * score
        for key, score in sem_norm.items():
            combined[key] = combined.get(key, 0.0) + w_sem * score

        ranked = sorted(
            combined.items(),
            key=lambda kv: (
                -kv[1],
                -self._counts.get(kv[0], 0),
                self._norm_to_original.get(kv[0], kv[0]),
            ),
        )

        out: list[_Suggestion] = []
        for norm, score in ranked[: self._max_suggestions]:
            out.append(
                _Suggestion(
                    text=self._norm_to_original.get(norm, norm),
                    score=float(score),
                    key=norm,
                )
            )
        return out

    def suggest_with_scores(
        self,
        query: str,
    ) -> list[_Suggestion]:
        """Return up to `limit` suggestions with scores.

        Args:
                query: User-typed query.
        """
        q_norm = _normalise(query)
        if len(q_norm) < self._min_chars:
            return []

        prefix_results = self._get_prefix_suggestions(q_norm)
        ngram_results = (
            self._get_ngram_suggestions(q_norm) if self._ngram_enable else []
        )
        semantic_results = (
            self._get_semantic_suggestions(q_norm) if self._semantic_enable else []
        )
        combined_results = self._rank_suggestions(
            prefix_results=prefix_results,
            ngram_results=ngram_results,
            semantic_results=semantic_results,
        )

        return combined_results

    def suggest(self, query: str) -> list[str]:
        """Return up list of suggestions (display text only) upto a maximum limit."""
        return list(dict.fromkeys(s.text for s in self.suggest_with_scores(query)))[
            : self._max_suggestions
        ]
