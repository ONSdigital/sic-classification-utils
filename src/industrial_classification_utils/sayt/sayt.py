"""Search-as-you-type (SAYT) utilities.

This module provides a lightweight SAYT implementation for suggesting free-text
survey responses for organisation/industry description questions.
"""

# ruff: noqa: PLR2004
# pylint: disable=too-many-instance-attributes,too-many-locals,protected-access,too-few-public-methods,disable=consider-using-with

import os
import re
import tempfile
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import NamedTuple

import numpy as np
import pandas as pd
from classifai.indexers import VectorStore
from classifai.vectorisers import HuggingFaceVectoriser, VectoriserBase
from sklearn.feature_extraction.text import CountVectorizer

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
    semantic_handler: VectoriserBase | str | None = None


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
    display_text: str
    score: float
    search_text: str = ""
    row_id: str = ""


class _L2NormalisingVectoriser(VectoriserBase):
    """Wraps a classifai vectoriser and L2-normalises its outputs.

    ClassifAI's VectorStore search uses dot products; unit-length vectors make this
    equivalent to cosine similarity.
    """

    def __init__(self, base: VectoriserBase) -> None:
        self._base = base

    def transform(self, texts: str | list[str]) -> np.ndarray:
        vectors = self._base.transform(texts)
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class _CharNgramVectoriser(VectoriserBase):
    """CountVectorizer char_wb n-gram vectoriser with unit-length outputs."""

    def __init__(self, corpus: list[str], *, n: int, max_df: float) -> None:
        self._vectoriser = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(n, n),
            max_df=max_df,
        )
        self._vectoriser.fit(corpus)

    def transform(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        matrix = self._vectoriser.transform(texts)
        vectors = matrix.toarray().astype(float, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class SAYTSuggester:
    """Suggests industry/organisation description text as the user types.

    Matching strategies:
    - Prefix: exact prefix, token-prefix, and fuzzy prefix (typo tolerant)
    - N-gram: char n-gram cosine similarity via classifai vectorisers/indexing
    - Semantic: embedding cosine similarity via classifai vectorisers/indexing
    """

    def __init__(
        self,
        corpus: Iterable[tuple[str, str]] | Iterable[str],
        **kwargs: object,
    ) -> None:
        self._build_config(kwargs)
        self._validate_weights()

        # Keep temporary vector-store dirs alive for the instance lifetime.
        self._tmp_dirs: list[tempfile.TemporaryDirectory[str]] = []

        self._clean_corpus(corpus)

        if self._prefix_enable:
            self._init_prefix_index()

        if self._ngram_enable:
            self._init_ngram_index()

        if self._semantic_enable:
            self._init_semantic_index()

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

        if self._ngram_enable:
            self._ngram_n = _coerce_int(cfg["ngram_n"], key="ngram_n")
            if not 2 <= self._ngram_n <= 5:
                raise ValueError("ngram_n must be between 2 and 5")
            self._ngram_max_df = _coerce_float(cfg["ngram_max_df"], key="ngram_max_df")
            if not 0.0 < self._ngram_max_df <= 1.0:
                raise ValueError("ngram_max_df must be in (0, 1]")

        self._semantic_enable = _coerce_bool(
            cfg["semantic_enable"], key="semantic_enable"
        )
        self._semantic_weight = (
            _coerce_float(cfg["semantic_weight"], key="semantic_weight")
            if self._semantic_enable
            else 0.0
        )

        if self._semantic_enable:
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
    ) -> list[tuple[str, str, str]]:
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
            cleaned.append((text, str(item_tuple[1])))
        if not cleaned:
            raise ValueError("corpus is empty after filtering")
        rows: list[tuple[str, str, str]] = [
            (f"{i:06d}", norm, display) for i, (norm, display) in enumerate(cleaned)
        ]

        self._corpus_rows = rows
        self._id_to_search: dict[str, str] = {rid: s for rid, s, _ in rows}
        self._id_to_display: dict[str, str] = {rid: d for rid, _, d in rows}
        self._display_text_count: dict[str, int] = {}
        for _, _, display in rows:
            self._display_text_count[display] = (
                self._display_text_count.get(display, 0) + 1
            )

        return rows

    def _init_prefix_index(self) -> None:
        # Prefix indexes operate over row_ids so duplicate normalised strings remain distinct.
        self._prefix_sorted: list[tuple[str, str]] = []  # (search_norm, row_id)
        self._prefix_terms: list[str] = []
        self._token_prefix_index: dict[str, set[str]] = {}

        for row_id, search_norm, _ in self._corpus_rows:
            self._prefix_sorted.append((search_norm, row_id))
            for token in search_norm.split():
                for i in range(1, min(len(token), len(search_norm)) + 1):
                    prefix = token[:i]
                    self._token_prefix_index.setdefault(prefix, set()).add(row_id)

        self._prefix_sorted.sort(key=lambda x: x[0])
        self._prefix_terms = [s for s, _ in self._prefix_sorted]

    def _build_vector_store(
        self,
        *,
        vectoriser: VectoriserBase,
        name: str,
    ) -> VectorStore:
        tmp = tempfile.TemporaryDirectory(prefix=f"sayt_{name}_")
        self._tmp_dirs.append(tmp)

        csv_path = os.path.join(tmp.name, "corpus.csv")
        out_dir = os.path.join(tmp.name, "vector_store")

        df = pd.DataFrame(
            {
                "id": [r[0] for r in self._corpus_rows],
                "text": [r[1] for r in self._corpus_rows],
                "display": [r[2] for r in self._corpus_rows],
            }
        )
        df.to_csv(csv_path, index=False)

        return VectorStore(
            file_name=csv_path,
            data_type="csv",
            vectoriser=vectoriser,
            batch_size=64,
            meta_data={"display": str},
            output_dir=out_dir,
            overwrite=True,
        )

    @staticmethod
    def _polars_embeddings_to_matrix(embeddings: list[object]) -> np.ndarray:
        # Polars stores per-row embeddings as list/np arrays.
        # Convert to a dense 2D matrix for fast dot products.
        rows = [np.asarray(e, dtype=float) for e in embeddings]
        if not rows:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(rows)

    def _init_ngram_index(self) -> None:
        ngram_vectoriser = _CharNgramVectoriser(
            [search for _, search, _ in self._corpus_rows],
            n=self._ngram_n,
            max_df=self._ngram_max_df,
        )
        self._ngram_vectoriser: VectoriserBase = ngram_vectoriser

        vs = self._build_vector_store(vectoriser=self._ngram_vectoriser, name="ngram")

        self._ngram_doc_ids = vs.vectors["id"].to_list()
        self._ngram_doc_search = vs.vectors["text"].to_list()
        self._ngram_doc_display = vs.vectors["display"].to_list()
        self._ngram_doc_matrix = self._polars_embeddings_to_matrix(
            vs.vectors["embeddings"].to_list()
        )

    def _init_semantic_index(self) -> None:
        handler = self._semantic_handler
        if handler is None:
            base_vectoriser: VectoriserBase = HuggingFaceVectoriser(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        elif isinstance(handler, str):
            base_vectoriser = HuggingFaceVectoriser(handler)
        elif isinstance(handler, VectoriserBase):
            base_vectoriser = handler
        else:
            raise TypeError(
                "semantic_handler must be a classifai VectoriserBase, a model-name string, or None"
            )

        self._semantic_vectoriser: VectoriserBase = _L2NormalisingVectoriser(
            base_vectoriser
        )
        vs = self._build_vector_store(
            vectoriser=self._semantic_vectoriser, name="semantic"
        )

        # Keep the display text from the semantic vector store documents.
        self._semantic_doc_ids = vs.vectors["id"].to_list()
        self._semantic_doc_search = vs.vectors["text"].to_list()
        self._semantic_doc_display = vs.vectors["display"].to_list()
        self._semantic_doc_matrix = self._polars_embeddings_to_matrix(
            vs.vectors["embeddings"].to_list()
        )

    @classmethod
    def from_csv(
        cls,
        file_path: str,
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

    def _get_prefix_suggestions(
        self, q_norm: str, num_suggestions: int | None = None
    ) -> list[_Suggestion]:
        if not self._prefix_enable or (len(q_norm) < self._min_chars):
            return []

        if num_suggestions is None:
            num_suggestions = self._max_suggestions

        scores: dict[str, float] = {}

        left = bisect_left(self._prefix_terms, q_norm)
        right = bisect_right(self._prefix_terms, q_norm + "\uffff")
        for _, row_id in self._prefix_sorted[left:right]:
            scores[row_id] = scores.get(row_id, 0.0) + 3.0

        for row_id in self._token_prefix_index.get(q_norm, set()):
            scores[row_id] = scores.get(row_id, 0.0) + 2.5

        for search_norm, row_id in self._prefix_sorted:
            prefix = search_norm[: len(q_norm)]
            if not prefix:
                continue
            ratio = SequenceMatcher(a=q_norm, b=prefix).ratio()
            if ratio >= _FUZZY_PREFIX_MIN_RATIO:
                scores[row_id] = scores.get(row_id, 0.0) + (2.4 * ratio)

        ranked = sorted(
            scores.items(),
            key=lambda kv: (
                -kv[1],
                self._id_to_search.get(kv[0], ""),
                len(self._id_to_display.get(kv[0], "")),
                self._id_to_display.get(kv[0], "").lower(),
                self._id_to_display.get(kv[0], ""),
                kv[0],
            ),
        )

        out: list[_Suggestion] = []
        for row_id, score in ranked[: self._max_suggestions]:
            out.append(
                _Suggestion(
                    display_text=self._id_to_display.get(row_id, ""),
                    score=float(score),
                    search_text=self._id_to_search.get(row_id, ""),
                    row_id=row_id,
                )
            )
        return out

    def _get_ngram_suggestions(
        self, q_norm: str, num_suggestions: int | None = None
    ) -> list[_Suggestion]:
        if not self._ngram_enable or len(q_norm) < self._min_chars:
            return []

        if num_suggestions is None:
            num_suggestions = self._max_suggestions

        q_vec = self._ngram_vectoriser.transform([q_norm])
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size == 0:
            return []
        q_vec = q_vec.reshape(-1)

        sims = self._ngram_doc_matrix @ q_vec
        if sims.size == 0:
            return []

        k = min(self._max_suggestions, int(sims.size))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[_Suggestion] = []
        for i in top_idx:
            sim = float(sims[int(i)])
            if sim <= 0:
                continue
            row_id = str(self._ngram_doc_ids[int(i)])
            out.append(
                _Suggestion(
                    display_text=str(self._ngram_doc_display[int(i)]),
                    score=sim,
                    search_text=str(self._ngram_doc_search[int(i)]),
                    row_id=row_id,
                )
            )
        return out

    def _get_semantic_suggestions(
        self, q_norm: str, num_suggestions: int | None = None
    ) -> list[_Suggestion]:
        if not self._semantic_enable or len(q_norm) < self._min_chars:
            return []

        if num_suggestions is None:
            num_suggestions = self._max_suggestions

        q_vec = self._semantic_vectoriser.transform([q_norm])
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size == 0:
            return []
        q_vec = q_vec.reshape(-1)

        sims = self._semantic_doc_matrix @ q_vec
        if sims.size == 0:
            return []

        k = min(self._max_suggestions, int(sims.size))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[_Suggestion] = []
        for i in top_idx:
            sim = float(sims[int(i)])
            if sim <= 0:
                continue
            row_id = str(self._semantic_doc_ids[int(i)])
            out.append(
                _Suggestion(
                    display_text=str(self._semantic_doc_display[int(i)]),
                    score=sim,
                    search_text=str(self._semantic_doc_search[int(i)]),
                    row_id=row_id,
                )
            )
        return out

    def _dedup_suggestions(self, suggestions: list[_Suggestion]) -> list[_Suggestion]:
        # sort by score and deduplicate by display text, keeping the highest-scoring variant.
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: (
                -s.score,
                -self._display_text_count.get(s.display_text, 0),
                s.display_text.lower(),
                s.search_text,
                s.row_id,
            ),
        )
        seen: set[str] = set()
        deduped: list[_Suggestion] = []
        for s in sorted_suggestions:
            if s.display_text not in seen:
                deduped.append(s)
                seen.add(s.display_text)
        return deduped

    def _combine_suggestions(
        self,
        *,
        prefix_results: list[_Suggestion],
        ngram_results: list[_Suggestion],
        semantic_results: list[_Suggestion],
    ) -> list[_Suggestion]:
        w_prefix = self._prefix_weight if self._prefix_enable else 0.0
        w_ngram = self._ngram_weight if self._ngram_enable else 0.0
        w_sem = self._semantic_weight if self._semantic_enable else 0.0

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

        return [
            _Suggestion(
                display_text=self._id_to_display.get(row_id, ""),
                score=float(score),
                search_text=self._id_to_search.get(row_id, ""),
                row_id=row_id,
            )
            for row_id, score in combined_scores.items()
        ]

    def suggest_with_scores(
        self, query: str, num_suggestions: int | None = None
    ) -> list[_Suggestion]:
        """Return suggestions for the given query, with relevance scores."""
        if num_suggestions is None:
            num_suggestions = self._max_suggestions
        q_norm = _normalise(query)
        if len(q_norm) < self._min_chars:
            return []

        # Ask for more suggestions, as some may be filtered out after deduplication
        prefix_results = self._get_prefix_suggestions(
            q_norm, num_suggestions=10 * num_suggestions
        )
        ngram_results = self._get_ngram_suggestions(
            q_norm, num_suggestions=10 * num_suggestions
        )
        semantic_results = self._get_semantic_suggestions(
            q_norm, num_suggestions=10 * num_suggestions
        )

        combined_result = self._combine_suggestions(
            prefix_results=prefix_results,
            ngram_results=ngram_results,
            semantic_results=semantic_results,
        )
        ranked_results = self._dedup_suggestions(combined_result)

        return ranked_results[:num_suggestions]

    def suggest(self, query: str, num_suggestions: int | None = None) -> list[str]:
        """Return list of suggestions (display text only) upto a maximum limit."""
        if num_suggestions is None:
            num_suggestions = self._max_suggestions
        return list(
            dict.fromkeys(
                s.display_text
                for s in self.suggest_with_scores(
                    query, num_suggestions=num_suggestions
                )
            )
        )[:num_suggestions]
