"""Search-as-you-type (SAYT) utilities.

This module provides a lightweight SAYT implementation for suggesting free-text
survey responses for organisation/industry description questions.
"""

# pylint: disable=too-many-instance-attributes,too-few-public-methods,consider-using-with

import os
import tempfile
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
from classifai.indexers import VectorStore
from classifai.vectorisers import HuggingFaceVectoriser, VectoriserBase
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from survey_assist_utils.logging import get_logger

from .sayt_common import CleanCorpus, Suggestion, _normalise
from .sayt_config import SaytConfig

logger = get_logger(__name__)
_FUZZY_PREFIX_MIN_RATIO = 0.75


@dataclass
class _PrefixIndex:
    sorted_terms: list[tuple[str, str]]
    prefix_terms: list[str]
    token_index: dict[str, set[str]]


@dataclass
class _DenseVectorIndex:
    vectoriser: VectoriserBase
    doc_ids: list[object]
    doc_search: list[object]
    doc_display: list[object]
    doc_matrix: np.ndarray


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
        matrix = cast(csr_matrix, self._vectoriser.transform(texts))
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
        self._corpus = CleanCorpus.model_validate(corpus)
        self._config = SaytConfig.model_validate(
            {**kwargs, "corpus_size": self._corpus.size}
        )

        # Keep temporary vector-store dirs alive for the instance lifetime.
        self._tmp_dirs: list[tempfile.TemporaryDirectory[str]] = []

        self._prefix_index = (
            self._init_prefix_index() if self._config.prefix_enable else None
        )
        self._ngram_index = (
            self._init_ngram_index() if self._config.ngram_enable else None
        )
        self._semantic_index = (
            self._init_semantic_index() if self._config.semantic_enable else None
        )
        logger.info(f"SAYT suggester initialized with config: {self.get_config()}")

    def _init_prefix_index(self) -> _PrefixIndex:
        # Prefix indexes operate over row_ids so duplicate normalised strings remain distinct.
        prefix = _PrefixIndex(sorted_terms=[], prefix_terms=[], token_index={})

        for row_id, search_norm, _ in self._corpus.rows:
            prefix.sorted_terms.append((search_norm, row_id))
            for token in search_norm.split():
                for i in range(1, min(len(token), len(search_norm)) + 1):
                    prefix_str = token[:i]
                    prefix.token_index.setdefault(prefix_str, set()).add(row_id)

        prefix.sorted_terms.sort(key=lambda x: x[0])
        prefix.prefix_terms = [s for s, _ in prefix.sorted_terms]
        return prefix

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
                "label": [r[0] for r in self._corpus.rows],
                "text": [r[1] for r in self._corpus.rows],
                "display": [r[2] for r in self._corpus.rows],
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
    def _get_vector_store_vectors(vector_store: VectorStore) -> pl.DataFrame:
        vectors = vector_store.vectors
        if vectors is None:
            raise RuntimeError("VectorStore was created without vectors")
        return vectors

    @staticmethod
    def _polars_embeddings_to_matrix(embeddings: list[object]) -> np.ndarray:
        # Polars stores per-row embeddings as list/np arrays.
        # Convert to a dense 2D matrix for fast dot products.
        rows = [np.asarray(e, dtype=float) for e in embeddings]
        if not rows:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(rows)

    def _init_ngram_index(self) -> _DenseVectorIndex:
        ngram_vectoriser = _CharNgramVectoriser(
            [search for _, search, _ in self._corpus.rows],
            n=self._config.ngram_n,
            max_df=self._config.ngram_max_df,
        )
        vs = self._build_vector_store(vectoriser=ngram_vectoriser, name="ngram")
        vectors = self._get_vector_store_vectors(vs)

        return _DenseVectorIndex(
            vectoriser=ngram_vectoriser,
            doc_ids=vectors["label"].to_list(),
            doc_search=vectors["text"].to_list(),
            doc_display=vectors["display"].to_list(),
            doc_matrix=self._polars_embeddings_to_matrix(
                vectors["embeddings"].to_list()
            ),
        )

    def _init_semantic_index(self) -> _DenseVectorIndex:
        base_vectoriser: VectoriserBase = HuggingFaceVectoriser(
            f"sentence-transformers/{self._config.semantic_model}"
        )

        semantic_vectoriser: VectoriserBase = _L2NormalisingVectoriser(base_vectoriser)
        vs = self._build_vector_store(vectoriser=semantic_vectoriser, name="semantic")
        vectors = self._get_vector_store_vectors(vs)

        # Keep the display text from the semantic vector store documents.
        return _DenseVectorIndex(
            vectoriser=semantic_vectoriser,
            doc_ids=vectors["label"].to_list(),
            doc_search=vectors["text"].to_list(),
            doc_display=vectors["display"].to_list(),
            doc_matrix=self._polars_embeddings_to_matrix(
                vectors["embeddings"].to_list()
            ),
        )

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

    def _get_prefix_suggestions(
        self, q_norm: str, num_suggestions: int | None = None
    ) -> list[Suggestion]:
        if not self._prefix_index or len(q_norm) < self._config.min_chars:
            return []

        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions

        scores: dict[str, float] = {}

        left = bisect_left(self._prefix_index.prefix_terms, q_norm)
        right = bisect_right(self._prefix_index.prefix_terms, q_norm + "\uffff")
        for _, row_id in self._prefix_index.sorted_terms[left:right]:
            scores[row_id] = scores.get(row_id, 0.0) + 3.0

        for row_id in self._prefix_index.token_index.get(q_norm, set()):
            scores[row_id] = scores.get(row_id, 0.0) + 2.5

        for search_norm, row_id in self._prefix_index.sorted_terms:
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
                self._corpus.id_to_search.get(kv[0], ""),
                len(self._corpus.id_to_display.get(kv[0], "")),
                self._corpus.id_to_display.get(kv[0], "").lower(),
                self._corpus.id_to_display.get(kv[0], ""),
                kv[0],
            ),
        )

        out: list[Suggestion] = []
        for row_id, score in ranked[:num_suggestions]:
            out.append(
                Suggestion(
                    display_text=self._corpus.id_to_display.get(row_id, ""),
                    score=float(score),
                    search_text=self._corpus.id_to_search.get(row_id, ""),
                    row_id=row_id,
                )
            )
        return out

    def _get_ngram_suggestions(
        self, q_norm: str, num_suggestions: int | None = None
    ) -> list[Suggestion]:
        if not self._ngram_index or len(q_norm) < self._config.min_chars:
            return []

        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions

        q_vec = self._ngram_index.vectoriser.transform([q_norm])
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size == 0:
            return []
        q_vec = q_vec.reshape(-1)

        sims = self._ngram_index.doc_matrix @ q_vec
        if sims.size == 0:
            return []

        k = min(num_suggestions, int(sims.size))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[Suggestion] = []
        for i in top_idx:
            sim = float(sims[int(i)])
            if sim <= 0:
                continue
            row_id = str(self._ngram_index.doc_ids[int(i)])
            out.append(
                Suggestion(
                    display_text=str(self._ngram_index.doc_display[int(i)]),
                    score=sim,
                    search_text=str(self._ngram_index.doc_search[int(i)]),
                    row_id=row_id,
                )
            )
        return out

    def _get_semantic_suggestions(
        self, q_norm: str, num_suggestions: int | None = None
    ) -> list[Suggestion]:
        if not self._semantic_index or len(q_norm) < self._config.min_chars:
            return []

        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions

        q_vec = self._semantic_index.vectoriser.transform([q_norm])
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size == 0:
            return []
        q_vec = q_vec.reshape(-1)

        sims = self._semantic_index.doc_matrix @ q_vec
        if sims.size == 0:
            return []

        k = min(num_suggestions, int(sims.size))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[Suggestion] = []
        for i in top_idx:
            sim = float(sims[int(i)])
            if sim <= 0:
                continue
            row_id = str(self._semantic_index.doc_ids[int(i)])
            out.append(
                Suggestion(
                    display_text=str(self._semantic_index.doc_display[int(i)]),
                    score=sim,
                    search_text=str(self._semantic_index.doc_search[int(i)]),
                    row_id=row_id,
                )
            )
        return out

    def _dedup_suggestions(self, suggestions: list[Suggestion]) -> list[Suggestion]:
        # sort by score and deduplicate by display text, keeping the highest-scoring variant.
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: (
                -s.score,
                -self._corpus.display_text_count.get(s.display_text, 0),
                s.display_text.lower(),
                s.search_text,
                s.row_id,
            ),
        )
        seen: set[str] = set()
        deduped: list[Suggestion] = []
        for s in sorted_suggestions:
            if s.display_text not in seen:
                deduped.append(s)
                seen.add(s.display_text)
        return deduped

    def _combine_suggestions(
        self,
        *,
        prefix_results: list[Suggestion],
        ngram_results: list[Suggestion],
        semantic_results: list[Suggestion],
    ) -> list[Suggestion]:
        w_prefix = self._config.prefix_weight if self._config.prefix_enable else 0.0
        w_ngram = self._config.ngram_weight if self._config.ngram_enable else 0.0
        w_sem = self._config.semantic_weight if self._config.semantic_enable else 0.0

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

        prefix_norm = normalise_scores(prefix_results, w_prefix)
        ngram_norm = normalise_scores(ngram_results, w_ngram)
        sem_norm = normalise_scores(semantic_results, w_sem)

        combined_scores: dict[str, float] = {}
        for d in (prefix_norm, ngram_norm, sem_norm):
            for k, v in d.items():
                combined_scores[k] = combined_scores.get(k, 0.0) + v

        return [
            Suggestion(
                display_text=self._corpus.id_to_display.get(row_id, ""),
                score=float(score),
                search_text=self._corpus.id_to_search.get(row_id, ""),
                row_id=row_id,
            )
            for row_id, score in combined_scores.items()
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

    def suggest(
        self, query: str | None, num_suggestions: int | None = None
    ) -> list[str]:
        """Return list of suggestions (display text only) upto a maximum limit."""
        if num_suggestions is None:
            num_suggestions = self._config.max_suggestions
        return list(
            dict.fromkeys(
                s.display_text
                for s in self.suggest_with_scores(
                    query, num_suggestions=num_suggestions
                )
            )
        )[:num_suggestions]

    def get_config(self) -> SaytConfig:
        """Return the validated configuration of this SAYT suggester instance."""
        return self._config
