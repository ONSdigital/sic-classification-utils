"""Private retrieval strategies for SAYT suggesters."""

# pylint: disable=too-few-public-methods,R0801

import os
import tempfile
from bisect import bisect_left, bisect_right
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

from .sayt_common import CleanCorpus

_FUZZY_PREFIX_MIN_RATIO = 0.75


@dataclass(frozen=True, slots=True)
class _Suggestion:
    """Internal suggestion record with score and row metadata."""

    display_text: str
    score: float
    search_text: str = ""
    row_id: str = ""


@dataclass(frozen=True, slots=True)
class _PrefixIndex:
    """Precomputed prefix lookup structures for prefix matching."""

    sorted_terms: list[tuple[str, str]]
    prefix_terms: list[str]
    token_index: dict[str, set[str]]


@dataclass(frozen=True, slots=True)
class _DenseVectorIndex:
    """In-memory dense vector search index built from the corpus."""

    vectoriser: VectoriserBase
    doc_ids: list[object]
    doc_search: list[object]
    doc_display: list[object]
    doc_matrix: np.ndarray


class _L2NormalisingVectoriser(VectoriserBase):
    """Wraps a classifai vectoriser and L2-normalises its outputs."""

    def __init__(self, base: VectoriserBase) -> None:
        """Store the wrapped vectoriser used for raw embeddings."""
        self._base = base

    def transform(self, texts: str | list[str]) -> np.ndarray:
        """Vectorise inputs and scale each output row to unit length."""
        vectors = self._base.transform(texts)
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


class _CharNgramVectoriser(VectoriserBase):
    """CountVectorizer char_wb n-gram vectoriser with unit-length outputs."""

    def __init__(self, corpus: list[str], *, n: int, max_df: float) -> None:
        """Fit a character n-gram vectoriser on the normalised corpus."""
        self._vectoriser = CountVectorizer(
            analyzer="char_wb",
            ngram_range=(n, n),
            max_df=max_df,
        )
        self._vectoriser.fit(corpus)

    def transform(self, texts: str | list[str]) -> np.ndarray:
        """Return unit-length dense n-gram vectors for the provided texts."""
        if isinstance(texts, str):
            texts = [texts]
        matrix = cast(csr_matrix, self._vectoriser.transform(texts))
        vectors = matrix.toarray().astype(float, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


def _build_dense_vector_index(
    *,
    corpus: CleanCorpus,
    vectoriser: VectoriserBase,
    name: str,
) -> _DenseVectorIndex:
    """Build a temporary VectorStore and materialise its vectors in memory."""
    with tempfile.TemporaryDirectory(prefix=f"sayt_{name}_") as tmp_dir:
        csv_path = os.path.join(tmp_dir, "corpus.csv")
        out_dir = os.path.join(tmp_dir, "vector_store")

        df = pd.DataFrame(
            {
                "label": [r[0] for r in corpus.rows],
                "text": [r[1] for r in corpus.rows],
                "display": [r[2] for r in corpus.rows],
            }
        )
        df.to_csv(csv_path, index=False)

        vector_store = VectorStore(
            file_name=csv_path,
            data_type="csv",
            vectoriser=vectoriser,
            batch_size=64,
            meta_data={"display": str},
            output_dir=out_dir,
            overwrite=True,
        )
        vectors = _get_vector_store_vectors(vector_store)
        return _DenseVectorIndex(
            vectoriser=vectoriser,
            doc_ids=vectors["label"].to_list(),
            doc_search=vectors["text"].to_list(),
            doc_display=vectors["display"].to_list(),
            doc_matrix=_polars_embeddings_to_matrix(vectors["embeddings"].to_list()),
        )


def _get_vector_store_vectors(vector_store: VectorStore) -> pl.DataFrame:
    """Return vector-store rows or fail if vectors were not materialised."""
    vectors = vector_store.vectors
    if vectors is None:
        raise RuntimeError("VectorStore was created without vectors")
    return vectors


def _polars_embeddings_to_matrix(embeddings: list[object]) -> np.ndarray:
    """Convert a list of embedding arrays into a 2D NumPy matrix."""
    rows = [np.asarray(e, dtype=float) for e in embeddings]
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows)


class PrefixRetriever:
    """Retrieve suggestions using exact, token, and fuzzy prefix matching."""

    def __init__(self, corpus: CleanCorpus, *, min_chars: int) -> None:
        """Build the prefix index for the cleaned corpus."""
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = self._build_index(corpus)

    @staticmethod
    def _build_index(corpus: CleanCorpus) -> _PrefixIndex:
        """Precompute sorted prefix terms and token-prefix lookup tables."""
        sorted_terms: list[tuple[str, str]] = []
        token_index: dict[str, set[str]] = {}

        for row_id, search_norm, _ in corpus.rows:
            sorted_terms.append((search_norm, row_id))
            for token in search_norm.split():
                for i in range(1, min(len(token), len(search_norm)) + 1):
                    prefix_str = token[:i]
                    token_index.setdefault(prefix_str, set()).add(row_id)

        sorted_terms.sort(key=lambda x: x[0])
        prefix_terms = [s for s, _ in sorted_terms]
        return _PrefixIndex(
            sorted_terms=sorted_terms,
            prefix_terms=prefix_terms,
            token_index=token_index,
        )

    def suggest(self, q_norm: str, num_suggestions: int) -> list[_Suggestion]:
        """Return ranked prefix-based suggestions for a normalised query."""
        if len(q_norm) < self._min_chars:
            return []

        scores: dict[str, float] = {}

        left = bisect_left(self._index.prefix_terms, q_norm)
        right = bisect_right(self._index.prefix_terms, q_norm + "\uffff")
        for _, row_id in self._index.sorted_terms[left:right]:
            scores[row_id] = scores.get(row_id, 0.0) + 3.0

        for row_id in self._index.token_index.get(q_norm, set()):
            scores[row_id] = scores.get(row_id, 0.0) + 2.5

        for search_norm, row_id in self._index.sorted_terms:
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

        return [
            _Suggestion(
                display_text=self._corpus.id_to_display.get(row_id, ""),
                score=float(score),
                search_text=self._corpus.id_to_search.get(row_id, ""),
                row_id=row_id,
            )
            for row_id, score in ranked[:num_suggestions]
        ]


class NgramRetriever:
    """Retrieve suggestions using character n-gram similarity."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        n: int,
        max_df: float,
        min_chars: int,
    ) -> None:
        """Build a dense n-gram index for the cleaned corpus."""
        self._min_chars = min_chars
        ngram_vectoriser = _CharNgramVectoriser(
            [search for _, search, _ in corpus.rows],
            n=n,
            max_df=max_df,
        )
        self._index = _build_dense_vector_index(
            corpus=corpus,
            vectoriser=ngram_vectoriser,
            name="ngram",
        )

    def suggest(self, q_norm: str, num_suggestions: int) -> list[_Suggestion]:
        """Return the top cosine-similar n-gram suggestions for a query."""
        if len(q_norm) < self._min_chars:
            return []

        q_vec = self._index.vectoriser.transform([q_norm])
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size == 0:
            return []
        q_vec = q_vec.reshape(-1)

        sims = self._index.doc_matrix @ q_vec
        if sims.size == 0:
            return []

        k = min(num_suggestions, int(sims.size))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[_Suggestion] = []
        for i in top_idx:
            sim = float(sims[int(i)])
            if sim <= 0:
                continue
            row_id = str(self._index.doc_ids[int(i)])
            out.append(
                _Suggestion(
                    display_text=str(self._index.doc_display[int(i)]),
                    score=sim,
                    search_text=str(self._index.doc_search[int(i)]),
                    row_id=row_id,
                )
            )
        return out


class SemanticRetriever:
    """Retrieve suggestions using sentence-transformer embeddings."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        model: str,
        min_chars: int,
    ) -> None:
        """Build a dense semantic index for the cleaned corpus."""
        self._min_chars = min_chars
        base_vectoriser: VectoriserBase = HuggingFaceVectoriser(
            f"sentence-transformers/{model}"
        )
        semantic_vectoriser: VectoriserBase = _L2NormalisingVectoriser(base_vectoriser)
        self._index = _build_dense_vector_index(
            corpus=corpus,
            vectoriser=semantic_vectoriser,
            name="semantic",
        )

    def suggest(self, q_norm: str, num_suggestions: int) -> list[_Suggestion]:
        """Return the top semantic matches for a normalised query."""
        if len(q_norm) < self._min_chars:
            return []

        q_vec = self._index.vectoriser.transform([q_norm])
        q_vec = np.asarray(q_vec, dtype=float)
        if q_vec.size == 0:
            return []
        q_vec = q_vec.reshape(-1)

        sims = self._index.doc_matrix @ q_vec
        if sims.size == 0:
            return []

        k = min(num_suggestions, int(sims.size))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out: list[_Suggestion] = []
        for i in top_idx:
            sim = float(sims[int(i)])
            if sim <= 0:
                continue
            row_id = str(self._index.doc_ids[int(i)])
            out.append(
                _Suggestion(
                    display_text=str(self._index.doc_display[int(i)]),
                    score=sim,
                    search_text=str(self._index.doc_search[int(i)]),
                    row_id=row_id,
                )
            )
        return out
