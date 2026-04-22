"""Private retrieval strategies for SAYT suggesters."""

# pylint: disable=too-few-public-methods, R0801

import csv
import os
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import cast

import numpy as np
from classifai.indexers import VectorStore, VectorStoreSearchInput
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


@dataclass(frozen=True, slots=True)
class _DenseVectorIndex:
    """ClassifAI-backed dense search index for in-memory query-time retrieval."""

    vector_store: VectorStore
    num_vectors: int

    @classmethod
    def from_corpus(
        cls,
        *,
        corpus: CleanCorpus,
        vectoriser: VectoriserBase,
    ) -> "_DenseVectorIndex":
        """Build a dense index using ClassifAI's native VectorStore pipeline."""
        with tempfile.TemporaryDirectory(prefix="sayt_") as tmp_dir:
            csv_path = os.path.join(tmp_dir, "corpus.csv")

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["label", "text"])
                writer.writeheader()
                writer.writerows(
                    {"label": row_id, "text": search_text}
                    for row_id, search_text, _ in corpus.rows
                )

            vector_store = VectorStore(
                file_name=csv_path,
                data_type="csv",
                vectoriser=vectoriser,
                batch_size=64,
                output_dir=None,
                overwrite=True,
                hooks=None,
            )

        return cls(
            vector_store=vector_store,
            num_vectors=int(vector_store.num_vectors or 0),
        )

    def query(self, q_norm: str, num_suggestions: int) -> list[tuple[str, float]]:
        """Return the top dense-vector matches as row ids with scores."""
        if self.num_vectors < 1 or num_suggestions < 1:
            return []

        n_results = min(num_suggestions, self.num_vectors)
        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [q_norm]})
        results = self.vector_store.search(search_input, n_results=n_results)
        out = [
            (row["doc_label"], float(row["score"]))
            for row in results.to_dict(orient="records")
        ]
        return out


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


class _DenseRetriever:
    """Shared cosine-similarity retrieval over an in-memory dense index."""

    _corpus: CleanCorpus
    _min_chars: int
    _index: _DenseVectorIndex

    def suggest(self, q_norm: str, num_suggestions: int) -> list[_Suggestion]:
        """Return dense-vector matches after applying retriever-level gating."""
        if len(q_norm) < self._min_chars:
            return []
        return [
            _Suggestion(
                display_text=self._corpus.id_to_display.get(row_id, ""),
                score=score,
                search_text=self._corpus.id_to_search.get(row_id, ""),
                row_id=row_id,
            )
            for row_id, score in self._index.query(q_norm, num_suggestions)
        ]


class NgramRetriever(_DenseRetriever):
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
        self._corpus = corpus
        self._min_chars = min_chars
        ngram_vectoriser = _CharNgramVectoriser(
            [search for _, search, _ in corpus.rows],
            n=n,
            max_df=max_df,
        )
        self._index = _DenseVectorIndex.from_corpus(
            corpus=corpus,
            vectoriser=ngram_vectoriser,
        )


class SemanticRetriever(_DenseRetriever):
    """Retrieve suggestions using sentence-transformer embeddings."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        model: str,
        min_chars: int,
    ) -> None:
        """Build a dense semantic index for the cleaned corpus."""
        self._corpus = corpus
        self._min_chars = min_chars
        base_vectoriser: VectoriserBase = HuggingFaceVectoriser(
            f"sentence-transformers/{model}"
        )
        semantic_vectoriser: VectoriserBase = _L2NormalisingVectoriser(base_vectoriser)
        self._index = _DenseVectorIndex.from_corpus(
            corpus=corpus,
            vectoriser=semantic_vectoriser,
        )
