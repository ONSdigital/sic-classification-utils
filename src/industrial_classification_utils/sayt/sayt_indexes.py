# pylint: disable=too-few-public-methods

"""Dense index construction helpers for SAYT retrievers."""

import csv
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import classifai.indexers.main as classifai_indexers_main
import numpy as np
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import HuggingFaceVectoriser, VectoriserBase
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from .sayt_core import CleanCorpus, take_with_ties


def _silent_tqdm(iterable, **_kwargs):
    """Pass through iterables unchanged to suppress ClassifAI progress bars."""
    return iterable


@contextmanager
def _silence_classifai_tqdm():
    """Temporarily replace ClassifAI's tqdm import with a no-op wrapper."""
    previous_tqdm = classifai_indexers_main.tqdm
    classifai_indexers_main.tqdm = _silent_tqdm
    try:
        yield
    finally:
        classifai_indexers_main.tqdm = previous_tqdm


@dataclass(frozen=True, slots=True)
class DenseVectorIndex:
    """Wrap a ClassifAI vector store for query-time dense retrieval."""

    _vector_store: VectorStore
    _num_vectors: int
    _corpus: CleanCorpus

    @classmethod
    def from_corpus(
        cls,
        *,
        corpus: CleanCorpus,
        vectoriser: VectoriserBase,
    ) -> "DenseVectorIndex":
        """Build a dense index from a cleaned corpus.

        Args:
            corpus: Cleaned corpus whose normalised search text should be
                indexed.
            vectoriser: Vectoriser used to embed corpus rows and future queries.

        Returns:
            A ``DenseVectorIndex`` backed by ClassifAI's ``VectorStore``.
        """
        with tempfile.TemporaryDirectory(prefix="sayt_") as tmp_dir:
            csv_path = os.path.join(tmp_dir, "corpus.csv")

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["label", "text"])
                writer.writeheader()
                writer.writerows(
                    {"label": row_id, "text": search_text}
                    for row_id, search_text, _ in corpus.rows
                )

            with _silence_classifai_tqdm():
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
            _vector_store=vector_store,
            _num_vectors=int(vector_store.num_vectors or 0),
            _corpus=corpus,
        )

    def query(self, q_norm: str, num_suggestions: int) -> list[tuple[str, float]]:
        """Query the dense index with a normalised string.

        Args:
            q_norm: Normalised query text.
            num_suggestions: Maximum number of scored row ids to return before
                tie expansion.

        Returns:
            Ranked ``(row_id, score)`` pairs from the dense vector store.
        """
        if self._num_vectors < 1 or num_suggestions < 1:
            return []

        n_results = min(self._num_vectors, num_suggestions * 2)
        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [q_norm]})
        with _silence_classifai_tqdm():
            results = self._vector_store.search(search_input, n_results=n_results)
        out = [
            (row["doc_label"], float(row["score"]))
            for row in results.to_dict(orient="records")
        ]
        return take_with_ties(out, limit=num_suggestions)


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


def build_ngram_index(
    corpus: CleanCorpus,
    *,
    n: int,
    max_df: float,
) -> DenseVectorIndex:
    """Build a dense index backed by character n-gram vectors.

    Args:
        corpus: Cleaned corpus to index.
        n: Character n-gram size.
        max_df: Maximum document frequency passed to ``CountVectorizer``.

    Returns:
        A dense index using character n-gram embeddings.
    """
    return DenseVectorIndex.from_corpus(
        corpus=corpus,
        vectoriser=_CharNgramVectoriser(
            [search for _, search, _ in corpus.rows],
            n=n,
            max_df=max_df,
        ),
    )


def build_semantic_index(
    corpus: CleanCorpus,
    *,
    model: str,
) -> DenseVectorIndex:
    """Build a dense index backed by sentence-transformer embeddings.

    Args:
        corpus: Cleaned corpus to index.
        model: Sentence-transformer model name without the repository prefix.

    Returns:
        A dense index using semantic embeddings.
    """
    base_vectoriser: VectoriserBase = HuggingFaceVectoriser(
        f"sentence-transformers/{model}"
    )
    semantic_vectoriser: VectoriserBase = _L2NormalisingVectoriser(base_vectoriser)
    return DenseVectorIndex.from_corpus(
        corpus=corpus,
        vectoriser=semantic_vectoriser,
    )
