"""This module provides utilities for embedding and searching index data (SIC or SOC)
using Classifai vector store.

It includes functionality to load existing vector store, create embeddings from
flat csv file, manage vector stores, and perform similarity searches.
"""

# pylint: disable=too-many-instance-attributes

from importlib import metadata
import json
import logging
import os
from typing import Any

from industrial_classification import meta
import numpy as np
from autocorrect import Speller
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import (
    HuggingFaceVectoriser,
    VectoriserBase
)

from industrial_classification_utils.models.config_model import EmbeddingStatus
from industrial_classification_utils.models.response_model import (
    SearchIndexItem,
    SearchIndexResponse,
)
from industrial_classification_utils.utils.constants import get_default_config
from industrial_classification_utils.utils.gcs_file_access import (
    DownloadedVectorStore,
    download_one_file_from_gcs,
    download_vector_store_from_gcs,
    is_gcs_path,
)

from time import perf_counter

SENTENCE_TRANSFORMERS_BACKEND = "sentence-transformers"
LIGHT_EMBED_ONNX_BACKEND = "light-embed-onnx"
DEFAULT_ONNX_MODEL_NAME = "onnx-models/all-MiniLM-L6-v2-onnx"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_default_config()


def _duration_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _docs_per_second(batch_size: int, duration_ms: float) -> float:
    return batch_size / (duration_ms / 1000) if duration_ms > 0 else 0.0


def _resolve_sentence_transformer_model_name(model_name: str) -> str:
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def _resolve_onnx_model_name(model_name: str) -> str:
    if model_name in {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}:
        return DEFAULT_ONNX_MODEL_NAME
    return model_name


class ChromaDBesqueHFVectoriser(HuggingFaceVectoriser):
    """Custom HuggingFaceVectoriser that normalizes vectors to unit length after embedding."""

    def __init__(self, model_name: str):
        started = perf_counter()
        super().__init__(model_name=model_name)
        logger.info(
            "embedding_model_loaded backend=%s model=%s duration_ms=%.2f",
            SENTENCE_TRANSFORMERS_BACKEND,
            model_name,
            _duration_ms(started),
        )
    

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes row vectors to unit length.

        Zero-norm vectors are left unchanged to avoid division by zero.

        Args:
            vectors: 2-D array of shape (n, d).

        Returns:
            Array of the same shape with each row scaled to unit L2 norm.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def transform(self, texts: list[str] | str) -> np.ndarray:
        """Transforms texts into normalized unit-length vectors.

        Args:
            texts: A single string or a list of strings to embed.

        Returns:
            2-D array of shape (n, d) with L2-normalised row vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        started = perf_counter()
        vectors = super().transform(texts)
        vectors = self._normalize(vectors)

        duration_ms = _duration_ms(started)

        logger.info(
            "embedding_transform_complete backend=%s batch_size=%s duration_ms=%.2f docs_per_second=%.2f",
            SENTENCE_TRANSFORMERS_BACKEND,
            len(texts),
            duration_ms,
            _docs_per_second(len(texts), duration_ms),
        )

        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents into normalized vectors."""
        return self.transform(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query into a normalized vector."""
        return self.transform([text]).tolist()[0]

    def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """(A)synchronously embeds a list of documents into normalized vectors."""
        return self.embed_documents(texts)

    def aembed_query(self, text: str) -> list[float]:
        """(A)synchronously embeds a single query into a normalized vector."""
        return self.embed_query(text)


class LightEmbedONNXVectoriser(VectoriserBase):
    """POC ONNX vectoriser using light-embed."""

    def __init__(self, model_name: str):
        from light_embed import TextEmbedding

        self.model_name = model_name
        started = perf_counter()
        self.model = TextEmbedding(model_name_or_path=model_name)

        logger.info(
            "embedding_model_loaded backend=%s model=%s duration_ms=%.2f",
            LIGHT_EMBED_ONNX_BACKEND,
            model_name,
            _duration_ms(started),
        )

    def transform(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        started = perf_counter()
        vectors = np.asarray(list(self.model.encode(texts)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        vectors = _normalise(vectors)
        duration_ms = _duration_ms(started)

        logger.info(
            "embedding_transform_complete backend=%s batch_size=%s duration_ms=%.2f docs_per_second=%.2f",
            LIGHT_EMBED_ONNX_BACKEND,
            len(texts),
            duration_ms,
            _docs_per_second(len(texts), duration_ms),
        )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.transform(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.transform([text]).tolist()[0]

    def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _create_vectoriser(embedding_backend: str, embedding_model_name: str) -> Any:
    if embedding_backend == SENTENCE_TRANSFORMERS_BACKEND:
        return ChromaDBesqueHFVectoriser(
            model_name=_resolve_sentence_transformer_model_name(embedding_model_name)
        )

    if embedding_backend == LIGHT_EMBED_ONNX_BACKEND:
        return LightEmbedONNXVectoriser(
            model_name=_resolve_onnx_model_name(embedding_model_name)
        )

    raise ValueError(f"Unsupported embedding backend: {embedding_backend}")


class EmbeddingHandler:
    """Handles embedding operations for the Classifai vector store.

    Attributes:
        embedding_model_name (str): Name of the HuggingFace sentence-transformer model.
        db_dir (str): Directory where the Classifai vector store database is located.
        index_source_file (str | None): Path or URI to the source file CSV that the current
            vector store was built from.
        k_matches (int): Number of nearest neighbours returned per search query.
        spell (Speller): Autocorrect spell-checker used in :meth:`search_index_multi`.
        embeddings (Any): The underlying vectoriser instance.
        vector_store (VectorStore): The loaded or built Classifai vector store.
        index_size (int): Number of entries in the vector store.
    """

    def __init__(
        self,
        embedding_model_name: str = config["embedding"].embedding_model_name,
        embedding_backend: str | None = None,
        db_dir: str = config["embedding"].db_dir,
        k_matches: int = config["embedding"].k_matches,
        index_source_file: str | None = None,
    ):
        """Initializes the EmbeddingHandler.

        Args:
            embedding_model_name: Name of the embedding model to use.
                Defaults to the value in the configuration file.
            db_dir: Directory for the vector store database.
                Defaults to the value in the configuration file.
            k_matches: Number of nearest matches to retrieve.
                Defaults to 20.
            index_source_file: Optional csv source file to build new embedding index.
                When provided, the vector store (db_dir) will be overwritten.
        """
        self.embedding_model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", config["embedding"].embedding_model_name
        )
        self.embedding_backend = embedding_backend or os.getenv(
            "EMBEDDING_BACKEND", config["embedding"].embedding_backend
        )       
        self.k_matches = k_matches
        self.db_dir = db_dir
        self.index_source_file = index_source_file

        self.embeddings: Any = _create_vectoriser(
            embedding_backend=self.embedding_backend,
            embedding_model_name=self.embedding_model_name,
        )
        logger.info(
            "Using embedding model: %s and backend: %s",
            self.embedding_model_name, self.embedding_backend
        )

        self.spell = Speller()

        self._downloaded_vector_store: DownloadedVectorStore | None = None
        self.vector_store: VectorStore
        if not self.index_source_file:
            # Update index_source_file to reflect the data source of the loaded vector store.
            self.vector_store, self.index_source_file = (
                self._load_existing_vector_store()
            )
        else:
            self.vector_store = self._build_vector_store()

        self.index_size = (
            self.vector_store.num_vectors if self.vector_store.num_vectors else 0
        )

        logger.info(
            "EmbeddingHandler initialised with config: %s", self.get_embed_config()
        )

    def _load_existing_vector_store(self) -> tuple[VectorStore, str | None]:
        """Load an existing vector store from a local folder or a GCS URI.

        Returns:
            A tuple containing a :class:`VectorStore` loaded from ``db_dir`` and
                the optional index source file.

        Raises:
            FileNotFoundError: If ``db_dir`` does not contain the required
                ``metadata.json`` and ``vectors.parquet`` files.
        """
        logger.info("Loading existing ClassifAI vector store from %s", self.db_dir)
        db_dir = self.db_dir

        if is_gcs_path(db_dir):
            self._downloaded_vector_store = download_vector_store_from_gcs(db_dir)
            db_dir = self._downloaded_vector_store.temp_dir.name

        metadata_path = os.path.join(db_dir, "metadata.json")
        vectors_path = os.path.join(db_dir, "vectors.parquet")

        has_existing_store = (
            os.path.isdir(db_dir)
            and os.path.exists(metadata_path)
            and os.path.exists(vectors_path)
        )

        if not has_existing_store:
            raise FileNotFoundError(
                f"No existing vector store found in {self.db_dir}. "
                "Please ensure the directory contains metadata.json and vectors.parquet, "
                "or provide a valid index source file."
            )

        vs = VectorStore.from_filespace(
            folder_path=db_dir,
            vectoriser=self.embeddings,
            hooks=None,
        )

        logger.info("Existing vector store loaded successfully from %s", self.db_dir)
        # read the source+index_file from metadata
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        index_source_file = meta.get("index_source_file", None)

        store_backend = meta.get("embedding_backend")
        store_model = meta.get("embedding_model_name")

        if store_backend and store_backend != self.embedding_backend:
            logger.warning(
                "Vector store backend mismatch. metadata=%s runtime=%s",
                store_backend,
                self.embedding_backend,
            )

        if store_model and store_model != self.embedding_model_name:
            logger.warning(
                "Vector store model mismatch. metadata=%s runtime=%s",
                store_model,
                self.embedding_model_name,
            )
        return (vs, index_source_file)

    def _build_vector_store(self) -> VectorStore:
        """Build a Classifai vector store from a CSV source file.

        The store is always written to ``db_dir``, overwriting any existing files.

        Returns:
            A newly created :class:`VectorStore`.

        Raises:
            ValueError: If ``db_dir`` is not set.
        """
        if not self.db_dir:
            raise ValueError("db_dir must be provided.")

        index_source_file = str(self.index_source_file)
        logger.info(
            "Building vector store in %s from source file %s.",
            self.db_dir,
            index_source_file,
        )

        if os.path.exists(os.path.join(self.db_dir, "vectors.parquet")):
            logger.warning(
                "Existing vector store files found in %s. They will be overwritten.",
                self.db_dir,
            )

        if is_gcs_path(index_source_file):
            downloaded_file = download_one_file_from_gcs(index_source_file)
            index_source_file = downloaded_file.path

        started = perf_counter()

        vector_store = VectorStore(
            file_name=str(index_source_file),
            data_type="csv",
            vectoriser=self.embeddings,
            batch_size=8,
            meta_data=None,
            output_dir=self.db_dir,
            overwrite=True,
            hooks=None,
        )

        # add file name to metadata for traceability
        metadata_path = os.path.join(self.db_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["index_source_file"] = str(self.index_source_file)
        metadata["embedding_backend"] = self.embedding_backend
        metadata["embedding_model_name"] = self.embedding_model_name
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        logger.info(
            "vector_store_build_complete backend=%s model=%s index_size=%s duration_ms=%.2f",
            self.embedding_backend,
            self.embedding_model_name,
            vector_store.num_vectors,
            _duration_ms(started),
        )

        logger.info(
            "Vector store built successfully in %s with data from %s.",
            self.db_dir,
            self.index_source_file,
        )

        return vector_store

    def search_index(
        self,
        query: str,
    ) -> SearchIndexResponse:
        """Returns k index entries with the highest relevance to the query.

        Args:
            query (str): Query string for which the most relevant index entries
                will be returned.

        Returns:
            SearchIndexResponse: List of top k index entries by relevance.
        """
        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [query]})

        n_results = (
            self.k_matches
            if self.index_size is None
            else min(self.index_size, self.k_matches)
        )

        started = perf_counter()
        results = self.vector_store.search(search_input, n_results=n_results)
        duration_ms = _duration_ms(started)

        logger.info(
            "search_index_complete backend=%s model=%s n_results=%s duration_ms=%.2f query_length=%s",
            self.embedding_backend,
            self.embedding_model_name,
            n_results,
            duration_ms,
            len(query),
        )

        # ClassifAI returns a dataframe-like object.
        # Depending on the exact backend/version, one of these usually works.
        if hasattr(results, "to_dicts"):  # noqa: SIM108
            rows = results.to_dicts()  # type: ignore
        else:
            rows = results.to_dict(orient="records")

        return SearchIndexResponse(
            results=[
                SearchIndexItem(
                    distance=float(1.0 - row["score"]),
                    title=row["doc_text"],
                    code=row["doc_label"],
                )
                for row in rows
            ]
        )

    def search_index_multi(self, query: list[str]) -> SearchIndexResponse:
        """Returns k document chunks with the highest relevance to a list of query fields.

        Args:
            query (list[str]): List of query fields (in priority order) for which
                the most relevant index entries will be returned.
                Example: [industry_descr, job_title, job_descr].

        Returns:
            SearchIndexResponse: List of top k index entries by relevance.
        """
        query = [x for x in query if x is not None]
        search_terms_list: set[str] = set()
        for i in range(1, len(query) + 1):
            term = " ".join(query[:i])
            search_terms_list.add(term)
            search_terms_list.add(self.spell(term))
        short_list = [
            hit
            for term in search_terms_list
            for hit in self.search_index(query=term).results
        ]
        return SearchIndexResponse(results=sorted(short_list, key=lambda x: x.distance))

    def get_embed_config(self) -> EmbeddingStatus:
        """Return the current embedding configuration.

        Returns:
            EmbeddingStatus: The current embedding configuration.
        """
        embed_config = EmbeddingStatus(
            embedding_model_name=self.embedding_model_name,
            embedding_backend=self.embedding_backend,
            db_dir=self.db_dir,
            k_matches=self.k_matches,
            index_source_file=self.index_source_file,
            index_size=self.index_size,
            status="ready",
        )
        return embed_config
