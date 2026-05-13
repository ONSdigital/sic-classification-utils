"""This module provides utilities for embedding and searching index data (SIC or SOC)
using Classifai vector store.

It includes functionality to load existing vector store, create embeddings from
flat csv file, manage vector stores, and perform similarity searches.
"""

# pylint: disable=too-many-instance-attributes

import logging
import os
from typing import Any

import numpy as np
from autocorrect import Speller
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import (
    HuggingFaceVectoriser,
)

from industrial_classification_utils.utils.constants import get_default_config
from industrial_classification_utils.utils.gcs_file_access import (
    DownloadedVectorStore,
    download_vector_store_from_gcs,
    is_gcs_path,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_default_config()


class ChromaDBesqueHFVectoriser(HuggingFaceVectoriser):
    """Custom HuggingFaceVectoriser that normalizes vectors to unit length after embedding."""

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

        vectors = super().transform(texts)
        vectors = self._normalize(vectors)

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


class EmbeddingHandler:
    """Handles embedding operations for the Classifai vector store.

    Attributes:
        embedding_model_name (str): Name of the HuggingFace sentence-transformer model.
        db_dir (str): Directory where the Classifai vector store database is located.
            When loaded from GCS this is updated to the local temp directory path.
        index_source_file (str | None): Path or URI that the current vector store was
            built from.  Set to ``db_dir`` when an existing store is loaded.
        k_matches (int): Number of nearest neighbours returned per search query.
        spell (Speller): Autocorrect spell-checker used in :meth:`search_index_multi`.
        embeddings (Any): The underlying vectoriser instance.
        vector_store (VectorStore): The loaded or built Classifai vector store.
        index_size (int): Number of entries in the vector store.
    """

    def __init__(
        self,
        embedding_model_name: str = config["embedding"]["embedding_model_name"],
        db_dir: str = config["embedding"]["db_dir"],
        k_matches: int = config["embedding"]["k_matches"],
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
        self.embedding_model_name = embedding_model_name
        self.k_matches = k_matches
        self.db_dir = db_dir
        self.index_source_file = index_source_file

        self.embeddings: Any = ChromaDBesqueHFVectoriser(
            model_name=f"sentence-transformers/{embedding_model_name}"
        )
        logger.info("Using embedding model: %s", embedding_model_name)

        self.spell = Speller()

        self._downloaded_vector_store: DownloadedVectorStore | None = None
        self.vector_store: VectorStore
        if not self.index_source_file:
            self.vector_store = self._load_existing_vector_store()
            # Update index_source_file to reflect the data source and db_dir to reflect
            # the actual location of the vector store (local or temp dir if downloaded from GCS).
            self.index_source_file = self.db_dir
            self.db_dir = (
                self.db_dir
                if self._downloaded_vector_store is None
                else self._downloaded_vector_store.temp_dir.name
            )
        else:
            self.vector_store = self._build_vector_store()

        self.index_size = self.vector_store.num_vectors

        logger.info(
            "Vector store created in: %s containing %s entries from: %s.",
            self.db_dir,
            self.index_size,
            self.index_source_file,
        )

        logger.debug(
            "EmbeddingHandler initialised with config: %s", self.get_embed_config()
        )

    def _load_existing_vector_store(self) -> VectorStore:
        """Load an existing vector store from a local folder or a GCS URI.

        Returns:
            A :class:`VectorStore` loaded from ``db_dir``.

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

        return VectorStore.from_filespace(
            folder_path=db_dir,
            vectoriser=self.embeddings,
            hooks=None,
        )

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

        logger.info(
            "Building vector store in %s from source file %s.",
            self.db_dir,
            self.index_source_file,
        )

        if os.path.exists(os.path.join(self.db_dir, "vectors.parquet")):
            logger.warning(
                "Existing vector store files found in %s. They will be overwritten.",
                self.db_dir,
            )

        vector_store = VectorStore(
            file_name=str(self.index_source_file),
            data_type="csv",
            vectoriser=self.embeddings,
            batch_size=8,
            meta_data=None,
            output_dir=self.db_dir,
            overwrite=True,
            hooks=None,
        )

        return vector_store

    def search_index(
        self, query: str, return_dicts: bool = True
    ) -> list[dict] | list[tuple[str, float]]:
        """Returns k index entries with the highest relevance to the query.

        Args:
            query (str): Query string for which the most relevant index entries
                will be returned.
            return_dicts (bool, optional): If True, returns data as a list of
                dictionaries. Otherwise, returns simple tuples. Defaults to True.

        Returns:
            Union[list[dict], list[tuple[str, float]]]: List of top k index entries
            by relevance.
        """
        search_input = VectorStoreSearchInput({"id": ["q1"], "query": [query]})
        results = self.vector_store.search(search_input, n_results=self.k_matches)

        # ClassifAI returns a dataframe-like object.
        # Depending on the exact backend/version, one of these usually works.
        if hasattr(results, "to_dicts"):  # noqa: SIM108
            rows = results.to_dicts()
        else:
            rows = results.to_dict(orient="records")

        if return_dicts:
            return [
                {
                    "distance": float(1.0 - row["score"]),
                    "title": row["doc_text"],
                    "code": row["doc_label"],
                }
                for row in rows
            ]

        return [(row["doc_label"], float(row["score"])) for row in rows]

    def search_index_multi(self, query: list[str]) -> list[dict]:
        """Returns k document chunks with the highest relevance to a list of query fields.

        Args:
            query (list[str]): List of query fields (in priority order) for which
                the most relevant index entries will be returned.
                Example: [industry_descr, job_title, job_descr].

        Returns:
            list[dict]: List of top k index entries by relevance.
        """
        query = [x for x in query if x is not None]
        search_terms_list: set[str] = set()
        for i in range(1, len(query) + 1):
            term = " ".join(query[:i])
            search_terms_list.add(term)
            search_terms_list.add(self.spell(term))
        short_list = [
            hit for term in search_terms_list for hit in self.search_index(query=term)
        ]
        return sorted(short_list, key=lambda x: x["distance"])  # type: ignore

    def get_embed_config(self) -> dict[str, Any]:
        """Return the current embedding configuration.

        Returns:
            Dictionary with keys compatible with vector-store-api:
            ``embedding_model_name``, ``db_dir``,
            ``matches``, ``sic_condensed``, ``index_size``.
        """
        embed_config = {
            "embedding_model_name": self.embedding_model_name,
            "db_dir": self.db_dir,
            "matches": self.k_matches,
            "sic_condensed": self.index_source_file,
            "index_size": self.index_size,
        }
        return embed_config
