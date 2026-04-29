"""This module provides utilities for embedding and searching industrial classification data
using Classifai vector stores and language models.

It includes functionality for embedding SIC hierarchy data, managing vector stores,
and performing similarity searches.
"""

import csv

# Optional but doesn't hurt
import logging
import os
import tempfile
import uuid
from typing import Any, Optional, Union

import numpy as np
from autocorrect import Speller
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import (
    HuggingFaceVectoriser,
)
from industrial_classification.hierarchy.sic_hierarchy import load_hierarchy
from langchain_google_vertexai import VertexAIEmbeddings

from industrial_classification_utils.utils.constants import get_default_config
from industrial_classification_utils.utils.gcs_file_access import (
    DownloadedVectorStore,
    download_vector_store_from_gcs,
    is_gcs_path,
)
from industrial_classification_utils.utils.sic_data_access import (
    load_sic_index,
    load_sic_structure,
)


class ChromaDBesqueHFVectoriser(HuggingFaceVectoriser):
    """Custom HuggingFaceVectoriser that normalizes vectors to unit length after embedding."""

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def transform(self, texts):
        """Transforms texts into normalized vectors."""
        single_input = isinstance(texts, str)
        if single_input:
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
        """Asynchronously embeds a list of documents into normalized vectors."""
        return self.embed_documents(texts)

    def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embeds a single query into a normalized vector."""
        return self.embed_query(text)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_default_config()
MAX_BATCH_SIZE = 5400

embedding_config = {
    "embedding_model_name": config["embedding"]["embedding_model_name"],
    "db_dir": config["embedding"]["db_dir"],
    "sic_index": config["lookups"]["sic_index"],
    "sic_structure": config["lookups"]["sic_structure"],
    "sic_condensed": config["lookups"]["sic_condensed"],
    "matches": config["embedding"]["k_matches"],
    "index_size": None,
}


class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    """Custom VertexAIEmbeddings to specify task type for embeddings."""

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 0,
        *,
        embeddings_task_type="SEMANTIC_SIMILARITY",
    ) -> list[list[float]]:
        """Embeds a list of documents using the specified task type."""
        return super().embed_documents(
            texts,
            batch_size=batch_size,
            embeddings_task_type=embeddings_task_type,
        )

    def embed_query(
        self,
        text: str,
        *,
        embeddings_task_type="SEMANTIC_SIMILARITY",
    ) -> list[float]:
        """Embeds a single query using the specified task type."""
        return super().embed_query(text, embeddings_task_type=embeddings_task_type)


class EmbeddingHandler:  # pylint: disable=too-many-instance-attributes
    """Handles embedding operations for the Classifai vector store.

    Attributes:
        embeddings (Any): The embedding model used for vectorization.
        db_dir (str): Directory where the (classifai) vector store database is located.
        vector_store (VectorStore): The Classifai vector store instance.
        k_matches (int): Number of nearest matches to retrieve during search.
        spell (Speller): Autocorrect spell checker instance.
        _index_size (int): Number of entries in the vector store.
    """

    def __init__(  # noqa: PLR0913 # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        embedding_model_name: str = config["embedding"]["embedding_model_name"],
        db_dir: str = config["embedding"]["db_dir"],
        knowledgebase_csv: Optional[str] = None,
        k_matches: int = config["embedding"]["k_matches"],
        sic_index_file=None,
        sic_structure_file=None,
    ):
        """Initializes the EmbeddingHandler.

        Args:
            embedding_model_name (str, optional): Name of the embedding model to use.
                Defaults to the value in the configuration file.
            db_dir (str | None, optional): Directory for the vector store database.
                Defaults to the value in the configuration file.
            knowledgebase_csv (str, optional): Path to a CSV file containing the knowledge base.
            k_matches (int, optional): Number of nearest matches to retrieve.
                Defaults to 20.
            sic_index_file (optional): Optional override for the SIC index source.
            sic_structure_file (optional): Optional override for the SIC structure source.
        """
        self.embeddings: Any
        if embedding_model_name.startswith(("textembedding-", "text-embedding-")):
            self.embeddings = CustomVertexAIEmbeddings(model=embedding_model_name)
        else:
            self.embeddings = ChromaDBesqueHFVectoriser(
                model_name=f"sentence-transformers/{embedding_model_name}"
            )

        logger.info("Using embedding model: %s", embedding_model_name)

        self.db_dir = db_dir
        self.knowledgebase_csv = knowledgebase_csv
        self.k_matches = k_matches
        self.spell = Speller()

        self.sic_index_file = sic_index_file or config["lookups"]["sic_index"]
        self.sic_structure_file = (
            sic_structure_file or config["lookups"]["sic_structure"]
        )

        self._downloaded_vector_store: DownloadedVectorStore | None = None
        self.vector_store = self._load_or_build_vector_store()
        self._index_size = self.vector_store.num_vectors

        logger.info(
            "Vector store created in: %s containing %s entries.",
            self.db_dir,
            self._index_size,
        )

        embedding_config["embedding_model_name"] = embedding_model_name
        embedding_config["db_dir"] = db_dir
        embedding_config["matches"] = self.k_matches
        embedding_config["index_size"] = self._index_size
        embedding_config["sic_index"] = self.sic_index_file
        embedding_config["sic_structure"] = self.sic_structure_file

        logger.debug("EmbeddingHandler initialised with config: %s", embedding_config)

    def _load_existing_vector_store(self) -> Optional[VectorStore]:
        """Load an existing vector store from either a local folder or a GCS folder.
        Returns None if no existing store is found locally.
        Raises FileNotFoundError for missing required files in GCS.
        """
        if is_gcs_path(self.db_dir):
            self._downloaded_vector_store = download_vector_store_from_gcs(self.db_dir)
            logger.info(
                "Loading existing ClassifAI vector store from GCS URI %s", self.db_dir
            )
            return VectorStore.from_filespace(
                folder_path=self._downloaded_vector_store.path,
                vectoriser=self.embeddings,
                hooks=None,
            )

        metadata_path = os.path.join(self.db_dir, "metadata.json")
        vectors_path = os.path.join(self.db_dir, "vectors.parquet")

        has_existing_store = (
            os.path.isdir(self.db_dir)
            and os.path.exists(metadata_path)
            and os.path.exists(vectors_path)
        )

        if has_existing_store:
            logger.info("Loading existing ClassifAI vector store from %s", self.db_dir)
            return VectorStore.from_filespace(
                folder_path=self.db_dir,
                vectoriser=self.embeddings,
                hooks=None,
            )

        return None

    def _load_or_build_vector_store(  # pylint: disable=too-many-locals
        self,
    ) -> VectorStore:
        """Load an existing ClassifAI vector store, or build one from SIC source files."""
        if not self.db_dir:
            raise ValueError("db_dir must be provided.")

        existing_store = self._load_existing_vector_store()
        if existing_store is not None:
            return existing_store

        logger.info(
            "No existing vector store found in %s. Building from SIC source files.",
            self.db_dir,
        )

        sic_index_file = self.sic_index_file
        sic_structure_file = self.sic_structure_file

        logger.info("Loading SIC index file: %s", sic_index_file)
        logger.info("Loading SIC structure file: %s", sic_structure_file)

        sic_index_df = load_sic_index(sic_index_file)
        sic_df = load_sic_structure(sic_structure_file)
        sic = load_hierarchy(sic_df, sic_index_df)

        rows: list[dict[str, str]] = []
        for _, row in sic.all_leaf_text().iterrows():
            code = (row["code"].replace(".", "").replace("/", "") + "0")[:5]
            rows.append(
                {
                    "label": str(uuid.uuid3(uuid.NAMESPACE_URL, row["text"])),
                    "text": row["text"],
                    "code": code,
                    "four_digit_code": code[0:4],
                    "two_digit_code": code[0:2],
                }
            )

        if not rows:
            raise ValueError("No SIC rows were generated for vector store build.")

        meta_data = {
            "code": str,
            "four_digit_code": str,
            "two_digit_code": str,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "sic_vectors.csv")

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "label",
                        "text",
                        "code",
                        "four_digit_code",
                        "two_digit_code",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            logger.info(
                "Building new ClassifAI vector store in %s from temporary CSV %s",
                self.db_dir,
                csv_path,
            )

            vector_store = VectorStore(
                file_name=csv_path,
                data_type="csv",
                vectoriser=self.embeddings,
                batch_size=8,
                meta_data=meta_data,
                output_dir=self.db_dir,
                overwrite=False,
                hooks=None,
            )

        return vector_store

    def search_index(
        self, query: str, return_dicts: bool = True
    ) -> Union[list[dict], list[tuple[str, float]]]:
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
            rows = results.to_dicts()  # type: ignore
        else:
            rows = results.to_dict(orient="records")

        if return_dicts:
            return [
                {
                    "distance": float(1.0 - row["score"]),
                    "title": row["doc_text"],
                    "code": row.get("code"),
                    "four_digit_code": row.get("four_digit_code"),
                    "two_digit_code": row.get("two_digit_code"),
                }
                for row in rows
            ]

        return [(row["doc_text"], float(row["score"])) for row in rows]

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
        search_terms_list = set()
        for i in range(len(query)):
            x = " ".join(query[: (i + 1)])
            search_terms_list.add(x)
            search_terms_list.add(self.spell(x))
        short_list = [y for x in search_terms_list for y in self.search_index(query=x)]
        return sorted(short_list, key=lambda x: x["distance"])  # type: ignore

    def get_embed_config(self) -> dict:
        """Returns the current embedding configuration as a dictionary."""
        return {
            "embedding_model_name": str(embedding_config["embedding_model_name"]),
            "db_dir": str(embedding_config["db_dir"]),
            "sic_index": str(embedding_config["sic_index"]),
            "sic_structure": str(embedding_config["sic_structure"]),
            "sic_condensed": str(embedding_config["sic_condensed"]),
            "matches": embedding_config["matches"],
            "index_size": embedding_config["index_size"],
        }
