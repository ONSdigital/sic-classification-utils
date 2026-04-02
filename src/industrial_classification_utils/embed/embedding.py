"""This module provides utilities for embedding and searching industrial classification data
using Chroma vector stores and language models.

It includes functionality for embedding SIC hierarchy data, managing vector stores,
and performing similarity searches.
"""

import csv

# Optional but doesn't hurt
import logging
import os
import posixpath
import shutil
import sqlite3  # noqa: F401 # pylint: disable=unused-import
import tempfile

# Docker Image may have old sqlite3 version for ChromaDB
# Top of your module (before any langchain or chroma import)
import uuid
from typing import Any, Optional, Union

import numpy as np
from autocorrect import Speller
from classifai.indexers import VectorStore, VectorStoreSearchInput
from classifai.vectorisers import (
    HuggingFaceVectoriser,
)
from google.cloud import storage
from industrial_classification.hierarchy.sic_hierarchy import SIC, load_hierarchy
from langchain_google_vertexai import VertexAIEmbeddings

from industrial_classification_utils.models.config_model import (
    FullConfig,
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

        if single_input:
            return vectors[0:1]
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

# Share configuration with other modules
embedding_config = {
    "embedding_model_name": "unknown",
    "llm_model_name": "unknown",
    "db_dir": "unknown",
    "sic_index": "unknown",
    "sic_structure": "unknown",
    "sic_condensed": "unknown",
    "matches": 0,
    "index_size": 0,
}

EXPECTED_SPLIT_PARTS = 2


def get_config() -> FullConfig:
    """Returns the configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "llm": {
            "llm_model_name": "gemini-1.0-pro",
            "embedding_model_name": "all-MiniLM-L6-v2",  # text-embedding-004
            "db_dir": "src/industrial_classification_utils/data/vector_store",
        },
        "lookups": {
            "sic_index": (
                "industrial_classification_utils.data.sic_index",
                "uksic2007indexeswithaddendumdecember2022.xlsx",
            ),
            "sic_structure": (
                "industrial_classification_utils.data.sic_index",
                "publisheduksicsummaryofstructureworksheet.xlsx",
            ),
            "sic_condensed": (
                "industrial_classification_utils.data.example",
                "sic_2d_condensed.txt",
            ),
        },
    }


config = get_config()
MAX_BATCH_SIZE = 5400


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
    """Handles embedding operations for the Chroma vector store.

    Attributes:
        embeddings (Any): The embedding model used for vectorization.
        db_dir (str): Directory where the (classifai) vector store database is located.
        vector_store (Chroma): The Chroma vector store instance.
        k_matches (int): Number of nearest matches to retrieve during search.
        spell (Speller): Autocorrect spell checker instance.
        _index_size (int): Number of entries in the vector store.
    """

    def __init__(  # noqa: PLR0913 # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        embedding_model_name: str = config["llm"]["embedding_model_name"],
        db_dir: str = config["llm"]["db_dir"],
        knowledgebase_csv: Optional[str] = None,
        meta_data: Optional[dict[str, type]] = None,
        k_matches: int = 20,
        sic_index_file=None,
        sic_structure_file=None,
    ):
        """Initializes the EmbeddingHandler.

        Args:
            embedding_model_name (str, optional): Name of the embedding model to use.
                Defaults to the value in the configuration file.
            db_dir (str, optional): Directory for the vector store database.
                Defaults to the value in the configuration file.
            knowledgebase_csv (str, optional): Path to a CSV file containing the knowledge base.
            meta_data (dict[str, type], optional): Metadata schema for the vector store.
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
        self.meta_data = meta_data or {}
        self.k_matches = k_matches
        self.spell = Speller()

        self.sic_index_file = sic_index_file or config["lookups"]["sic_index"]
        self.sic_structure_file = (
            sic_structure_file or config["lookups"]["sic_structure"]
        )

        self._vector_store_tmp_dir: tempfile.TemporaryDirectory | None = None
        self.vector_store = self._load_or_build_vector_store()
        self._index_size = self.vector_store.num_vectors

        logger.info(
            "Vector store created in: %s containing %s entries.",
            self.db_dir,
            self._index_size,
        )

        embedding_config["embedding_model_name"] = embedding_model_name
        embedding_config["llm_model_name"] = config["llm"].get(
            "llm_model_name", "unknown"
        )
        embedding_config["db_dir"] = db_dir
        embedding_config["matches"] = self.k_matches
        embedding_config["index_size"] = self._index_size
        embedding_config["sic_index"] = self.sic_index_file
        embedding_config["sic_structure"] = self.sic_structure_file

        logger.debug("EmbeddingHandler initialised with config: %s", embedding_config)

    def _is_gcs_path(self, path: str) -> bool:
        return path.startswith("gs://")

    def _parse_gcs_uri(self, gcs_uri: str) -> tuple[str, str]:
        """Parse gs://bucket/path/to/folder into (bucket_name, prefix)."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Not a valid GCS URI: {gcs_uri}")

        without_scheme = gcs_uri[5:]
        parts = without_scheme.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

        if not bucket_name:
            raise ValueError(f"Invalid GCS URI, bucket missing: {gcs_uri}")

        return bucket_name, prefix

    def _download_vector_store_from_gcs(self, gcs_uri: str) -> str:
        """Download metadata.json and vectors.parquet from a GCS folder into a
        temporary local directory and return that directory path.

        Raises:
            FileNotFoundError: if either metadata.json or vectors.parquet is missing.
        """
        bucket_name, prefix = self._parse_gcs_uri(gcs_uri)

        metadata_blob_name = (
            posixpath.join(prefix, "metadata.json") if prefix else "metadata.json"
        )
        vectors_blob_name = (
            posixpath.join(prefix, "vectors.parquet") if prefix else "vectors.parquet"
        )

        logger.info(
            "Attempting to load ClassifAI vector store from GCS bucket=%s prefix=%s",
            bucket_name,
            prefix,
        )

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        metadata_blob = bucket.blob(metadata_blob_name)
        vectors_blob = bucket.blob(vectors_blob_name)

        missing = []
        if not metadata_blob.exists():
            missing.append(f"gs://{bucket_name}/{metadata_blob_name}")
        if not vectors_blob.exists():
            missing.append(f"gs://{bucket_name}/{vectors_blob_name}")

        if missing:
            raise FileNotFoundError(
                "Required vector store file(s) not found in GCS: " + ", ".join(missing)
            )

        tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        local_dir = tmp_dir.name

        # Keep the tempdir alive for the lifetime of the handler
        self._vector_store_tmp_dir = tmp_dir

        local_metadata = os.path.join(local_dir, "metadata.json")
        local_vectors = os.path.join(local_dir, "vectors.parquet")

        metadata_blob.download_to_filename(local_metadata)
        vectors_blob.download_to_filename(local_vectors)

        logger.info(
            "Downloaded vector store files from gs://%s/%s to %s",
            bucket_name,
            prefix,
            local_dir,
        )

        return local_dir

    def _load_existing_vector_store(self) -> Optional[VectorStore]:
        """Load an existing vector store from either a local folder or a GCS folder.
        Returns None if no existing store is found locally.
        Raises FileNotFoundError for missing required files in GCS.
        """
        if self._is_gcs_path(self.db_dir):
            local_dir = self._download_vector_store_from_gcs(self.db_dir)
            logger.info(
                "Loading existing ClassifAI vector store from GCS URI %s", self.db_dir
            )
            return VectorStore.from_filespace(
                folder_path=local_dir,
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
                    "id": str(uuid.uuid3(uuid.NAMESPACE_URL, row["text"])),
                    "text": row["text"],
                    "code": code,
                    "four_digit_code": code[0:4],
                    "two_digit_code": code[0:2],
                }
            )

        if not rows:
            raise ValueError("No SIC rows were generated for vector store build.")

        meta_data = self.meta_data or {
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
                        "id",
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

    def embed_index(  # noqa: PLR0915, C901 # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments, too-many-statements
        self,
        from_empty: bool = True,
        sic: Optional[SIC] = None,
        file_object=None,
        sic_index_file=None,
        sic_structure_file=None,
    ):
        """Embeds the index entries into the vector store.

        For ClassifAI, this rebuilds the vector store from source data by first
        generating a CSV knowledgebase and then creating a new VectorStore from it.

        Args:
            from_empty (bool, optional): Whether to rebuild the vector store from scratch.
                Defaults to True.
            sic (Optional[SIC], optional): The SIC hierarchy object. If None, the hierarchy
                is loaded from files specified in the config. Defaults to None.
            file_object (StringIO object, optional): The index file as a StringIO object.
                If provided, the file will be read line by line.
                Each line should have the format **code**: **description**.
            sic_index_file (optional): Custom path or file-like object to override
                default SIC index source.
            sic_structure_file (optional): Custom path or file-like object to override
                default SIC structure source.
        """
        logger.info(
            "Embedding index: from_empty=%s, sic=%s, file_object=%s, "
            "sic_index_file=%s, sic_structure_file=%s",
            from_empty,
            sic,
            file_object,
            sic_index_file,
            sic_structure_file,
        )

        rows: list[dict[str, str]] = []

        if file_object is not None:
            for line in file_object:
                if not line:
                    continue

                bits = line.split(":", 1)
                if len(bits) != EXPECTED_SPLIT_PARTS:
                    logger.warning("Skipping malformed line: %s", line)
                    continue

                code = bits[0].strip()
                text = bits[1].strip()

                rows.append(
                    {
                        "id": str(uuid.uuid3(uuid.NAMESPACE_URL, line.strip())),
                        "text": text,
                        "code": code,
                        "four_digit_code": code[0:4],
                        "two_digit_code": code[0:2],
                    }
                )

        else:
            if sic is None:
                logger.info(
                    "Loading SIC hierarchy from files: %s, %s",
                    sic_index_file,
                    sic_structure_file,
                )

                if sic_index_file is None:
                    sic_index_file = config["lookups"]["sic_index"]
                sic_index_df = load_sic_index(sic_index_file)

                if sic_structure_file is None:
                    sic_structure_file = config["lookups"]["sic_structure"]
                sic_df = load_sic_structure(sic_structure_file)

                sic = load_hierarchy(sic_df, sic_index_df)

            for _, row in sic.all_leaf_text().iterrows():
                code = (row["code"].replace(".", "").replace("/", "") + "0")[:5]

                rows.append(
                    {
                        "id": str(uuid.uuid3(uuid.NAMESPACE_URL, row["text"])),
                        "text": row["text"],
                        "code": code,
                        "four_digit_code": code[0:4],
                        "two_digit_code": code[0:2],
                    }
                )

        if not rows:
            logger.warning("No rows were generated for embedding.")
            self._index_size = 0
            return

        if from_empty and os.path.isdir(self.db_dir):
            logger.info("Removing existing vector store directory: %s", self.db_dir)
            shutil.rmtree(self.db_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "sic_vectors.csv")

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "id",
                        "text",
                        "code",
                        "four_digit_code",
                        "two_digit_code",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            self.knowledgebase_csv = csv_path
            self.meta_data = {
                "code": str,
                "four_digit_code": str,
                "two_digit_code": str,
            }

            self.vector_store = VectorStore(
                file_name=csv_path,
                data_type="csv",
                vectoriser=self.embeddings,
                batch_size=8,
                meta_data=self.meta_data,
                output_dir=self.db_dir,
                overwrite=from_empty,
                hooks=None,
            )

        self._index_size = self.vector_store.num_vectors

        logger.debug(
            "Inserted %s entries into vector embedding database.", f"{len(rows):,}"
        )

        embedding_config["index_size"] = self._index_size
        embedding_config["sic_index"] = sic_index_file or config["lookups"]["sic_index"]
        embedding_config["sic_structure"] = (
            sic_structure_file or config["lookups"]["sic_structure"]
        )
        embedding_config["sic_condensed"] = config["lookups"]["sic_condensed"]
        embedding_config["matches"] = self.k_matches
        embedding_config["db_dir"] = self.db_dir
        embedding_config["embedding_model_name"] = getattr(
            self.embeddings, "model_name", type(self.embeddings).__name__
        )
        embedding_config["llm_model_name"] = config["llm"].get(
            "llm_model_name", "unknown"
        )

        logger.info("Embedding config updated: %s", embedding_config)

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
            rows = results.to_dicts()
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
            "llm_model_name": str(embedding_config["llm_model_name"]),
            "db_dir": str(embedding_config["db_dir"]),
            "sic_index": str(embedding_config["sic_index"]),
            "sic_structure": str(embedding_config["sic_structure"]),
            "sic_condensed": str(embedding_config["sic_condensed"]),
            "matches": embedding_config["matches"],
            "index_size": embedding_config["index_size"],
        }
