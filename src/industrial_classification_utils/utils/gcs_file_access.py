"""Provides GCS utils for file access.

This module contains utility functions to access files stored in Google Cloud Storage (GCS).
"""

import logging
import os
import posixpath
import tempfile
from dataclasses import dataclass

from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DownloadedVectorStore:
    """Represents a vector store downloaded from GCS.

    Attributes:
        path: The local path to the downloaded vector store.
        temp_dir: The temporary directory object that should be kept alive
            for as long as the downloaded files are needed.
    """

    path: str
    temp_dir: tempfile.TemporaryDirectory


def is_gcs_path(path: str) -> bool:
    """Check if a given path is a GCS URI."""
    return path.startswith("gs://")


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
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


def download_vector_store_from_gcs(gcs_uri: str) -> DownloadedVectorStore:
    """Download metadata.json and vectors.parquet from GCS into a temp directory.

    The returned object must be kept alive for as long as the downloaded files are needed.
    """
    bucket_name, prefix = parse_gcs_uri(gcs_uri)

    metadata_blob_name = (
        posixpath.join(prefix, "metadata.json") if prefix else "metadata.json"
    )
    vectors_blob_name = (
        posixpath.join(prefix, "vectors.parquet") if prefix else "vectors.parquet"
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

    temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    local_dir = temp_dir.name

    metadata_blob.download_to_filename(os.path.join(local_dir, "metadata.json"))
    vectors_blob.download_to_filename(os.path.join(local_dir, "vectors.parquet"))

    return DownloadedVectorStore(path=local_dir, temp_dir=temp_dir)
