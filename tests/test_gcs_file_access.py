# pylint: disable=missing-function-docstring
"""Tests for the EmbeddingHandler class and GCS file access helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from industrial_classification_utils.utils.gcs_file_access import (
    DownloadedVectorStore,
    download_vector_store_from_gcs,
    is_gcs_path,
    parse_gcs_uri,
)

NON_GCS_PATH = "local/path"


class _FakeBlob:
    """Fake GCS blob."""

    def __init__(self, exists: bool):
        self._exists = exists
        self.downloaded_to: str | None = None

    def exists(self):
        return self._exists

    def download_to_filename(self, filename: str):
        """Simulate downloading the blob by writing dummy content."""
        self.downloaded_to = filename
        Path(filename).write_text("test", encoding="utf-8")


class _FakeBucket:  # pylint: disable=too-few-public-methods
    """Fake GCS bucket."""

    def __init__(self, blob_map: dict[str, _FakeBlob]):
        self._blob_map = blob_map

    def blob(self, name: str):
        return self._blob_map[name]


class _FakeStorageClient:  # pylint: disable=too-few-public-methods
    """Fake GCS storage client."""

    def __init__(self, bucket_map: dict[str, _FakeBucket]):
        self._bucket_map = bucket_map

    def bucket(self, name: str):
        return self._bucket_map[name]


@pytest.mark.embed
def test_is_gcs_path():
    assert is_gcs_path("gs://bucket/path") is True
    assert is_gcs_path(NON_GCS_PATH) is False


@pytest.mark.embed
def test_parse_gcs_uri_valid():
    bucket, prefix = parse_gcs_uri("gs://my-bucket/path/to/store")

    assert bucket == "my-bucket"
    assert prefix == "path/to/store"


@pytest.mark.embed
def test_parse_gcs_uri_bucket_only():
    bucket, prefix = parse_gcs_uri("gs://my-bucket")

    assert bucket == "my-bucket"
    assert prefix == ""


@pytest.mark.embed
def test_parse_gcs_uri_invalid():
    with pytest.raises(ValueError, match="Not a valid GCS URI"):
        parse_gcs_uri("local/not-gcs")


@pytest.mark.embed
def test_parse_gcs_uri_missing_bucket():
    with pytest.raises(ValueError, match="bucket missing"):
        parse_gcs_uri("gs://")


@pytest.mark.embed
def test_download_vector_store_from_gcs_success():
    metadata_blob = _FakeBlob(exists=True)
    vectors_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "prefix/metadata.json": metadata_blob,
                    "prefix/vectors.parquet": vectors_blob,
                }
            )
        }
    )

    with patch(
        "industrial_classification_utils.utils.gcs_file_access.storage.Client",
        return_value=fake_client,
    ):
        downloaded = download_vector_store_from_gcs("gs://my-bucket/prefix")

    assert isinstance(downloaded, DownloadedVectorStore)
    assert Path(downloaded.path, "metadata.json").exists()
    assert Path(downloaded.path, "vectors.parquet").exists()
    assert downloaded.temp_dir is not None

    downloaded.temp_dir.cleanup()


@pytest.mark.embed
def test_download_vector_store_from_gcs_missing_files():
    metadata_blob = _FakeBlob(exists=False)
    vectors_blob = _FakeBlob(exists=True)

    fake_client = _FakeStorageClient(
        {
            "my-bucket": _FakeBucket(
                {
                    "prefix/metadata.json": metadata_blob,
                    "prefix/vectors.parquet": vectors_blob,
                }
            )
        }
    )

    with (
        patch(
            "industrial_classification_utils.utils.gcs_file_access.storage.Client",
            return_value=fake_client,
        ),
        pytest.raises(FileNotFoundError, match="metadata.json"),
    ):
        download_vector_store_from_gcs("gs://my-bucket/prefix")
