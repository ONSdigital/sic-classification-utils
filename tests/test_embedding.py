# pylint: disable=missing-function-docstring
"""Tests for the EmbeddingHandler class and GCS file access helpers."""

from __future__ import annotations

import logging
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from industrial_classification_utils.embed import (
    EmbeddingHandler,
    load_embedding_handler_from_sic_index_files,
)
from industrial_classification_utils.embed.embedding import ChromaDBesqueHFVectoriser
from industrial_classification_utils.models.config_model import EmbeddingStatus
from industrial_classification_utils.models.response_model import (
    SearchIndexItem,
    SearchIndexResponse,
)
from industrial_classification_utils.utils.gcs_file_access import (
    DownloadedVectorStore,
)

# pylint: disable=redefined-outer-name, protected-access

EXPECTED_TOY_INDEX_SIZE = 4
EXPECTED_SIC_INDEX_SIZE = 2
EXPECTED_MULTI_RESULTS = 4
EXPECTED_TOP_DISTANCE = 0.1


class FakeSearchResults:  # pylint: disable=too-few-public-methods
    """Simple dataframe-like object for VectorStore search results."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_dict(self, orient: str = "records") -> list[dict]:
        """Return the search results as a list of dicts."""
        assert orient == "records"
        return self._rows


class FakeSearchResultsWithToDicts:  # pylint: disable=too-few-public-methods
    """Simple object exposing to_dicts() for search results."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_dicts(self) -> list[dict]:
        """Return the search results as a list of dicts."""
        return self._rows


@pytest.fixture
def toy_index_file() -> StringIO:
    """Return a toy index as an in-memory file."""
    return StringIO(
        "\n".join(
            [
                "01: cat",
                "02: dog",
                "03: fish",
                "04: lizard",
            ]
        )
    )


@pytest.fixture
def embedding_handler_for_embed(tmp_path: Path) -> EmbeddingHandler:
    """Return a handler safe for embedding-related tests."""
    placeholder_store = SimpleNamespace(num_vectors=1)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(placeholder_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    return handler


@pytest.fixture
def embedding_handler_search(tmp_path: Path) -> EmbeddingHandler:
    """Return a handler with a fake searchable vector store."""
    rows = [
        {
            "doc_text": "dog",
            "score": 0.99,
            "doc_label": "02",
        },
        {
            "doc_text": "cat",
            "score": 0.75,
            "doc_label": "01",
        },
    ]
    fake_store = SimpleNamespace(
        num_vectors=4,
        search=MagicMock(return_value=FakeSearchResults(rows)),
    )
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(fake_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    return handler


@pytest.fixture
def embedding_handler_sic(tmp_path: Path) -> EmbeddingHandler:
    """Return an embedding handler backed by a mocked vector store."""
    built_store = SimpleNamespace(num_vectors=2)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(built_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    return handler


@pytest.mark.embed
def test_embedding_handler_init_sets_vector_store(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(built_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler.vector_store is built_store


@pytest.mark.embed
def test_embedding_handler_init_sets_index_size(tmp_path: Path) -> None:
    built_store = SimpleNamespace(num_vectors=EXPECTED_TOY_INDEX_SIZE)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(built_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler.index_size == EXPECTED_TOY_INDEX_SIZE


@pytest.mark.embed
def test_search_index(embedding_handler_search: EmbeddingHandler):
    response = embedding_handler_search.search_index("mens best friend")

    assert isinstance(response, SearchIndexResponse)
    assert response.results[0].code == "02"
    assert response.results[0].title == "dog"
    assert response.results[0].distance == pytest.approx(0.01)


@pytest.mark.embed
def test_search_index_uses_to_dicts_when_available(tmp_path: Path):
    rows = [
        {
            "doc_text": "dog",
            "score": 0.9,
            "doc_label": "02",
        }
    ]
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=FakeSearchResultsWithToDicts(rows)),
    )
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(fake_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    results = handler.search_index("dog")

    assert isinstance(results, SearchIndexResponse)
    assert results.results[0].code == "02"
    assert results.results[0].title == "dog"


@pytest.mark.embed
def test_search_index_multi(tmp_path: Path):
    placeholder_store = SimpleNamespace(num_vectors=1)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(placeholder_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    with (
        patch.object(handler, "spell", side_effect=lambda x: x),
        patch.object(
            handler,
            "search_index",
            side_effect=[
                SearchIndexResponse(
                    results=[
                        SearchIndexItem(code="03", title="fish", distance=0.4),
                        SearchIndexItem(code="04", title="lizard", distance=0.6),
                    ]
                ),
                SearchIndexResponse(
                    results=[
                        SearchIndexItem(code="03", title="fish", distance=0.1),
                        SearchIndexItem(code="04", title="lizard", distance=0.2),
                    ]
                ),
            ],
        ),
    ):
        response = handler.search_index_multi(["has gills", "has scales"])

    assert len(response.results) == EXPECTED_MULTI_RESULTS
    assert response.results[0].code == "03"
    assert response.results[0].distance == EXPECTED_TOP_DISTANCE


@pytest.mark.embed
def test_search_index_multi_filters_none_values(tmp_path: Path):
    placeholder_store = SimpleNamespace(num_vectors=1)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(placeholder_store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    with (
        patch.object(handler, "spell", side_effect=lambda x: x),
        patch.object(
            handler,
            "search_index",
            return_value=SearchIndexResponse(
                results=[SearchIndexItem(code="03", title="fish", distance=0.3)]
            ),
        ) as mock_search,
    ):
        response = handler.search_index_multi([None, "has gills"])

    assert response.results == [SearchIndexItem(code="03", title="fish", distance=0.3)]
    mock_search.assert_called_once_with(query="has gills")


@pytest.mark.embed
def test_embed_index_with_sic_object(embedding_handler_sic: EmbeddingHandler):
    assert embedding_handler_sic.index_size == EXPECTED_SIC_INDEX_SIZE


@pytest.mark.embed
def test_embedding_handler_initialization(tmp_path: Path):
    mock_vector_store = SimpleNamespace(num_vectors=123)

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser"
        ) as mock_hf,
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(mock_vector_store, "mock-source.csv"),
        ),
    ):
        EmbeddingHandler("other", db_dir=str(tmp_path / "vector_store"))

        mock_hf.assert_called_once_with(model_name="sentence-transformers/other")


@pytest.mark.embed
def test_load_existing_vector_store_local(tmp_path: Path):
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (db_dir / "vectors.parquet").write_text("dummy", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    fake_store = SimpleNamespace(num_vectors=42)

    with (
        patch(
            "industrial_classification_utils.embed.embedding.is_gcs_path",
            return_value=False,
        ),
        patch(
            "industrial_classification_utils.embed.embedding.VectorStore.from_filespace",
            return_value=fake_store,
        ) as mock_from_filespace,
    ):
        result = handler._load_existing_vector_store()

    assert result == (fake_store, None)
    mock_from_filespace.assert_called_once_with(
        folder_path=str(db_dir),
        vectoriser=handler.embeddings,
        hooks=None,
    )


@pytest.mark.embed
def test_load_existing_vector_store_local_missing_files(tmp_path: Path):
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "metadata.json").write_text("{}", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    with (
        patch(
            "industrial_classification_utils.embed.embedding.is_gcs_path",
            return_value=False,
        ),
        pytest.raises(FileNotFoundError, match="No existing vector store found"),
    ):
        handler._load_existing_vector_store()


@pytest.mark.embed
def test_load_existing_vector_store_gcs():
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = "gs://my-bucket/prefix"
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    fake_store = SimpleNamespace(num_vectors=55)

    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "metadata.json").write_text("{}", encoding="utf-8")
        Path(temp_dir, "vectors.parquet").write_text("dummy", encoding="utf-8")
        mock_temp_dir = SimpleNamespace(name=temp_dir)
        downloaded = DownloadedVectorStore(
            path=temp_dir,
            temp_dir=mock_temp_dir,
        )

        with (
            patch(
                "industrial_classification_utils.embed.embedding.is_gcs_path",
                return_value=True,
            ),
            patch(
                "industrial_classification_utils.embed.embedding.download_vector_store_from_gcs",
                return_value=downloaded,
            ) as mock_download,
            patch(
                "industrial_classification_utils.embed.embedding.VectorStore.from_filespace",
                return_value=fake_store,
            ) as mock_from_filespace,
        ):
            result = handler._load_existing_vector_store()

    assert result == (fake_store, None)
    assert handler._downloaded_vector_store is downloaded
    mock_download.assert_called_once_with("gs://my-bucket/prefix")
    mock_from_filespace.assert_called_once_with(
        folder_path=downloaded.temp_dir.name,
        vectoriser=handler.embeddings,
        hooks=None,
    )


@pytest.mark.embed
def test_load_existing_vector_store_raises_when_dir_missing(tmp_path: Path):
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(tmp_path / "nonexistent")
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    with (
        patch(
            "industrial_classification_utils.embed.embedding.is_gcs_path",
            return_value=False,
        ),
        pytest.raises(FileNotFoundError, match="No existing vector store found"),
    ):
        handler._load_existing_vector_store()


@pytest.mark.embed
def test_build_vector_store_raises_when_db_dir_missing():
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = None

    with pytest.raises(ValueError, match="db_dir must be provided"):
        handler._build_vector_store()


@pytest.mark.embed
def test_init_calls_load_existing_when_no_index_source(tmp_path: Path):
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")
    existing_store = SimpleNamespace(num_vectors=7)

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(existing_store, "mock-source.csv"),
        ) as mock_existing,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler.vector_store is existing_store
    mock_existing.assert_called_once()


@pytest.mark.embed
def test_load_embedding_handler_from_sic_index_files_builds(tmp_path: Path):
    fake_leaf_df = MagicMock()
    fake_leaf_df.rename.return_value = fake_leaf_df
    fake_leaf_df.to_csv = MagicMock()

    fake_sic = MagicMock()
    fake_sic.all_leaf_text.return_value = fake_leaf_df

    built_handler = SimpleNamespace(vector_store=SimpleNamespace(num_vectors=2))

    with (
        patch(
            "industrial_classification_utils.embed.sic_specific_embed.load_sic_hierarchy",
            return_value=fake_sic,
        ) as mock_load_sic_hierarchy,
        patch(
            "industrial_classification_utils.embed.sic_specific_embed.EmbeddingHandler",
            return_value=built_handler,
        ) as mock_handler_cls,
    ):
        result = load_embedding_handler_from_sic_index_files(
            db_dir=str(tmp_path / "vector_store"),
            sic_index_file="sic-index.csv",
            sic_structure_file="sic-structure.csv",
        )

    assert result is built_handler
    mock_load_sic_hierarchy.assert_called_once_with(
        "sic-index.csv", "sic-structure.csv"
    )
    assert mock_handler_cls.called
    call_kwargs = mock_handler_cls.call_args.kwargs
    assert call_kwargs["db_dir"] == str(tmp_path / "vector_store")
    assert "index_source_file" in call_kwargs


@pytest.mark.embed
def test_build_vector_store_passes_none_metadata(tmp_path: Path):
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler.index_source_file = "some-file.csv"

    built_store = SimpleNamespace(num_vectors=1)

    with patch(
        "industrial_classification_utils.embed.embedding.VectorStore",
        return_value=built_store,
    ) as mock_vector_store:
        handler._build_vector_store()

    assert mock_vector_store.call_args.kwargs["meta_data"] is None


@pytest.mark.embed
def test_embedding_handler_builds_vector_store_from_sic(tmp_path: Path) -> None:
    """It builds the vector store when index_source_file is provided."""
    built_store = SimpleNamespace(num_vectors=2)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._build_vector_store",
            return_value=built_store,
        ) as mock_build,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
            index_source_file="some-source.csv",
        )

    assert handler.vector_store is built_store
    mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# ChromaDBesqueHFVectoriser
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_chromadbesque_normalize_unit_vectors():
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]])
    result = inst._normalize(vectors)

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


@pytest.mark.embed
def test_chromadbesque_normalize_zero_vectors_do_not_divide_by_zero():
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = inst._normalize(vectors)

    assert np.allclose(result[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(result[1]), 1.0)


@pytest.mark.embed
def test_chromadbesque_transform_single_string_wraps_in_list():
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    fake_vec = np.array([[1.0, 0.0]])

    with patch(
        "industrial_classification_utils.embed.embedding.HuggingFaceVectoriser.transform",
        return_value=fake_vec,
    ) as mock_super:
        result = inst.transform("hello")

    mock_super.assert_called_once_with(["hello"])
    assert result.shape == (1, 2)


@pytest.mark.embed
def test_chromadbesque_transform_list_passes_through():
    inst = ChromaDBesqueHFVectoriser.__new__(ChromaDBesqueHFVectoriser)
    fake_vec = np.array([[1.0, 0.0], [0.0, 1.0]])

    with patch(
        "industrial_classification_utils.embed.embedding.HuggingFaceVectoriser.transform",
        return_value=fake_vec,
    ) as mock_super:
        result = inst.transform(["hello", "world"])

    mock_super.assert_called_once_with(["hello", "world"])
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# EmbeddingHandler.__init__ — GCS db_dir update
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_init_updates_db_dir_to_temp_dir_when_loading_from_gcs(tmp_path: Path):
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")
    gcs_temp_dir = tmp_path / "gcs-tempdir"
    gcs_temp_dir.mkdir()
    mock_temp = SimpleNamespace(name=str(gcs_temp_dir))
    downloaded_store = SimpleNamespace(temp_dir=mock_temp, num_vectors=3)
    # Simulate _load_existing_vector_store setting _downloaded_vector_store
    original_db_dir = "gs://my-bucket/prefix"

    def fake_load(self):
        self._downloaded_vector_store = downloaded_store
        return (downloaded_store, original_db_dir)

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch.object(EmbeddingHandler, "_load_existing_vector_store", fake_load),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=original_db_dir,
        )

    assert handler.db_dir == original_db_dir
    assert handler.index_source_file == original_db_dir


# ---------------------------------------------------------------------------
# _build_vector_store — overwrite warning
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_build_vector_store_logs_warning_when_parquet_exists(tmp_path: Path, caplog):
    db_dir = tmp_path / "vector_store"
    db_dir.mkdir()
    (db_dir / "vectors.parquet").write_text("dummy", encoding="utf-8")

    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(db_dir)
    handler.embeddings = object()
    handler.index_source_file = "some-file.csv"

    built_store = SimpleNamespace(num_vectors=1)

    with (
        caplog.at_level(
            logging.WARNING, logger="industrial_classification_utils.embed.embedding"
        ),
        patch(
            "industrial_classification_utils.embed.embedding.VectorStore",
            return_value=built_store,
        ),
    ):
        handler._build_vector_store()

    assert any("overwritten" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# search_index_multi — all-None input
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_search_index_multi_all_none_returns_empty(
    embedding_handler_for_embed: EmbeddingHandler,
):
    response = embedding_handler_for_embed.search_index_multi([None, None])

    assert isinstance(response, SearchIndexResponse)
    assert response.results == []


# ---------------------------------------------------------------------------
# get_embed_config — values
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_get_embed_config_returns_correct_values(tmp_path: Path):
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")
    store = SimpleNamespace(num_vectors=7)

    with (
        patch(
            "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
            return_value=fake_embeddings,
        ),
        patch(
            "industrial_classification_utils.embed.embedding."
            "EmbeddingHandler._load_existing_vector_store",
            return_value=(store, "mock-source.csv"),
        ),
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
            k_matches=5,
        )

    cfg = handler.get_embed_config()

    assert cfg.embedding_model_name == "other"
    assert cfg.db_dir == str(tmp_path / "vector_store")
    assert cfg.k_matches == 5
    assert cfg.index_size == 7


# ---------------------------------------------------------------------------
# load_embedding_handler_from_sic_index_files — kwargs forwarding
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_load_embedding_handler_from_sic_index_files_forwards_kwargs(tmp_path: Path):
    fake_leaf_df = MagicMock()
    fake_leaf_df.rename.return_value = fake_leaf_df
    fake_leaf_df.to_csv = MagicMock()

    fake_sic = MagicMock()
    fake_sic.all_leaf_text.return_value = fake_leaf_df

    built_handler = SimpleNamespace()

    with (
        patch(
            "industrial_classification_utils.embed.sic_specific_embed.load_sic_hierarchy",
            return_value=fake_sic,
        ),
        patch(
            "industrial_classification_utils.embed.sic_specific_embed.EmbeddingHandler",
            return_value=built_handler,
        ) as mock_handler_cls,
    ):
        load_embedding_handler_from_sic_index_files(
            db_dir=str(tmp_path / "vs"),
            sic_index_file="idx.csv",
            sic_structure_file="struct.csv",
            k_matches=42,
        )

    call_kwargs = mock_handler_cls.call_args.kwargs
    assert call_kwargs.get("k_matches") == 42


# ---------------------------------------------------------------------------
# EmbeddingStatus
# ---------------------------------------------------------------------------


@pytest.mark.embed
def test_embedding_status_valid():
    status = EmbeddingStatus(
        embedding_model_name="all-MiniLM-L6-v2",
        db_dir="/some/dir",
        k_matches=10,
        index_source_file="source.csv",
        status="ready",
        index_size=100,
    )
    assert status.status == "ready"
    assert status.index_size == 100


@pytest.mark.embed
def test_embedding_status_rejects_zero_index_size():
    with pytest.raises(ValueError, match="index_size must be at least 1"):
        EmbeddingStatus(
            embedding_model_name="all-MiniLM-L6-v2",
            db_dir="/some/dir",
            k_matches=10,
            index_source_file="source.csv",
            status="ready",
            index_size=0,
        )


@pytest.mark.embed
def test_embedding_status_rejects_empty_model_name():
    with pytest.raises(ValueError, match="embedding_model_name must be a valid value"):
        EmbeddingStatus(
            embedding_model_name="",
            db_dir="/some/dir",
            k_matches=10,
            index_source_file="source.csv",
            status="ready",
            index_size=5,
        )


@pytest.mark.embed
def test_embedding_status_rejects_unknown_db_dir():
    with pytest.raises(ValueError, match="db_dir must be a valid value"):
        EmbeddingStatus(
            embedding_model_name="all-MiniLM-L6-v2",
            db_dir="unknown",
            k_matches=10,
            index_source_file="source.csv",
            status="ready",
            index_size=5,
        )


@pytest.mark.embed
def test_embedding_status_non_ready_skips_validation():
    # index_size=0 and index_source_file=None are fine when not "ready"
    status = EmbeddingStatus(
        embedding_model_name="all-MiniLM-L6-v2",
        db_dir="/some/dir",
        k_matches=10,
        index_source_file=None,
        status="initialised",
        index_size=0,
    )
    assert status.status == "initialised"
