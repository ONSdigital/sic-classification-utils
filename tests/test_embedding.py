# pylint: disable=missing-function-docstring
"""Tests for the EmbeddingHandler class and GCS file access helpers."""

from __future__ import annotations

import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from industrial_classification_utils.embed.embedding import EmbeddingHandler
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
    placeholder_store = SimpleNamespace(num_vectors=0)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=placeholder_store,
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
            "code": "02",
            "four_digit_code": "0200",
            "two_digit_code": "02",
        },
        {
            "doc_text": "cat",
            "score": 0.75,
            "code": "01",
            "four_digit_code": "0100",
            "two_digit_code": "01",
        },
    ]
    fake_store = SimpleNamespace(
        num_vectors=4,
        search=MagicMock(return_value=FakeSearchResults(rows)),
    )
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=fake_store,
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

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=built_store,
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

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=built_store,
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

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=built_store,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler._index_size == EXPECTED_TOY_INDEX_SIZE


@pytest.mark.embed
def test_search_index(embedding_handler_search: EmbeddingHandler):
    results = embedding_handler_search.search_index("mens best friend")

    assert results[0]["code"] == "02"
    assert results[0]["title"] == "dog"
    assert results[0]["distance"] == pytest.approx(0.01)


@pytest.mark.embed
def test_search_index_returns_tuples_when_requested(
    embedding_handler_search: EmbeddingHandler,
):
    results = embedding_handler_search.search_index(
        "mens best friend", return_dicts=False
    )

    assert results == [("dog", 0.99), ("cat", 0.75)]


@pytest.mark.embed
def test_search_index_uses_to_dicts_when_available(tmp_path: Path):
    rows = [
        {
            "doc_text": "dog",
            "score": 0.9,
            "code": "02",
            "four_digit_code": "0200",
            "two_digit_code": "02",
        }
    ]
    fake_store = SimpleNamespace(
        num_vectors=1,
        search=MagicMock(return_value=FakeSearchResultsWithToDicts(rows)),
    )
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=fake_store,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    results = handler.search_index("dog")

    assert results[0]["code"] == "02"
    assert results[0]["title"] == "dog"


@pytest.mark.embed
def test_search_index_multi(tmp_path: Path):
    placeholder_store = SimpleNamespace(num_vectors=0)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=placeholder_store,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    with patch.object(handler, "spell", side_effect=lambda x: x), patch.object(
        handler,
        "search_index",
        side_effect=[
            [{"code": "03", "distance": 0.4}, {"code": "04", "distance": 0.6}],
            [{"code": "03", "distance": 0.1}, {"code": "04", "distance": 0.2}],
        ],
    ):
        results = handler.search_index_multi(["has gills", "has scales"])

    assert len(results) == EXPECTED_MULTI_RESULTS
    assert results[0]["code"] == "03"
    assert results[0]["distance"] == EXPECTED_TOP_DISTANCE


@pytest.mark.embed
def test_search_index_multi_filters_none_values(tmp_path: Path):
    placeholder_store = SimpleNamespace(num_vectors=0)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=placeholder_store,
    ):
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    with patch.object(handler, "spell", side_effect=lambda x: x), patch.object(
        handler,
        "search_index",
        return_value=[{"code": "03", "distance": 0.3}],
    ) as mock_search:
        results = handler.search_index_multi([None, "has gills"])

    assert results == [{"code": "03", "distance": 0.3}]
    mock_search.assert_called_once_with(query="has gills")


@pytest.mark.embed
def test_embed_index_with_sic_object(embedding_handler_sic: EmbeddingHandler):
    assert embedding_handler_sic._index_size == EXPECTED_SIC_INDEX_SIZE


@pytest.mark.parametrize(
    ("model_name", "expected_class"),
    [
        ("textembedding-abc", "CustomVertexAIEmbeddings"),
        ("text-embedding-xyz", "CustomVertexAIEmbeddings"),
        ("other", "ChromaDBesqueHFVectoriser"),
    ],
)
@pytest.mark.embed
def test_embedding_handler_initialization(model_name, expected_class, tmp_path: Path):
    mock_vector_store = SimpleNamespace(num_vectors=123)

    with patch(
        "industrial_classification_utils.embed.embedding.CustomVertexAIEmbeddings"
    ) as mock_google, patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser"
    ) as mock_hf, patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=mock_vector_store,
    ):
        EmbeddingHandler(model_name, db_dir=str(tmp_path / "vector_store"))

        if expected_class == "ChromaDBesqueHFVectoriser":
            mock_hf.assert_called_once_with(
                model_name=f"sentence-transformers/{model_name}"
            )
            mock_google.assert_not_called()
        else:
            mock_google.assert_called_once_with(model=model_name)
            mock_hf.assert_not_called()


@pytest.mark.embed
def test_get_embed_config_returns_expected_keys(
    embedding_handler_for_embed: EmbeddingHandler,
):
    config = embedding_handler_for_embed.get_embed_config()

    assert "embedding_model_name" in config
    assert "db_dir" in config
    assert "sic_index" in config
    assert "sic_structure" in config
    assert "sic_condensed" in config
    assert "matches" in config
    assert "index_size" in config


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

    with patch(
        "industrial_classification_utils.embed.embedding.is_gcs_path",
        return_value=False,
    ), patch(
        "industrial_classification_utils.embed.embedding.VectorStore.from_filespace",
        return_value=fake_store,
    ) as mock_from_filespace:
        result = handler._load_existing_vector_store()

    assert result is fake_store
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

    with patch(
        "industrial_classification_utils.embed.embedding.is_gcs_path",
        return_value=False,
    ):
        result = handler._load_existing_vector_store()

    assert result is None


@pytest.mark.embed
def test_load_existing_vector_store_gcs():
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = "gs://my-bucket/prefix"
    handler.embeddings = object()
    handler._downloaded_vector_store = None

    fake_store = SimpleNamespace(num_vectors=55)

    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded = DownloadedVectorStore(
            path=temp_dir,
            temp_dir=None,
        )

        with patch(
            "industrial_classification_utils.embed.embedding.is_gcs_path",
            return_value=True,
        ), patch(
            "industrial_classification_utils.embed.embedding.download_vector_store_from_gcs",
            return_value=downloaded,
        ) as mock_download, patch(
            "industrial_classification_utils.embed.embedding.VectorStore.from_filespace",
            return_value=fake_store,
        ) as mock_from_filespace:
            result = handler._load_existing_vector_store()

    assert result is fake_store
    assert handler._downloaded_vector_store is downloaded
    mock_download.assert_called_once_with("gs://my-bucket/prefix")
    mock_from_filespace.assert_called_once_with(
        folder_path=downloaded.path,
        vectoriser=handler.embeddings,
        hooks=None,
    )


@pytest.mark.embed
def test_load_or_build_vector_store_raises_when_db_dir_missing():
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = None

    with pytest.raises(ValueError, match="db_dir must be provided"):
        handler._load_or_build_vector_store()


@pytest.mark.embed
def test_load_or_build_vector_store_returns_existing_store(tmp_path: Path):
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(tmp_path / "vector_store")

    existing_store = SimpleNamespace(num_vectors=7)

    with patch.object(
        handler, "_load_existing_vector_store", return_value=existing_store
    ) as mock_existing:
        result = handler._load_or_build_vector_store()

    assert result is existing_store
    mock_existing.assert_called_once()


@pytest.mark.embed
def test_load_or_build_vector_store_builds_from_sic_sources(tmp_path: Path):
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(tmp_path / "vector_store")
    handler.embeddings = object()
    handler.sic_index_file = "sic-index.csv"
    handler.sic_structure_file = "sic-structure.csv"

    fake_leaf_df = MagicMock()
    fake_leaf_df.iterrows.return_value = iter(
        [
            (0, {"code": "01.11", "text": "Growing of cereals"}),
            (1, {"code": "02/20", "text": "Forestry support"}),
        ]
    )

    fake_sic = MagicMock()
    fake_sic.all_leaf_text.return_value = fake_leaf_df

    built_store = SimpleNamespace(num_vectors=2)

    with patch.object(handler, "_load_existing_vector_store", return_value=None), patch(
        "industrial_classification_utils.embed.embedding.load_sic_index",
        return_value=MagicMock(),
    ) as mock_load_index, patch(
        "industrial_classification_utils.embed.embedding.load_sic_structure",
        return_value=MagicMock(),
    ) as mock_load_structure, patch(
        "industrial_classification_utils.embed.embedding.load_hierarchy",
        return_value=fake_sic,
    ) as mock_load_hierarchy, patch(
        "industrial_classification_utils.embed.embedding.VectorStore",
        return_value=built_store,
    ) as mock_vector_store:
        result = handler._load_or_build_vector_store()

    assert result is built_store
    mock_load_index.assert_called_once_with("sic-index.csv")
    mock_load_structure.assert_called_once_with("sic-structure.csv")
    mock_load_hierarchy.assert_called_once()
    assert mock_vector_store.called


@pytest.mark.embed
def test_load_or_build_vector_store_uses_default_metadata(tmp_path: Path):
    handler = EmbeddingHandler.__new__(EmbeddingHandler)
    handler.db_dir = str(tmp_path / "vector_store")
    handler.embeddings = object()
    handler.sic_index_file = "sic-index.csv"
    handler.sic_structure_file = "sic-structure.csv"

    fake_leaf_df = MagicMock()
    fake_leaf_df.iterrows.return_value = iter(
        [(0, {"code": "01.11", "text": "Growing of cereals"})]
    )

    fake_sic = MagicMock()
    fake_sic.all_leaf_text.return_value = fake_leaf_df

    built_store = SimpleNamespace(num_vectors=1)

    with patch.object(handler, "_load_existing_vector_store", return_value=None), patch(
        "industrial_classification_utils.embed.embedding.load_sic_index",
        return_value=MagicMock(),
    ), patch(
        "industrial_classification_utils.embed.embedding.load_sic_structure",
        return_value=MagicMock(),
    ), patch(
        "industrial_classification_utils.embed.embedding.load_hierarchy",
        return_value=fake_sic,
    ), patch(
        "industrial_classification_utils.embed.embedding.VectorStore",
        return_value=built_store,
    ) as mock_vector_store:
        handler._load_or_build_vector_store()

    assert mock_vector_store.call_args.kwargs["meta_data"] == {
        "code": str,
        "four_digit_code": str,
        "two_digit_code": str,
    }


@pytest.mark.embed
def test_embedding_handler_builds_vector_store_from_sic(tmp_path: Path) -> None:
    """It builds or loads the vector store during handler setup."""
    built_store = SimpleNamespace(num_vectors=2)
    fake_embeddings = SimpleNamespace(model_name="sentence-transformers/other")

    with patch(
        "industrial_classification_utils.embed.embedding.ChromaDBesqueHFVectoriser",
        return_value=fake_embeddings,
    ), patch(
        "industrial_classification_utils.embed.embedding."
        "EmbeddingHandler._load_or_build_vector_store",
        return_value=built_store,
    ) as mock_load_or_build:
        handler = EmbeddingHandler(
            embedding_model_name="other",
            db_dir=str(tmp_path / "vector_store"),
        )

    assert handler.vector_store is built_store
    mock_load_or_build.assert_called_once()
