"""This module contains tests for the EmbeddingHandler class, focusing on embedding
and searching functionalities.
"""

import pytest

from industrial_classification_utils.embed.embedding import EmbeddingHandler

# pylint: disable=redefined-outer-name


# %%
@pytest.fixture
def embedding_handler():
    """Fixture to initialize an EmbeddingHandler instance with a toy index.

    Returns:
        EmbeddingHandler: An instance of EmbeddingHandler with a toy index embedded.
    """
    embedding_handler = EmbeddingHandler(db_dir=None)
    file_path = "src/industrial_classification_utils/data/example/toy_index.txt"
    with open(file_path, encoding="utf-8") as file_object:
        embedding_handler.embed_index(from_empty=True, file_object=file_object)
    return embedding_handler


@pytest.mark.embed
def test_embed_index_with_file_object(embedding_handler):
    """Test embedding an index from a file object.

    Args:
        embedding_handler (EmbeddingHandler): The fixture providing the handler.

    Asserts:
        The number of entries in the index is as expected.
    """
    # Count number of entries
    assert (
        embedding_handler._index_size  # pylint: disable=protected-access
        == 4  # noqa: PLR2004
    )


@pytest.mark.embed
def test_search_index(embedding_handler):
    """Test searching the index with a single query.

    Args:
        embedding_handler (EmbeddingHandler): The fixture providing the handler.

    Asserts:
        The top result matches the expected code.
    """
    # Test searching index with a query
    query = "mens best friend"
    results = embedding_handler.search_index(query)
    assert results[0]["code"] == "02"


@pytest.mark.embed
def test_search_index_multi(embedding_handler):
    """Test searching the index with multiple queries.

    Args:
        embedding_handler (EmbeddingHandler): The fixture providing the handler.

    Asserts:
        The total number of results matches the expected count.
    """
    # Test searching index with multiple queries
    queries = ["has gills", "has scales"]
    results = embedding_handler.search_index_multi(queries)
    assert len(results) == 8  # noqa: PLR2004
