"""This module contains tests for the EmbeddingHandler class, focusing on embedding
and searching functionalities.
"""

from unittest.mock import patch

import pytest
from industrial_classification.hierarchy.sic_hierarchy import SIC, SicCode, SicNode

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


@pytest.fixture
def embedding_handler_sic():
    """Fixture to initialize an EmbeddingHandler instance with mock nodes.

    Returns:
        EmbeddingHandler: An instance of EmbeddingHandler with mock index embedded.
    """
    embedding_handler_sic = EmbeddingHandler(db_dir=None)
    nodes = [
        SicNode(sic_code=SicCode("A0111x"), description="Bird watching"),
        SicNode(sic_code=SicCode("A0112x"), description="Petting animals"),
    ]
    lookup = {}
    for node in nodes:
        lookup[str(node.sic_code)] = node
        lookup[node.sic_code.alpha_code] = node
        lookup[node.sic_code.alpha_code.replace("x", "")] = node
        if node.sic_code.n_digits > 1:
            lookup[node.sic_code.alpha_code[1:].replace("x", "")] = node

        if node.sic_code.n_digits == 4 and not node.children:  # noqa: PLR2004
            key = node.sic_code.alpha_code[1:5] + "0"
            lookup[key] = node
    sic = SIC(nodes=nodes, code_lookup=lookup)
    embedding_handler_sic.embed_index(sic=sic)
    return embedding_handler_sic


@pytest.mark.embed
def test_embed_index_with_sic_object(embedding_handler_sic):
    """Test embedding an index without a file object.

    Args:
        embedding_handler_sic  (EmbeddingHandler): The fixture providing the handler.
    """
    assert (
        embedding_handler_sic._index_size  # pylint: disable=protected-access
        == 2  # noqa: PLR2004
    )


@pytest.mark.parametrize(
    "model_name, expected_class",
    [
        ("textembedding-abc", "CustomVertexAIEmbeddings"),
        ("text-embedding-xyz", "CustomVertexAIEmbeddings"),
        ("other", "HuggingFaceEmbeddings"),
    ],
)
@pytest.mark.embed
def test_embedding_handler_initialization(model_name, expected_class):
    """Test embedding handles initialization.

    Args:
        model_name (str): name of embedding to be used in English.
        expected_class (str): name of embedding.
    """
    with patch(
        "industrial_classification_utils.embed.embedding.CustomVertexAIEmbeddings"
    ) as mock_google, patch(
        "industrial_classification_utils.embed.embedding.HuggingFaceEmbeddings"
    ) as mock_huggingface:
        EmbeddingHandler(model_name)

        if expected_class == "HuggingFaceEmbeddings":
            mock_huggingface.assert_called_once_with(model_name=model_name)
            mock_google.assert_not_called()
        else:
            mock_google.assert_called_once_with(model=model_name)
            mock_huggingface.assert_not_called()
