"""This module initializes and updates an embeddings index for industrial classification.

The module uses the `EmbeddingHandler` class from the
`industrial_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

"""

from industrial_classification_utils.embed.embedding import EmbeddingHandler

EXAMPLE_QUERY = "school teacher primary education"

print("Creating embeddings index...")
# Create the embeddings index
embed = EmbeddingHandler()
embed.embed_index(from_empty=False)
print(
    f"Embeddings index created with {embed._index_size} entries."  # pylint: disable=protected-access
)
results = embed.search_index(EXAMPLE_QUERY)
print(f"Search results for '{EXAMPLE_QUERY}': {results}")
