"""This module initializes and updates an embeddings index for industrial classification.

The module uses the `EmbeddingHandler` class from the
`industrial_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

"""

# %%
from industrial_classification_utils.embed import (
    load_embedding_handler_from_sic_index_files,
)

EXAMPLE_QUERY = "school teacher primary education"

# %%
print("Creating embeddings index...")
# Create the embeddings index
embed = load_embedding_handler_from_sic_index_files(db_dir="./data/vector_store")
print(
    f"Embeddings index created with {embed.index_size} entries."  # pylint: disable=protected-access
)

# %%
results = embed.search_index(EXAMPLE_QUERY)
print(f"Search results for '{EXAMPLE_QUERY}': {results}")

# %%
