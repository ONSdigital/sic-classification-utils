"""This module initializes and updates an embeddings index for industrial classification.

The module uses the `EmbeddingHandler` class from the
`industrial_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

"""

# %%
from importlib.resources import files

from industrial_classification_utils.embed import (
    EmbeddingHandler,
    load_embedding_handler_from_sic_index_files,
)

DB_DIR = "./data/vector_store"

# %%
print("Creating embeddings example index...")
# Create the embeddings index

example_data = files("industrial_classification_utils.data.example").joinpath(
    "toy_index.txt"
)
embed1 = EmbeddingHandler(db_dir=DB_DIR, index_source_file=str(example_data))
print(
    f"Embeddings index created with {embed1.index_size} entries."  # pylint: disable=protected-access
)

print("\nExample search for most 'loyal' in the toy index:")
print(embed1.search_index("loyal").model_dump_json(indent=2))


# %%
# Alternative loading method using large published sic indices (xlsx)
embed2 = load_embedding_handler_from_sic_index_files(db_dir=DB_DIR)
print(
    f"Embeddings index created with {embed2.index_size} entries."  # pylint: disable=protected-access
)

# %%
EXAMPLE_QUERY = "Primary education"
results = embed2.search_index(EXAMPLE_QUERY)

print(f"Results for query '{EXAMPLE_QUERY}':", results.model_dump_json(indent=2))

# %%
