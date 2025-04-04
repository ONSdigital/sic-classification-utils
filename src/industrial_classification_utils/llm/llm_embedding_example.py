"""This module initializes and updates an embeddings index for industrial classification
and then performs an llm lookup.

The module uses the `EmbeddingHandler` class from the
`industrial_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

The example then uses llm from the `industrial_classification_utils.llm.llm_embedding_example`
package to perform a lookup using the embeddings index.
"""

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.llm.llm import ClassificationLLM

EXAMPLE_QUERY = "school teacher primary education"
LLM_MODEL = "gemini-1.5-flash"
JOB_TITLE = "school teacher"
JOB_DESCRIPTION = "teach maths"
ORG_DESCRIPTION = "school"
CANDIDATE_LIMIT = 100

print("Creating embeddings index...")
# Create the embeddings index
embed = EmbeddingHandler()
gemini_llm = ClassificationLLM(model_name=LLM_MODEL, embedding_handler=embed)

embed.embed_index(from_empty=False)
print(
    f"Embeddings index created with {embed._index_size} entries."  # pylint: disable=protected-access
)

print(f"Performing LLM lookup for {JOB_TITLE}...")

response, short_list, prompt = gemini_llm.sa_rag_sic_code(
    ORG_DESCRIPTION,
    JOB_TITLE,
    JOB_DESCRIPTION,
    candidates_limit=CANDIDATE_LIMIT,
)

# Print the response
print(response)
