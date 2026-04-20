"""This module initializes and updates an embeddings index for industrial classification
and then performs an llm lookup.

The module uses the `EmbeddingHandler` class from the
`industrial_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

The example then uses llm from the `industrial_classification_utils.llm.llm_embedding_example`
package to perform a lookup using the embeddings index.
"""

# %%
import asyncio
import json
from pathlib import Path

import nest_asyncio

from industrial_classification_utils.llm import ClassificationLLM

# pylint: disable=duplicate-code
# %%
nest_asyncio.apply()
DATA_DIR = Path(__file__).resolve().parent / "data"

EXAMPLE_QUERY = "school teacher primary education"
LLM_MODEL = "gemini-2.5-flash"
JOB_TITLE = "school teacher"
JOB_DESCRIPTION = "teach maths"
ORG_DESCRIPTION = "school"
CANDIDATE_LIMIT = 100

with (DATA_DIR / "school_embed_short_list.json").open(encoding="utf-8") as handle:
    EXAMPLE_EMBED_SHORT_LIST = json.load(handle)

# %%
# The vector store is decoupled from the LLM.
# The expectation is that the embedding will be queried and then
# the results will be passed to the LLM for classification.
# This example uses mocked data for the embedding search.
gemini_llm = ClassificationLLM(model_name=LLM_MODEL)
# %%
# Create single event loop for each async method to use
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

response_sic_code = loop.run_until_complete(
    gemini_llm.get_sic_code(ORG_DESCRIPTION, JOB_TITLE, JOB_DESCRIPTION)
)

print(response_sic_code.model_dump_json(indent=2))
# %%
response, short_list, prompt = loop.run_until_complete(
    gemini_llm.sa_rag_sic_code(
        ORG_DESCRIPTION,
        JOB_TITLE,
        JOB_DESCRIPTION,
        candidates_limit=CANDIDATE_LIMIT,
        short_list=EXAMPLE_EMBED_SHORT_LIST,
    )
)
# Print the response
print(response.model_dump_json(indent=2))
# %%
query_response, call_dict = loop.run_until_complete(
    gemini_llm.unambiguous_sic_code(
        industry_descr=ORG_DESCRIPTION,
        semantic_search_results=EXAMPLE_EMBED_SHORT_LIST,
        job_title=JOB_TITLE,
        job_description=JOB_DESCRIPTION,
    )
)

# Print the response
print(query_response.model_dump_json(indent=2))

# %%
