"""This module initializes and updates an embeddings index for industrial classification
and then performs an llm lookup.

The module uses the `EmbeddingHandler` class from the
`industrial_classification_utils.embed.embedding` package to manage embeddings.
It provides functionality to create or update an embeddings index,
which can be used for tasks such as similarity searches or classification.

The example then uses llm from the `industrial_classification_utils.llm.llm_embedding_example`
package to perform a lookup using the embeddings index.
"""

import asyncio

from industrial_classification_utils.llm.llm import ClassificationLLM

# pylint: disable=duplicate-code

EXAMPLE_QUERY = "school teacher primary education"
LLM_MODEL = "gemini-2.5-flash"
JOB_TITLE = "school teacher"
JOB_DESCRIPTION = "teach maths"
ORG_DESCRIPTION = "school"
CANDIDATE_LIMIT = 100

# The following is a mock response for the embedding search
EXAMPLE_EMBED_SHORT_LIST = [
    {
        "distance": 0.6347243785858154,
        "title": "Education agent",
        "code": "85600",
        "four_digit_code": "8560",
        "two_digit_code": "85",
    },
    {
        "distance": 0.6422433257102966,
        "title": "Teacher n.e.c.",
        "code": "85590",
        "four_digit_code": "8559",
        "two_digit_code": "85",
    },
    {
        "distance": 0.7757259607315063,
        "title": "Teachers of sport",
        "code": "85510",
        "four_digit_code": "8551",
        "two_digit_code": "85",
    },
    {
        "distance": 0.8803297281265259,
        "title": "Kindergartens",
        "code": "85100",
        "four_digit_code": "8510",
        "two_digit_code": "85",
    },
]


# The vector store is decoupled from the LLM.
# The expectation is that the embedding will be queried and then
# the results will be passed to the LLM for classification.
# This example uses mocked data for the embedding search.
gemini_llm = ClassificationLLM(model_name=LLM_MODEL)

# Create single event loop for each async method to use
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

response_sic_code = loop.run_until_complete(
    gemini_llm.get_sic_code(ORG_DESCRIPTION, JOB_TITLE, JOB_DESCRIPTION)
)

print(response_sic_code)

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
print(response)

query_response, call_dict = loop.run_until_complete(
    gemini_llm.unambiguous_sic_code(
        industry_descr=ORG_DESCRIPTION,
        semantic_search_results=EXAMPLE_EMBED_SHORT_LIST,
        job_title=JOB_TITLE,
        job_description=JOB_DESCRIPTION,
    )
)

# Print the response
print(query_response)
