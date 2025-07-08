"""This module provides illustration for the use of unambiguous prompt with an LLM."""

from pprint import pprint

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.llm.llm import ClassificationLLM

embed = EmbeddingHandler()
uni_chat = ClassificationLLM("gemini-1.5-flash", embedding_handler=embed, verbose=True)

# Check if index was embedded
print(f"Size of the vector store is {uni_chat.embed._index_size}")

industry_descr = "education"
job_title = "teach"
job_description = "teach eglish"

search_results = uni_chat.embed.search_index_multi(
    [industry_descr, job_title, job_description]
)

short_list = uni_chat._prompt_candidate_list(
    search_results, code_digits=5, candidates_limit=7
)

sa_response = uni_chat.unambiguous_sic_code(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    shortlist=short_list,
)

pprint(sa_response[0].model_dump(), indent=2, width=80)
