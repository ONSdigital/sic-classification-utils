# pylint: disable=invalid-name, protected-access, line-too-long, missing-module-docstring, duplicate-code

from pprint import pprint

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.llm.llm import ClassificationLLM

embed = EmbeddingHandler()
uni_chat = ClassificationLLM(model_name="gemini-2.0-flash", verbose=True)

# Inputs for ClassificationLLM methods
industry_descr = "adult social care"
job_title = "community assessment officer"
job_description = "social services"

short_list = embed.search_index_multi(
    query=[industry_descr, job_title, job_description]
)

sa_soc_rag = uni_chat.sa_rag_sic_code(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    candidates_limit=10,
    short_list=short_list,
)

candidate_list = uni_chat._prompt_candidate_list(sa_soc_rag[1])  # type: ignore

sic_response_unambiguous = uni_chat.unambiguous_sic_code(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    sic_candidates=candidate_list,
)

# Formulate Open Question
sic_followup = uni_chat.formulate_open_question(
    industry_descr=industry_descr,
    job_title=job_title,
    job_description=job_description,
    llm_output=sa_soc_rag[0].sic_candidates,  # type: ignore
)

print("Open Question answer: Follow-up and Reasoning")
pprint(sic_followup[0].model_dump(), indent=2, width=80)

filtered_list = [elem.class_code for elem in sic_response_unambiguous[0].alt_candidates]
filtered_candidates = uni_chat._prompt_candidate_list_filtered(
    sa_soc_rag[1], filtered_list=filtered_list, activities_limit=5  # type: ignore
)

# # Formulate Closed Quesiton
# sic_closed_followup = uni_chat.formulate_closed_question(
#     industry_descr=industry_descr,
#     job_title=job_title,
#     job_description=job_description,
#     llm_output=filtered_candidates,
# )
# print(
#     """\nClosed Quesiton answer: Follow-up, Reasoning, and List of simplified SIC options to choose from"""
# )
# pprint(sic_closed_followup[0].model_dump(), indent=2, width=80)

# print("\nCandidate list for the prompt")
# pprint(filtered_candidates, indent=2, width=80)
