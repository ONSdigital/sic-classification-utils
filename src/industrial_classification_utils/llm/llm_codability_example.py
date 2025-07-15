"""This module provides illustration for the use of unambiguous prompt with an LLM."""

from pprint import pprint

from industrial_classification_utils.llm.llm import ClassificationLLM

LLM_MODEL = "gemini-1.5-flash"
JOB_TITLE = "school teacher"
JOB_DESCRIPTION = "teach maths"
ORG_DESCRIPTION = "school"


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

uni_chat = ClassificationLLM(model_name=LLM_MODEL, verbose=True)

sic_candidates = uni_chat._prompt_candidate_list(
    EXAMPLE_EMBED_SHORT_LIST, code_digits=5, candidates_limit=7
)

sa_response = uni_chat.unambiguous_sic_code(
    industry_descr=ORG_DESCRIPTION,
    job_title=JOB_TITLE,
    job_description=JOB_DESCRIPTION,
    sic_candidates=sic_candidates,
)

pprint(sa_response[0].model_dump(), indent=2, width=80)
