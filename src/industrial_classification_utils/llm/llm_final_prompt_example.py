"""This module provides illustration for the use of final SIC assignment prompt with an LLM."""

from pprint import pprint

from industrial_classification_utils.llm.llm import ClassificationLLM

# pylint: disable=duplicate-code

LLM_MODEL = "gemini-1.5-flash"
JOB_TITLE = "community assessment officer"
JOB_DESCRIPTION = "social services"
ORG_DESCRIPTION = "adult social care"
OPEN_QUESTION = "Could you briefly describe the main types of support or assessments"
"that you provide to individuals in your community?"
ANSWER_TO_OPEN_QUESTION = "housing and benefits assistance"
CLOSED_QUESTION = "Which of these best describes your organisation's activities?"
ANSWER_TO_CLOSED_QUESTION = "other social work activities without accommodation nec"


# The following is a mock LLM output for unambiguous_sic_code
SIC_CANDIDATES = [
    {
        "class_code": "88990",
        "class_descriptive": "Other social work activities without accommodation nec",
        "likelihood": 0.8,
    },
    {
        "class_code": "88100",
        "class_descriptive": "Social work activities without accommodation for the"
        "elderly and disabled",
        "likelihood": 0.6,
    },
    {
        "class_code": "84120",
        "class_descriptive": "Regulation of the activities of providing health care, education,"
        "cultural services and other social services, excluding social security",
        "likelihood": 0.3,
    },
]

uni_chat = ClassificationLLM(model_name=LLM_MODEL, verbose=True)

sa_response = uni_chat.final_sic_code(
    industry_descr=ORG_DESCRIPTION,
    job_title=JOB_TITLE,
    job_description=JOB_DESCRIPTION,
    sic_candidates=str(SIC_CANDIDATES),
    open_question=OPEN_QUESTION,
    answer_to_open_question=ANSWER_TO_OPEN_QUESTION,
    closed_question=CLOSED_QUESTION,
    answer_to_closed_question=ANSWER_TO_CLOSED_QUESTION,
)

pprint(sa_response[0].model_dump(), indent=2, width=80)
