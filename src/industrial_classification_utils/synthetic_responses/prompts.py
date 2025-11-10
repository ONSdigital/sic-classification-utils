"""This module contains prompt component templates, and helper functions for
constructing prompt templates, which enable generation of synthetic responses
to survey questions.

This module is currently limited to a template and template constructor to
request an LLM to answer a SIC follow-up question.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from industrial_classification_utils.llm.prompt import _core_prompt

from .response_models import FollowupAnswerResponse, RephraseDescription

# from industrial_classification_utils.models.response_model import RephraseDescription


def _persona_prompt(persona) -> str:
    """Constructs a section of the LLM prompt template, informing it about
    the personal characteristics (if any) of the persona it should
    emulate.
    """
    if persona is None:
        return """
               You are a UK worker, responding to a survey that has something
               to do with jobs or industries.
               You are a busy person, and don't want to spend too long on this
               survey, you want to provide an answer to the question, but you
               don't have the time to provide a very detailed response.
               """
    # TODO # pylint: disable=fixme
    return ""


_REMINDER_TEMPLATE = """
Below is a reminder of the main activity your
company does, your job title and job description.
The survey interviewer has not been able to fully classify your company's
Standard Industrial Classification (SIC) from the Job Data you have provided so far,
and so they have asked you a clarifying question.
Please answer this clarifying question, using the Output Format specified below.

===Output Format===
{format_instructions}

===Your Job Data===
- Company's main activity: {org_description}
- Job Title: {job_title}
- Job Description: {job_description}

===Survey Data===
- clarifying question: {followup_question}
"""


def make_followup_answer_prompt_pydantic(
    persona, request_body: dict, followup_question: str
):
    """Constructs a prompt for answering a follow-up question, formatted for a Pydantic output.

    Args:
        persona (TODO): An object describing the characteristics of the persona to emulate.
        request_body (dict): A dictionary containing the survey response data, including
                             "org_description",
                             "job_title",
                             "job_description".
        followup_question (str): The clarifying question from the survey interviewer.

    Returns:
        PromptTemplate: A Langchain PromptTemplate ready to be used with an LLM.
    """
    parser = PydanticOutputParser(pydantic_object=FollowupAnswerResponse)  # type: ignore
    persona_prompt = _persona_prompt(persona)
    return PromptTemplate.from_template(
        template=persona_prompt + _REMINDER_TEMPLATE,
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "org_description": request_body["industry_descr"],
            "job_title": request_body["job_title"],
            "job_description": request_body["job_description"],
            "followup_question": followup_question,
        },
    )


# pylint: disable=C0103
_rephrase_job_description = """You are an Expert Editorial Director and Information Synthesis
Specialist. Your task is to consolidate complex, multi-part descriptions of business activity
into a single, concise, and comprehensive label (2-10 words) for the main business's activity
and industry.

Chain of thought (DO NOT OUTPUT):
1. Extract **core activity**: identify the fundamental function or main business purpose
    described in the Original response.
2. Integrate **contextual details**: Identify clarifying details from the "follow up answer".

Objective:
- Produce a single response, that integrates the **core activity** and **contextual details**
    to fully describe the main activity of the business or organisation.

Input:
- Original response: {job_description}
Input content: The input consists of two elements:
1. Original response to question "Describe the main activity of the business or organisation".
2. Follow up answer.

Desired output:
- The output must consist **only** of the rephrased single label, followinng the format instructions.
- NEVER include the reasoning nor the chain of thought in your response.
- The final response must be a single label. It must be consise, and capture all details from the input.

Output format:
- Return output that strictly follows:
{format_instructions}
"""

parser_rephrase_job_description = PydanticOutputParser(
    pydantic_object=RephraseDescription
)


REPHRASE_JOB_DESCRIPTION = PromptTemplate.from_template(
    template=_core_prompt + _rephrase_job_description,
    partial_variables={
        "format_instructions": parser_rephrase_job_description.get_format_instructions(),
    },
)
