"""This module contains prompt component templates, and helper functions for
constructing prompt templates, which enable generation of synthetic responses
to survey questions.

This module is currently limited to a template and template constructor to
request an LLM to answer a SIC follow-up question.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from .response_models import FollowupAnswerResponse


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
               survey, you want to provide an answer the question, but you
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
