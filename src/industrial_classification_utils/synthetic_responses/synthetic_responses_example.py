"""This is a basic example usage script for the synthetic_responses module."""

from industrial_classification_utils.synthetic_responses.synthetic_response_utils import (
    SyntheticResponder,
)

EXAMPLE_BODY = {
    "industry_descr": "small scale, independently owned bakery",
    "job_title": "baker",
    "job_description": "Bake bread, cakes, pastries",
}
EXAMPLE_FOLLOWUP = (
    "Does your bakery primarily sell its products directly to"
    " consumers or to other businesses (e.g., cafes, restaurants)?"
)

SR = SyntheticResponder(
    persona=None, get_question_function=None, model_name="gemini-1.5-flash"
)
answer_followup_prompt = SR.construct_prompt(EXAMPLE_BODY, EXAMPLE_FOLLOWUP)
llm_response = SR.answer_followup(answer_followup_prompt, EXAMPLE_BODY)

print(f"FOLLOWUP QUESTION: {EXAMPLE_FOLLOWUP}")
print(f"LLM ANSWER: {llm_response}")
