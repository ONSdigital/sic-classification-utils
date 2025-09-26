
# Synthetic Responses Module

## Overview

This module allows a user to generate synthetic responses to survey questions,
currently focusing on Survey Assist follow-up questions for SIC coding, using a specified 
persona and a Large Language Model (LLM).

It defines Pydantic models for validating responses to questions,
methods for constructing prompt templates for generating these synthetic responses, 
and querying an LLM with the constructed prompt to receive answers.

It provides a friendly interface for interacting with the synthetic response functionality,
through a `SyntheticResponder` class.
An instance of this class can be created with a specified LLM to query, persona characteristics
to emulate, and an (optional) specified function to retrieve questions - allowing for flexibility,
to retrieve questions from a variety of sources.

#### Assumptions

Whenever the Synthetic Responder is asked to do something (retrieve a question, construct a prompt, 
or answer a question), the method called will expect a parameter `body` to be given - this should
be either a JSON file or a dictionary, containing contextual infomation that the LLM should know
(e.g. the survey respondent's job information in the case of answering SA 
follow-up questions). 
To maintain flexibility, no requirements on the contents are set, but they must contain at least the 
keys which are referenced in the PromptTemplate used.

For the currently-provided PromptTemplates, the following keys are required to be present in the `body`;
* `job_description` 
* `job_title` 
* `industry_descr`

## TODO

#### Persona Specification
This module does not yet contain functionality to customise LLM prompts to have it emulate
specified characteristics (i.e. give it a fake persona), but boilerplate code has been put
in place so that that capability can be added at a later date.

#### Testing
Unit tests have been developed to provide coverage of this module, but will not be included in the initial
Pull Request to narrow the scope of what is to be reviewed. 
They will be added in a new PR shortly after, and this documentation will be updated.

#### Data Access Utilities
A set of utilities to extract questions from the (live, PoC) API, a locally running API, and
local datafiles has been developed, and will be added in a followup PR.

## Usage Example
```python
from industrial_classification_utils.synthetic_responses.synthetic_response_utils import SyntheticResponder

EXAMPLE_BODY = {"job_description": "Bake bread, cakes, pastries", 
                "job_title": "baker", 
                "industry_descr": "small scale, independently owned bakery"
                }

def get_question_example(body):
    # In this example, we return the same quetion each time,
    #  without considering the context of the survey response.
    return "Is your business better described as wholesale or retail?"

SR = SyntheticResponder(persona=None, 
                        get_question_function=get_question_example, 
                        model_name="gemini-2.5-flash")

follow_up_question = SR.get_question_function(EXAMPLE_BODY)
prompt_to_answer_followup = SR.construct_prompt(EXAMPLE_BODY, follow_up_question)
answer_to_followup_question = SR.answer_followup(prompt_to_answer_followup, EXAMPLE_BODY)

# follow_up_question = "Is your business better described as wholesale or retail?"
# answer_to_followup_question = "Retail"
    