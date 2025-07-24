"""This module provides utilities for generating synthetic responses to survey questions,
focusing initially on follow-up questions, using a specified persona and a Large Language
Model (LLM).

Classes:
    SyntheticResponder: A class for generating synthetic responses to survey questions.
    - Inputs:
        - persona (optional): TODO object describing the characteristics of the persona
          to emulate.
        - get_question_function (optional, callable): a helper function to retrieve
          question(s) from an API / data file.
        - model_name (str): The name of the LLM to use. Defaults to "gemini-1.5-flash".

    - Methods:
        - instantiate_llm: Initialises a VertexAI instance, using the model specified in
                           the class.
        - construct_prompt: Constructs a prompt for answering a follow-up question.
                            Requires arguments of 'body', a dictionary containing
                            contextual information about the survey response, and
                            'followup', a string containing the question to be answered.
        - answer_followup: Gets the LLM's response to the follow-up question.
                           Requires arguments of 'prompt', a PromptTemplate object
                           constructed to have the LLM respond to the question in the
                           given persona, and 'body', a dictionary containing contextual
                           information about the survey response.

Typical usage example:
    ```python
    from survey_assist_utils.synthetic_responses.synthetic_response_utils import SyntheticResponder

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
                            model_name="gemini-1.5-flash")

    follow_up_question = SR.get_question_function(EXAMPLE_BODY)

    prompt_to_answer_followup = SR.construct_prompt(EXAMPLE_BODY, follow_up_question)
    answer_to_followup_question = SR.answer_followup(prompt_to_answer_followup, EXAMPLE_BODY)
    ```


"""

import json
import logging
from typing import Callable, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain_google_vertexai import VertexAI

from industrial_classification_utils.embed.embedding import get_config

from .prompts import make_followup_answer_prompt_pydantic
from .response_models import FollowupAnswerResponse

logger = logging.getLogger(__name__)
config = get_config()


class SyntheticResponder:
    """This class provides functionality for generating synthetic responses to survey questions,
    particularly follow-up questions, using a specified persona and a Large Language Model (LLM).

    Attributes:
        get_question_function (optional, callable): A function that retrieves the follow-up
                                                    question from an API.
                                                    Defaults to None.
        persona (optional): A dictionary describing the demographic characteristics of the
                            persona the LLM should emulate.
                            Defaults to None.
        model_name (str): The name of the LLM to use.
                          Defaults to the value in the `config` file.
        llm: An instance of the LLM (currently VertexAI) used for generating responses.

    Methods:
        instantiate_llm:
            Initializes a VertexAI instance. Defaults to "gemini-1.5-flash".
        construct_prompt:
            Constructs the LLM prompt to answer a follow-up question, incorporating the persona
            and survey information.
        answer_followup:
            Gets the LLM's response to the follow-up question, using the provided prompt.
    """

    def __init__(
        self,
        get_question_function: Optional[Callable] = None,
        persona: Optional[dict] = None,
        model_name: str = config["llm"]["llm_model_name"],
    ):
        self.persona = persona
        self.get_question_function = get_question_function
        self.model_name = model_name
        self.instantiate_llm(model_name=self.model_name)
        logger.debug("SyntheticResponder initialised, connection to LLM established.")

    def instantiate_llm(self, model_name: str = "gemini-1.5-flash"):
        """Initialises a VertexAI instance."""
        try:
            self.llm = VertexAI(
                model_name=model_name,
                max_output_tokens=1_600,
                temperature=0.0,
                location="europe-west2",
            )
        except Exception as e:
            logger.error(f"{e}")
            logger.warning("connection to LLM failed")
            raise

    def construct_prompt(self, body: dict | str, followup: str) -> PromptTemplate:
        """Constructs and LLM prompt to respond to the followup question
        in a specified persona.
        """
        if type(body) not in (dict, str):
            logger.warning(
                "The object describing the context (body) could not be interpreted"
            )
            raise TypeError(
                "'body' argument must be either a dictionary or a (string) path to a JSON file"
            )
        if isinstance(body, str):
            body = json.load(body)  # type: ignore[arg-type]
        if isinstance(followup, str):
            return make_followup_answer_prompt_pydantic(
                persona=self.persona,
                request_body=body,  # type: ignore[arg-type]
                followup_question=followup,
            )
        logger.warning("No follow-up question provided")
        raise ValueError("No follow-up question provided.")

    def answer_followup(self, prompt: PromptTemplate, body: dict | str) -> str:
        """Gets the LLM's response to the followup question,
        as specified in the constructed prompt.
        """
        if type(body) not in (dict, str):
            logger.error(
                "The object describing the context (body) could not be interpreted"
            )
            raise TypeError(
                "'body' argument must be either a dictionary or a (string) path to a JSON file"
            )
        if isinstance(body, str):
            body = json.load(body)  # type: ignore[arg-type]
        call_dict = body.copy()  # type: ignore
        call_dict["followup_question"] = prompt.partial_variables["followup_question"]
        chain = prompt | self.llm  # LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke(call_dict, return_only_outputs=True)
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=FollowupAnswerResponse
        )
        try:
            validated_answer = parser.parse(str(response)).answer
            logger.debug("Answer received from LLM, and successfully parsed")
        except ValueError as parse_error:
            logger.error(f"{parse_error}")
            logger.warning(f"Failed to parse response:\n{response}")
        return validated_answer
