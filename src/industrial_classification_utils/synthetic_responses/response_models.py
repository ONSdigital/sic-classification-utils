"""This module defines Pydantic models for handling responses to questions in a survey.

Currently, the models in this module are limited to the handling of SurveyAssist
follow-up questions.

Classes:
    FollowupAnswerResponse: A Pydantic model representing a response to a followup question.
    FollowupAnswerRequest: A Pydantic model representing a request for an LLM to answer a
                           followup question.
"""

from typing import Optional

from pydantic import BaseModel, Field


class FollowupAnswerResponse(BaseModel):
    """Represents a response model for answering of the followup question.

    Attributes:
        answer (str): The answer to the followup question.
    """

    answer: str = Field(
        description="Answer to the followup question.",
        min_length=2,  # Ensure non-empty response, but it may be possible that
        # 'yes' or 'no' are acceptable answers to the question.
    )


class FollowupAnswerRequest(BaseModel):
    """Represents a response model for answering of the followup question.

    Attributes:
    .
            org_description (str): The company's main activity.
            job_title (str): The respondant's job title.
            job_description (str): The respondant's job description.
            followup_question (str): The LLM-generated followup question.
    """

    org_description: Optional[str] = Field(description="The company's main activity.")
    job_title: Optional[str] = Field(description="The respondant's job title.")
    job_description: Optional[str] = Field(
        description="The respondant's job description."
    )
    followup_question: Optional[str] = Field(
        description="The LLM-generated followup question."
    )


class RephraseDescription(BaseModel):
    """Represents a response model for rephrasing Job Description and Follow up answer.

    Attributes:
        job_description (str): Rephrased job description with information
    """

    job_description: str = Field(
        description="""Rephrased job description with follow up answer.""",
        default="",
    )
