"""This module defines response models for industrial classification utilities.

The models are implemented using Pydantic's `BaseModel` and are used to represent
various response structures for SIC (Standard Industrial Classification) code
assignment and classification tasks. These models include validation logic and
field-level constraints to ensure data integrity.

Classes:
    SicCandidate: Represents a candidate SIC code with associated information.
    SicResponse: Represents a response model for SIC code assignment.
    RagCandidate: Represents a candidate classification code with associated information.
    RagResponse: Represents a response model for classification code assignment.
    SurveyAssistSicResponse: Represents a response model for Survey Assist
                             classification code assignment.
    UnambiguousResponse: Represents a response model for unambiguous
                         classification code assignment.
    RankedCandidate: Represents a single ranked SIC code candidate
                     with its relevance score and reasoning.
    RerankingResponse: Response model for SIC code re-ranking results.

Constants:
    MAX_ALT_CANDIDATES: Maximum number of alternative candidates allowed in certain models.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from industrial_classification_utils.utils.constants import MAX_ALT_CANDIDATES


class SicCandidate(BaseModel):
    """Represents a candidate SIC code with associated information.

    Attributes:
        sic_code (str): Plausible SIC code based on the company activity description.
        sic_descriptive (str): Descriptive label of the SIC category associated with
            sic_code.
        likelihood (float): Likelihood of this sic_code with a value between 0 and 1.

    """

    sic_code: str = Field(
        description="Plausible SIC code based on the company activity description."
    )
    sic_descriptive: str = Field(
        description="Descriptive label of the SIC category associated with sic_code."
    )
    likelihood: float = Field(
        description="Likelihood of this sic_code with value between 0 and 1."
    )


class SicResponse(BaseModel):
    """Represents a response model for SIC code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide SIC code,
            False otherwise.
        followup (Optional[str]): Question to ask user in order to collect additional
            information to enable reliable SIC assignment. Empty if codable=True.
        sic_code (Optional[str]): Full SIC code (to the required number of digits)
            assigned based on the provided company activity description.
            Empty if codable=False.
        sic_descriptive (Optional[str]): Descriptive label of the SIC category
            associated with sic_code if provided. Empty if codable=False.
        sic_candidates (List[SicCandidate]): Short list of less than ten possible or
            alternative sic codes that may be applicable with their descriptive label
            and estimated likelihood.
        sic_code_2digits (Optional[str]): First two digits of the hierarchical SIC
            code assigned. This field should be non empty if the larger (two-digit)
            group of SIC codes can be determined even in cases where additional
            information is needed to code to four digits (for example when all
            SIC candidates share the same first two digits).
        reasoning (str): Specifies the information used to assign the SIC code or any
            additional information required to assign a SIC code.
    """

    codable: bool = Field(
        description="""True if enough information is provided to decide
        SIC code, False otherwise.""",
        default=False,
    )
    followup: Optional[str] = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable SIC assignment. Empty if codable=True.""",
        default=None,
    )
    sic_code: Optional[str] = Field(
        description="""Full SIC code (to the required number of digits) assigned based
        on provided the company activity description.  Empty if codable=False.""",
        default=None,
    )
    sic_descriptive: Optional[str] = Field(
        description="""Descriptive label of the SIC category associated with sic_code
        if provided. Empty if codable=False.""",
        default=None,
    )
    sic_candidates: list[SicCandidate] = Field(
        description="""Short list of less than ten possible or alternative SIC codes
        that may be applicable with their descriptive label and estimated likelihood.""",
        default=[],
    )

    reasoning: str = Field(
        description="""Step by step reasoning behind classification selected. Specifies
            the information used to assign the SIC code or any additional information
            required to assign a SIC code.""",
        default="No reasoning provided.",
    )

    @classmethod
    def sic_code_validator(cls, v):
        """Validates that a valid SIC code is provided if the response is codable.

        Args:
            v (str): The SIC code to validate.

        Returns:
            str: The validated SIC code.

        Raises:
            ValueError: If the SIC code is empty when codable is True.
        """
        if v == "":
            raise ValueError("If codable, then valid sic_code needs to be provided")
        return v

    @model_validator(mode="before")
    @classmethod
    def check_valid_fields(cls, values):
        """Validates the fields of the model before instantiation.

        Ensures that:
        - If `codable` is True, a valid `sic_code` is provided.
        - If `codable` is False, a follow-up question is provided.

        Args:
            values (dict): The dictionary of field values.

        Returns:
            dict: The validated field values.

        Raises:
            ValueError: If validation conditions are not met.
        """
        if values.get("codable"):
            cls.sic_code_validator(values.get("sic_code"))
        elif not values.get("followup"):  # This checks for None or empty string
            raise ValueError("If uncodable, a follow-up question needs to be provided.")
        return values


class RagCandidate(BaseModel):
    """Represents a candidate classification code with associated information.

    Attributes:
        class_code (str): Plausible classification code based on the respondent's data.
        class_descriptive (str): Descriptive label of the classification category
            associated with class_code.
        likelihood (float): Likelihood of this class_code with a value between 0 and 1.

    """

    class_code: str = Field(
        description="Plausible classification code based on the respondent's data."
    )
    class_descriptive: str = Field(
        description="""Descriptive label of the classification category
        associated with class_code."""
    )
    likelihood: float = Field(
        description="Likelihood of this class_code with value between 0 and 1."
    )


class RagResponse(BaseModel):
    """Represents a response model for classification code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide
            classification code, False otherwise.
        followup (Optional[str]): Question to ask user in order to collect
            additional information to enable reliable classification assignment.
            Empty if codable=True.
        class_code (Optional[str]): Full classification code (to the required
            number of digits) assigned based on provided respondent's data.
            Empty if codable=False.
        class_descriptive (Optional[str]): Descriptive label of the classification
            category associated with class_code if provided.
            Empty if codable=False.
        alt_candidates (list[RagCandidate]): Short list of less than ten possible
            or alternative classification codes that may be applicable with their
            descriptive label and estimated likelihood.
        reasoning (str): Step by step reasoning behind the classification selected.
            Specifies the information used to assign the SIC code or any additional
            information required to assign a SIC code.
    """

    codable: bool = Field(
        description="""True if enough information is provided to decide
        classification code, False otherwise."""
    )
    followup: Optional[str] = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment. Empty if codable=True.""",
        default=None,
    )
    class_code: Optional[str] = Field(
        description="""Full classification code (to the required number of digits)
        assigned based on provided respondent's data. Empty if codable=False.""",
        default=None,
    )
    class_descriptive: Optional[str] = Field(
        description="""Descriptive label of the classification category associated
        with class_code if provided. Empty if codable=False.""",
        default=None,
    )
    alt_candidates: list[RagCandidate] = Field(
        description="""Short list of less than ten possible or alternative
        classification codes that may be applicable with their descriptive label
        and estimated likelihood."""
    )
    reasoning: str = Field(
        description="""Step by step reasoning behind classification selected. Specifies
            the information used to assign the SIC code or any additional information
            required to assign a SIC code."""
    )


class SurveyAssistSicResponse(BaseModel):
    """Represents a response model for Survey Assist classification code assignment.

    Attributes:
        followup (str): Question to ask user in order to collect
            additional information to enable reliable classification assignment.
        sic_code (str): Full classification code (to the required
            number of digits) assigned based on provided respondent's data.
            This is the most likely coding.
        sic_descriptive (str): Descriptive label of the classification
            category associated with class_code if provided.
            This is the most likely coding.
        sic_candidates (list[RagCandidate]): Short list of less than ten possible
            or alternative classification codes that may be applicable with their
            descriptive label and estimated likelihood.
        reasoning (str): Step by step reasoning for the most likely classification
            selected.
            Specifies the information used to assign the SIC code or any additional
            information required to assign a SIC code.
    """

    followup: str = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment.""",
    )
    sic_code: Optional[str] = Field(
        description="""Full classification code (to the required number of digits)
        of the most likely canddate assigned based on provided respondent's data.""",
        default="",
    )
    sic_descriptive: Optional[str] = Field(
        description="""Descriptive label of the most likely classification category
        associated with sic_code.""",
        default="",
    )
    sic_candidates: list[SicCandidate] = Field(
        description="""Short list of less than ten possible or alternative SIC codes
        that may be applicable with their descriptive label and estimated likelihood."""
    )
    reasoning: str = Field(
        description="""Step by step reasoning behind the most likely classification
        selected. Specifies the information used to assign the SIC code or any
        additional information required to assign a SIC code."""
    )


class UnambiguousResponse(BaseModel):
    """Represents a response model for classification code assignment.

    Attributes:
        codable (bool): True only if enough information is provided to assign
            an unambiguous single classification code, False otherwise.
        class_code (Optional[str]): Full classification code (to the required number of digits)
            assigned based on provided respondent's data. Must be present if codable=True,
            must be None if codable=False.
        class_descriptive (Optional[str]): Descriptive label of the classification category.
            Must be present if codable=True, must be None if codable=False.
        alt_candidates (list[RagCandidate]): Short list of possible classification codes with their
            descriptive labels and estimated likelihoods.
        reasoning (str): Step by step reasoning behind the classification selected.
    """

    codable: bool = Field(
        description="True only if enough information is provided to decide an unambiguous "
        "classification code, False otherwise."
    )

    class_code: Optional[str] = Field(
        default=None,
        description="Full classification code (to the required number of digits) "
        "assigned based on provided respondent's data. Must be present if codable=True, "
        "must be None if codable=False.",
    )

    class_descriptive: Optional[str] = Field(
        default=None,
        description="Descriptive label of the classification category. "
        "Must be present if codable=True, must be None if codable=False.",
    )

    alt_candidates: list[RagCandidate] = Field(
        default_factory=list,
        description="Short list of possible classification codes with their "
        "descriptive labels and estimated likelihoods.",
        min_length=1,  # Ensure there's always at least one candidate
        max_length=10,  # Limit to less than 10 candidates
    )

    higher_level_code: Optional[str] = Field(
        description="Classification code with X notation to pad to 5 digits. "
        "Must be present if codable=False, must be None if codable=True.",
        default=None,
    )

    reasoning: str = Field(
        description="Step by step reasoning behind the classification selected.",
        min_length=50,  # Ensure detailed reasoning is provided
    )

    @field_validator("alt_candidates")
    @classmethod
    def validate_alt_candidates(cls, v):
        """Validates the number of alternative candidates.

        Ensures that the number of candidates is between 1 and the maximum allowed.

        Args:
            v (list): The list of alternative candidates.

        Returns:
            list: The validated list of candidates.

        Raises:
            ValueError: If the number of candidates is not within the allowed range.
        """
        if not 1 <= len(v) <= MAX_ALT_CANDIDATES:
            raise ValueError("alt_candidates must contain between 1 and 10 items.")
        return v


class RankedCandidate(BaseModel):
    """Represents a single ranked SIC code candidate with its relevance
        score and reasoning.

    Attributes:
        code (str): The SIC classification code
        title (str): The descriptive title of the SIC code
        relevance_score (float): Score between 0.0 and 1.0 representing relevance
        reasoning (str): Detailed explanation of the scoring decision
    """

    code: str = Field(
        description="The SIC classification code",
        # min_length=4,  # SIC codes are typically at least 4 digits
    )

    title: str = Field(
        description="The descriptive title of the SIC code", min_length=3
    )

    relevance_score: Annotated[
        float,
        Field(
            strict=True,
            ge=0,
            le=1,
            description="Score between 0.0 and 1.0 representing overall relevance",
        ),
    ]

    reasoning: str = Field(
        description="Detailed explanation of how the score was determined",
        min_length=30,  # Ensure substantial reasoning is provided
    )

    class Config:  # pylint: disable=too-few-public-methods
        """Configuration class for defining additional metadata for JSON schema.

        Attributes:
            json_schema_extra (dict): A dictionary containing extra information
                for the JSON schema. Includes an example with the following keys:
                - code (str): The industrial classification code.
                - title (str): The title or description of the classification.
                - relevance_score (float): A score indicating the relevance of the classification.
                - reasoning (str): A detailed explanation of the relevance score.
        """

        json_schema_extra = {  # noqa: RUF012
            "example": {
                "code": "11050",
                "title": "Manufacture of beer",
                "relevance_score": 0.95,
                "reasoning": "Perfect match with primary activity"
                "(manufacture of beer)",
            }
        }


class RerankingResponse(BaseModel):
    """Response model for SIC code re-ranking results.

    Attributes:
        selected_codes (List[RankedCandidate]): The top N most relevant codes
        excluded_codes (List[RankedCandidate]): All other considered codes
        status (str): Status of the response
        n_requested (int): Number of codes requested in the ranking
    """

    selected_codes: list[RankedCandidate] = Field(
        description="The top N most relevant codes", default_factory=list
    )

    excluded_codes: list[RankedCandidate] = Field(
        description="All other considered codes that weren't selected",
        default_factory=list,
    )

    status: str = Field(
        description="Status of the response", default="success", min_length=1
    )

    n_requested: int = Field(
        description="Number of codes requested in the ranking", gt=0
    )

    @model_validator(mode="after")
    def validate_selected_codes_count(self) -> "RerankingResponse":
        """Validates that the number of selected codes matches the requested count.

        This validation is only performed for successful responses.

        Returns:
            RerankingResponse: The validated response object.

        Raises:
            ValueError: If the number of selected codes is invalid for the given status.
        """
        if self.status == "success":  # Only validate count for successful responses
            if len(self.selected_codes) == 0:
                raise ValueError(
                    "Selected codes cannot be empty for successful responses"
                )
            if len(self.selected_codes) > self.n_requested:
                raise ValueError(
                    f"Number of selected codes ({len(self.selected_codes)}) "
                    f"must not exceed n_requested ({self.n_requested})"
                )
        return self


class FinalSICAssignment(BaseModel):
    """Response model for final assignment of a SIC code.

    Attributes:
        codable (bool): True if enough information is provided to assign
            an unambiguous single 5-digit classification code, False otherwise.
        unambiguous_code (Optional[str]): Full 5-digit classification code
            assigned based on provided respondent's data. Must be present if codable=True,
            must be None if codable=False.
        unambiguous_code_descriptive (Optional[str]): Descriptive label of the classification
            category. Must be present if codable=True, must be None if codable=False.
        higher_level_code (Optional[str]): Classification code with X notation to pad to 5 digits.
            Must be present if codable=False, must be None if codable=True.
        reasoning (str): Step by step reasoning behind the classification selected.
    """

    codable: bool = Field(
        description="True only if enough information is provided to decide an unambiguous "
        "classification code, False otherwise."
    )
    unambiguous_code: Optional[str] = Field(
        description="Full 5-digit classification code "
        "assigned based on provided respondent's data. Must be present if codable=True, "
        "must be None if codable=False."
    )
    unambiguous_code_descriptive: Optional[str] = Field(
        description="Descriptive label of the classification category. "
        "Must be present if codable=True, must be None if codable=False."
    )
    higher_level_code: Optional[str] = Field(
        description="Classification code with X notation to pad to 5 digits. "
        "Must be present if codable=False, must be None if codable=True."
    )
    reasoning: str = Field(
        description="Step by step reasoning behind the classification selected.",
        min_length=50,
    )


class OpenFollowUp(BaseModel):
    """Represents a response model for open ended follow-up question.

    Attributes:
        followup (str): Question to ask user in order to collect
            additional information to enable reliable classification assignment.
        reasoning (str): Reasoning explaining how follow-up question will help
            assign classification code.
    """

    followup: Optional[str] = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment.""",
        default="",
    )
    reasoning: str = Field(
        description="""Reasoning explaining how follow-up question will help
            assign classification code.""",
        default="",
    )


class ClosedFollowUpCandidate(BaseModel):
    """Represents a candidate classification code with associated information.

    Attributes:
        class_code (str): Standard Industrical Classification (SIC) code.
        class_descriptive (str): Simplified description of a SIC code that reflects
            respondent answers and context.
        class_example (str): Example of a business activity illustrating a SIC code.
    """

    class_code: str = Field(
        description="Standard Industrical Classification (SIC) code"
    )
    class_descriptive: str = Field(
        description="""Simplified description of a SIC code that reflects
            respondent answers and context."""
    )
    class_example: str = Field(
        description="""Example of a business activity illustrating a SIC code."""
    )


class ClosedFollowUp(BaseModel):
    """Represents a response model for open ended follow-up question.

    Attributes:
        followup (str): Question to ask user to choose from the provided options
            to enable reliable classification assignment.
        sic_options list[ClosedFollowUpCandidate]: List of simplified options
            for respondent to choose from.
        reasoning (str): Reasoning explaining how follow-up question will help
            assign classification code.
    """

    followup: Optional[str] = Field(
        description="""Question to ask user in order to collect additional information
        to enable reliable classification assignment.""",
        default="",
    )
    sic_options: list[ClosedFollowUpCandidate] = Field(
        description="""List of simplified options
            for respondent to choose from.""",
        default_factory=list,
    )
    reasoning: str = Field(
        description="""Reasoning explaining how official SIC code descriptions were
        simplified.""",
        default="",
    )
