# pylint: disable=logging-not-lazy,logging-fstring-interpolation,too-many-lines
"""This module provides utilities for leveraging Large Language Models (LLMs)
to classify respondent data into Standard Industrial Classification (SIC) codes.

The `ClassificationLLM` class encapsulates the logic for using LLMs to perform
classification tasks, including direct generative methods and Retrieval Augmented
Generation (RAG). It supports various prompts and configurations for different
classification scenarios, such as unambiguous classification, reranking, and
general-purpose classification.

Classes:
    ClassificationLLM: A wrapper for LLM-based SIC classification logic.

Functions:
    (None at the module level)
"""

import time
from collections import defaultdict
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
from industrial_classification.hierarchy.sic_hierarchy import load_hierarchy
from industrial_classification.meta import sic_meta
from langchain.output_parsers import PydanticOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from survey_assist_utils.logging import get_logger

from industrial_classification_utils.embed.embedding import get_config
from industrial_classification_utils.llm.prompt import (
    FIX_PARSING_PROMPT,
    GENERAL_PROMPT_RAG,
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_CLOSEDFOLLOWUP,
    SIC_PROMPT_FINAL_ASSIGNMENT,
    SIC_PROMPT_OPENFOLLOWUP,
    SIC_PROMPT_PYDANTIC,
    SIC_PROMPT_RAG,
    SIC_PROMPT_RERANKER,
    SIC_PROMPT_UNAMBIGUOUS,
)
from industrial_classification_utils.models.response_model import (
    ClosedFollowUp,
    FinalSICAssignment,
    OpenFollowUp,
    RerankingResponse,
    SicCandidate,
    SicResponse,
    UnambiguousResponse,
)
from industrial_classification_utils.utils.constants import (
    truncate_identifier,
)
from industrial_classification_utils.utils.sic_data_access import (
    load_sic_index,
    load_sic_structure,
)

logger = get_logger(__name__)
config = get_config()


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
class ClassificationLLM:
    """Wraps the logic for using an LLM to classify respondent's data
    based on provided index. Includes direct (one-shot) generative llm
    method and Retrieval Augmented Generation (RAG).

    Args:
        model_name (str): Name of the model. Defaults to the value in the `config` file.
            Used if no LLM object is passed.
        llm (LLM): LLM to use. Optional.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 1600.
        temperature (float): Temperature of the LLM model. Defaults to 0.0.
        verbose (bool): Whether to print verbose output. Defaults to False.
        openai_api_key (str): OpenAI API key. Optional, but needed for OpenAI models.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = config["llm"]["llm_model_name"],
        llm: Optional[Union[ChatVertexAI, ChatOpenAI]] = None,
        max_tokens: int = 1600,
        temperature: float = 0.0,
        verbose: bool = True,
        openai_api_key: Optional[SecretStr] = None,
    ):
        """Initialises the ClassificationLLM object."""
        print(f"model_name: {model_name}")
        if llm is not None:
            self.llm = llm
        elif model_name.startswith("text-") or model_name.startswith("gemini"):
            self.llm = ChatVertexAI(
                model_name=model_name,
                max_output_tokens=max_tokens,
                temperature=temperature,
                location="europe-west1",
                model_kwargs={"thinking_budget": 0},  # Reduce latency
            )
        elif model_name.startswith("gpt"):
            if openai_api_key is None:
                raise NotImplementedError("Need to provide an OpenAI API key")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=openai_api_key,
                temperature=temperature,
                model_kwargs={"max_tokens": max_tokens},
            )
        else:
            raise NotImplementedError("Unsupported model family")

        self.sic_prompt = SIC_PROMPT_PYDANTIC
        self.sic_meta = sic_meta
        self.sic_prompt_rag = SIC_PROMPT_RAG
        self.sa_sic_prompt_rag = SA_SIC_PROMPT_RAG
        self.general_prompt_rag = GENERAL_PROMPT_RAG
        self.sic_prompt_unambiguous = SIC_PROMPT_UNAMBIGUOUS
        self.sic_prompt_reranker = SIC_PROMPT_RERANKER
        self.sic_prompt_openfollowup = SIC_PROMPT_OPENFOLLOWUP
        self.sic_prompt_closedfollowup = SIC_PROMPT_CLOSEDFOLLOWUP
        self.sic_prompt_final = SIC_PROMPT_FINAL_ASSIGNMENT
        self.sic = None
        self.verbose = verbose

    @lru_cache  # noqa: B019
    async def get_sic_code(
        self,
        industry_descr: str,
        job_title: str,
        job_description: str,
    ) -> SicResponse:
        """Generates a SIC classification based on respondent's data
        using a whole condensed index embedded in the query.

        Args:
            industry_descr (str): Description of the industry.
            job_title (str): Title of the job.
            job_description (str): Description of the job.

        Returns:
            SicResponse: Generated response to the query.
        """
        chain = self.sic_prompt | self.llm
        response = await chain.ainvoke(
            {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
            },
            return_only_outputs=True,
        )
        if self.verbose:
            logger.debug(f"LLM response: {response}")
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SicResponse
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.debug(
                f"Retrying llm response parsing due to an error: {parse_error}"
            )
            logger.error(f"Unable to parse llm response: {parse_error}")

            reasoning = (
                f"ERROR parse_error=<{parse_error}>, response=<{response.content}>"
            )
            validated_answer = SicResponse(
                codable=False,
                sic_candidates=[],
                reasoning=reasoning,
            )

        return validated_answer

    def _prompt_candidate(
        self, code: str, activities: list[str], include_all: bool = False
    ) -> str:
        """Reformat the candidate activities for the prompt.

        Args:
            code (str): The code for the item.
            activities (list[str]): The list of example activities.
            include_all (bool, optional): Whether to include all the sic metadata.

        Returns:
            str: A formatted string containing the code, title, and example activities.
        """
        if self.sic is None:
            sic_index_df = load_sic_index(config["lookups"]["sic_index"])
            sic_df = load_sic_structure(config["lookups"]["sic_structure"])
            self.sic = load_hierarchy(sic_df, sic_index_df)

        item = self.sic[code]  # type: ignore # MyPy false positive
        txt = "{" + f"Code: {item.numeric_string_padded()}, Title: {item.description}"
        txt += f", Example activities: {', '.join(activities)}"
        if include_all:
            if item.sic_meta.detail:
                txt += f", Details: {item.sic_meta.detail}"
            if item.sic_meta.includes:
                txt += f", Includes: {', '.join(item.sic_meta.includes)}"
            if item.sic_meta.excludes:
                txt += f", Excludes: {', '.join(item.sic_meta.excludes)}"
        return txt + "}"

    def _prompt_candidate_list(
        self,
        short_list: list[dict],
        chars_limit: int = 14000,
        candidates_limit: int = 5,
        activities_limit: int = 3,
        code_digits: int = 5,
    ) -> str:
        """Create candidate list for the prompt based on the given parameters.

        This method takes a structured list of candidates and generates a short
        string list based on the provided parameters. It filters the candidates
        based on the code digits and activities limit, and shortens the list to
        fit the character limit.

        Args:
            short_list (list[dict]): A list of candidate dictionaries.
            chars_limit (int, optional): The character limit for the generated
                prompt. Defaults to 14000.
            candidates_limit (int, optional): The maximum number of candidates
                to include in the prompt. Defaults to 5.
            activities_limit (int, optional): The maximum number of activities
                to include for each code. Defaults to 3.
            code_digits (int, optional): The number of digits to consider from
                the code for filtering candidates. Defaults to 5.

        Returns:
            str: The generated candidate list for the prompt.
        """
        a: defaultdict[Any, list] = defaultdict(list)

        logger.debug(
            f"Chars Lmt: {chars_limit} Candidate Lmt: {candidates_limit} "
            f"Activities Lmt: {activities_limit} Short List Len: {len(short_list)} "
            f"Code Digits: {code_digits}"
        )

        for item in short_list:
            if item["title"] not in a[item["code"][:code_digits]]:
                a[item["code"][:code_digits]].append(item["title"])

        sic_candidates = [
            self._prompt_candidate(code, activities[:activities_limit])
            for code, activities in a.items()
        ][:candidates_limit]

        if chars_limit:
            chars_count = np.cumsum([len(x) for x in sic_candidates])
            nn = sum(x <= chars_limit for x in chars_count)
            # nn = sum([x <= chars_limit for x in chars_count])
            if nn < len(sic_candidates):
                logger.warning(
                    f"Shortening list of candidates to fit token limit from "
                    f"{len(sic_candidates)} to {nn}"
                )
                sic_candidates = sic_candidates[:nn]

        return "\n".join(sic_candidates)

    def _prompt_candidate_list_filtered(  # noqa: PLR0913
        self,
        short_list: list[dict],
        chars_limit: int = 14000,
        candidates_limit: int = 5,
        activities_limit: int = 3,
        code_digits: int = 5,
        filtered_list: Optional[list[str]] = None,
    ) -> str:
        """Create candidate list for the prompt based on the given parameters.

        This method takes a structured list of candidates and generates a short
        string list based on the provided parameters. It filters the candidates
        based on the code digits and activities limit, and shortens the list to
        fit the character limit.

        Args:
            short_list (list[dict]): A list of candidate dictionaries.
            chars_limit (int, optional): The character limit for the generated
                prompt. Defaults to 14000.
            candidates_limit (int, optional): The maximum number of candidates
                to include in the prompt. Defaults to 5.
            activities_limit (int, optional): The maximum number of activities
                to include for each code. Defaults to 3.
            code_digits (int, optional): The number of digits to consider from
                the code for filtering candidates. Defaults to 5.
            filtered_list (list[str], optional): A list of alternative
                candidates.

        Returns:
            str: The generated candidate list for the prompt.
        """
        if not filtered_list:
            logger.warning("Empty list")
            return ""

        a: defaultdict[Any, list] = defaultdict(list)
        for item in short_list:
            if (
                item["code"] in filtered_list
                and item["title"] not in a[item["code"][:code_digits]]
            ):
                a[item["code"][:code_digits]].append(item["title"])

            sic_candidates = [
                self._prompt_candidate(code, activities[:activities_limit])
                for code, activities in a.items()
            ][:candidates_limit]

        if chars_limit:
            chars_count = np.cumsum([len(x) for x in sic_candidates])
            nn = sum(x <= chars_limit for x in chars_count)
            if nn < len(sic_candidates):
                logger.warning(  # pylint: disable=logging-not-lazy
                    "Shortening list of candidates to fit token limit "
                    + f"from {len(sic_candidates)} to {nn}"
                )
                sic_candidates = sic_candidates[:nn]

        return "\n".join(sic_candidates)

    async def sa_rag_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        code_digits: int = 5,
        candidates_limit: int = 5,
        short_list: Optional[list[dict[Any, Any]]] = None,
    ) -> tuple[SicResponse, Optional[list[dict[Any, Any]]], Optional[Any]]:
        """Generates a SIC classification based on respondent's data using RAG approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 5.
            short_list (list[dict[Any, Any]], optional): A list of results from embedding search

        Returns:
            SicResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """

        def prep_call_dict(industry_descr, job_title, job_description, sic_codes):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_index": sic_codes,
            }
            return call_dict

        if short_list is None:
            raise ValueError(
                "Short list is None - list provided from embedding search."
            )

        sic_codes = self._prompt_candidate_list(
            short_list, code_digits=code_digits, candidates_limit=candidates_limit
        )

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            sic_codes=sic_codes,
        )

        if self.verbose:
            final_prompt = self.sa_sic_prompt_rag.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sa_sic_prompt_rag | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = SicResponse(
                followup="Follow-up question not available due to error.",
                reasoning="Error from chain, exit early",
            )
            return validated_answer, short_list, call_dict
        if self.verbose:
            logger.debug(f"LLM response: {response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SicResponse
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            # send another llm request to fix the format (1 attempt)
            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")
            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = SicResponse(
                    followup="Follow-up question not available due to error.",
                    reasoning=reasoning,
                )

        return validated_answer, short_list, call_dict

    async def unambiguous_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        semantic_search_results: list[dict],
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        candidates_limit: int = 5,
        code_digits: int = 5,
        correlation_id: Optional[str] = None,
    ) -> tuple[UnambiguousResponse, Optional[Any]]:
        """Evaluates codability to a single 5-digit SIC code based on respondent's data.

        Args:
            industry_descr (str): The description of the industry.
            semantic_search_results (list of dicts): List of semantic search results.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            candidates_limit (int, optional): The maximum number of candidates
                to include in the prompt. Defaults to 5.
            code_digits (int, optional): The number of digits to consider from
                the code for filtering candidates. Defaults to 5.
            correlation_id (str, optional): Optional correlation ID for request tracking.

        Returns:
            UnambiguousResponse: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """
        sic_candidates = self._prompt_candidate_list(
            short_list=semantic_search_results,
            code_digits=code_digits,
            candidates_limit=candidates_limit,
        )

        job_title = (
            "Unknown" if (job_title is None or job_title in {"", " "}) else job_title
        )
        job_description = (
            "Unknown"
            if (job_description is None or job_description in {"", " "})
            else job_description
        )

        call_dict = {
            "industry_descr": industry_descr,
            "job_title": job_title,
            "job_description": job_description,
            "sic_candidates": sic_candidates,
        }

        if self.verbose:
            final_prompt = self.sic_prompt_unambiguous.format(**call_dict)
            logger.debug(final_prompt)

        chain = self.sic_prompt_unambiguous | self.llm

        # Log LLM request sent
        logger.info(
            "LLM request sent - unambiguous_sic_code",
            job_title=truncate_identifier(job_title),
            job_description=truncate_identifier(job_description),
            industry_descr=truncate_identifier(industry_descr),
            correlation_id=correlation_id or "",
        )
        llm_start = time.perf_counter()

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(
                f"Error from chain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            validated_answer = UnambiguousResponse(
                codable=False,
                alt_candidates=[],
                reasoning="Error from chain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"llm_response={response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=UnambiguousResponse)  # type: ignore
        try:
            validated_answer = parser.parse(str(response.content))
            # Log LLM response received after successful parse
            alt_candidates_count = len(
                getattr(validated_answer, "alt_candidates", []) or []
            )
            codable = bool(getattr(validated_answer, "codable", False))
            selected_code = (
                str(getattr(validated_answer, "class_code", "")) if codable else ""
            )
            llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.info(
                "LLM response received for unambiguous sic prompt",
                codable=str(codable),
                selected_code=selected_code,
                alt_candidates_count=str(alt_candidates_count),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )

            # send another llm request to fix the format (1 attempt)
            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")

            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = UnambiguousResponse(
                    codable=False,
                    alt_candidates=[],
                    reasoning=reasoning,
                )

        return validated_answer, call_dict

    async def reranker_sic(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        code_digits: int = 5,
        candidates_limit: int = 7,
        output_limit: int = 5,
        short_list: Optional[list[dict[Any, Any]]] = None,
    ) -> Union[tuple[Any, Optional[list], Optional[dict[str, Any]]], dict[str, Any]]:
        """Generates a set of relevant SIC codes based on respondent's data
            using reranking approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 7.
            output_limit (int, optional): The maximum number of SIC codes to return.
                Defaults to 5.
            short_list (list[dict[Any, Any]], optional): A list of results from embedding search.

        Returns:
            tuple[RerankingResponse, dict[str, Any]]: The reranking response and additional data.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """

        def prep_call_dict(
            industry_descr, job_title, job_description, sic_codes, output_limit
        ):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_index": sic_codes,
                "n": output_limit,
            }
            return call_dict

        if short_list is None:
            raise ValueError(
                "Short list is None - list provided from embedding search."
            )

        sic_codes = self._prompt_candidate_list(
            short_list, code_digits=code_digits, candidates_limit=candidates_limit
        )

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            sic_codes=sic_codes,
            output_limit=output_limit,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_reranker.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sic_prompt_reranker | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = RerankingResponse(
                selected_codes=[],
                excluded_codes=[],
                status="Error from chain, exit early",
                n_requested=output_limit,
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug(f"LLM response: {response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=RerankingResponse
        )
        try:
            validated_answer = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            reasoning = (
                f"ERROR parse_error=<{parse_error}>, response=<{response.content}>"
            )
            validated_answer = RerankingResponse(
                selected_codes=[],
                excluded_codes=[],
                status=reasoning,
                n_requested=output_limit,
            )

        return validated_answer, short_list, call_dict

    async def final_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        sic_candidates: Optional[str] = None,
        open_question: Optional[str] = None,
        answer_to_open_question: Optional[str] = None,
        closed_question: Optional[str] = None,
        answer_to_closed_question: Optional[str] = None,
    ) -> tuple[FinalSICAssignment, Optional[Any]]:
        """Evaluates codability to a single 5-digit SIC code based on respondent's data
            and answers to follow-up questions.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            sic_candidates: (str, optional): Short list of SIC candidates to pass to LLM.
            open_question (str, optional): The open question. Defaults to None.
            answer_to_open_question (str, optional): The answer to the open question.
                Defaults to None.
            closed_question (str, optional): The closed question. Defaults to None.
            answer_to_closed_question (str, optional): The answer to the closed question.
                Defaults to None.

        Returns:
            FinalSICAssignment: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """

        def prep_call_dict(  # noqa: PLR0913
            industry_descr,
            job_title,
            job_description,
            sic_candidates,
            open_question,
            answer_to_open_question,
            closed_question,
            answer_to_closed_question,
        ):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "sic_candidates": sic_candidates,
                "open_question": open_question,
                "answer_to_open_question": answer_to_open_question,
                "closed_question": closed_question,
                "answer_to_closed_question": answer_to_closed_question,
            }
            return call_dict

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            sic_candidates=sic_candidates,
            open_question=open_question,
            answer_to_open_question=answer_to_open_question,
            closed_question=closed_question,
            answer_to_closed_question=answer_to_closed_question,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_final.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sic_prompt_final | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(f"Error from chain, exit early: {err}", error=str(err))
            validated_answer = FinalSICAssignment(
                codable=False,
                unambiguous_code="N/A",
                unambiguous_code_descriptive="N/A",
                higher_level_code="N/A",
                reasoning="Error from chain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"llm_response={response}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=FinalSICAssignment)  # type: ignore
        try:
            validated_answer = parser.parse(str(response.content))
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}", error=str(parse_error)
            )
            logger.warning(
                "Failed to parse response", response_content=str(response.content)
            )

            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")

            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = FinalSICAssignment(
                    codable=False,
                    unambiguous_code="N/A",
                    unambiguous_code_descriptive="N/A",
                    higher_level_code="N/A",
                    reasoning=reasoning,
                )

        return validated_answer, call_dict

    async def formulate_open_question(
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        llm_output: Optional[SicCandidate] = None,
        correlation_id: Optional[str] = None,
    ) -> tuple[OpenFollowUp, Any]:
        """Formulates an open-ended question using respondent data and survey design guidelines.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            llm_output (SicCandidate, optional): The response from the LLM model.
            correlation_id (str, optional): Optional correlation ID for request tracking.

        Returns:
            OpenFollowUp: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """

        def prep_call_dict(industry_descr, job_title, job_description, llm_output):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "llm_output": str(llm_output),
            }
            return call_dict

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            llm_output=llm_output,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_openfollowup.format(**call_dict)
            logger.debug(final_prompt)

        chain = self.sic_prompt_openfollowup | self.llm

        # Log LLM request sent
        logger.info(
            "LLM request sent - formulate_open_question",
            job_title=truncate_identifier(job_title),
            job_description=truncate_identifier(job_description),
            industry_descr=truncate_identifier(industry_descr),
            correlation_id=correlation_id or "",
        )
        llm_start = time.perf_counter()

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except (ValueError, AttributeError) as err:
            logger.error(
                f"Error from LLMChain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Error from LLMChain, exit early",
                correlation_id=correlation_id or "",
            )
            validated_answer = OpenFollowUp(
                followup=None,
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, call_dict

        llm_duration_ms = int((time.perf_counter() - llm_start) * 1000)

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=OpenFollowUp)
        try:
            validated_answer = parser.parse(str(response.content))
            # Log LLM response received after successful parse
            has_followup = bool(getattr(validated_answer, "followup", None))
            logger.info(
                "LLM response received for open question prompt",
                has_followup=str(has_followup),
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
        except (ValueError, AttributeError) as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                correlation_id=correlation_id or "",
            )
            logger.info(
                "LLM response received for open question prompt",
                has_followup="False",
                duration_ms=str(llm_duration_ms),
                correlation_id=correlation_id or "",
            )
            try:
                chain = FIX_PARSING_PROMPT | self.llm
                response = await chain.ainvoke(
                    {
                        "llm_output": str(response.content),
                        "format_instructions": parser.get_format_instructions(),
                    },
                    return_only_outputs=True,
                )
                validated_answer = parser.parse(str(response.content))
                logger.debug("Successfully parsed reformatted response.")

            except (ValueError, AttributeError) as parse_error2:
                logger.error(
                    f"Failed to parse response again: {parse_error2}",
                    error=str(parse_error2),
                )
                logger.warning(
                    "Failed to parse response again",
                    response_content=str(response.content),
                )
                reasoning = (
                    f"ERROR parse_error=<{parse_error2}>, response=<{response.content}>"
                )
                validated_answer = OpenFollowUp(
                    followup=None,
                    reasoning=reasoning,
                )

        if self.verbose:
            logger.debug(f"{response=}")

        return validated_answer, call_dict

    async def formulate_closed_question(
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        llm_output: Optional[UnambiguousResponse] = None,
        correlation_id: Optional[str] = None,
    ) -> tuple[ClosedFollowUp, Any]:
        """Formulates a closed follow-up question using respondent data
            and survey design guidelines.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            llm_output (UnambiguousResponse, optional): The response from the LLM model.
            correlation_id (str, optional): Optional correlation ID for request tracking.

        Returns:
            ClosedFollowUp: The generated response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """

        def prep_call_dict(industry_descr, job_title, job_description, llm_output):
            # Helper function to prepare the call dictionary
            is_job_title_present = job_title is None or job_title in {"", " "}
            job_title = "Unknown" if is_job_title_present else job_title

            is_job_description_present = job_description is None or job_description in {
                "",
                " ",
            }
            job_description = (
                "Unknown" if is_job_description_present else job_description
            )

            call_dict = {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
                "llm_output": str(llm_output),
            }
            return call_dict

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            llm_output=llm_output,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_closedfollowup.format(**call_dict)
            logger.debug(f"Final prompt: {final_prompt}")

        chain = self.sic_prompt_closedfollowup | self.llm

        try:
            response = await chain.ainvoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.error(
                f"Error from LLMChain, exit early: {err}",
                error=str(err),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Error from LLMChain, exit early",
                correlation_id=correlation_id or "",
            )
            validated_answer = ClosedFollowUp(
                followup=None,
                sic_options=[],
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug(f"{response=}")

        # Parse the output to the desired format
        parser = PydanticOutputParser(pydantic_object=ClosedFollowUp)
        try:
            validated_answer = parser.parse(str(response.content))
        except ValueError as parse_error:
            logger.error(
                f"Failed to parse response: {parse_error}",
                error=str(parse_error),
                correlation_id=correlation_id or "",
            )
            logger.warning(
                "Failed to parse response",
                response_content=str(response.content),
                correlation_id=correlation_id or "",
            )

            reasoning = (
                f"ERROR parse_error=<{parse_error}>, response=<{response.content}>"
            )
            validated_answer = ClosedFollowUp(
                followup=None,
                sic_options=[],
                reasoning=reasoning,
            )

        return validated_answer, call_dict
