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

import logging
from collections import defaultdict
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
from industrial_classification.hierarchy.sic_hierarchy import load_hierarchy
from industrial_classification.meta import sic_meta
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_google_vertexai import VertexAI
from langchain_openai import ChatOpenAI

from industrial_classification_utils.embed.embedding import EmbeddingHandler, get_config
from industrial_classification_utils.llm.prompt import (
    GENERAL_PROMPT_RAG,
    SA_SIC_PROMPT_RAG,
    SIC_PROMPT_PYDANTIC,
    SIC_PROMPT_RAG,
    SIC_PROMPT_RERANKER,
    SIC_PROMPT_UNAMBIGUOUS,
)
from industrial_classification_utils.models.response_model import (
    RagResponse,
    RerankingResponse,
    SicResponse,
    SurveyAssistSicResponse,
    UnambiguousResponse,
)
from industrial_classification_utils.utils.sic_data_access import (
    load_sic_index,
    load_sic_structure,
)

logger = logging.getLogger(__name__)
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
        embedding_handler (EmbeddingHandler): Embedding handler. Optional.
            If None a default embedding handler is retrieved based on config file.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 1600.
        temperature (float): Temperature of the LLM model. Defaults to 0.0.
        verbose (bool): Whether to print verbose output. Defaults to False.
        openai_api_key (str): OpenAI API key. Optional, but needed for OpenAI models.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = config["llm"]["llm_model_name"],
        llm: Optional[Union[VertexAI, ChatOpenAI]] = None,
        embedding_handler: Optional[EmbeddingHandler] = None,
        max_tokens: int = 1600,
        temperature: float = 0.0,
        verbose: bool = True,
        openai_api_key: Optional[str] = None,
    ):
        """Initialises the ClassificationLLM object."""
        print(f"model_name: {model_name}")
        if llm is not None:
            self.llm = llm
        elif model_name.startswith("text-") or model_name.startswith("gemini"):
            self.llm = VertexAI(
                model_name=model_name,
                max_output_tokens=max_tokens,
                temperature=temperature,
                location="europe-west2",
            )
        elif model_name.startswith("gpt"):
            if openai_api_key is None:
                raise NotImplementedError("Need to provide an OpenAI API key")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
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
        self.embed = embedding_handler
        self.sic = None
        self.verbose = verbose

    def _load_embedding_handler(self):
        """Loads the default embedding handler according to the 'config' file.
        Expects an existing and populated persistent vector store.

        Raises:
            ValueError: If the retrieved embedding handler has an empty vector store.
                Please embed an index before using it in the ClassificationLLM.
        """
        logger.info(
            """Loading default embedding handler according to 'config' file.
            Expecting existing & populated persistent vector store."""
        )
        self.embed = EmbeddingHandler()
        if self.embed._index_size == 0:  # pylint: disable=protected-access
            raise ValueError(
                """The retrieved embedding handler has an empty vector store.
                Please embed an index before using in the ClassificationLLM."""
            )

    @lru_cache  # noqa: B019
    def get_sic_code(
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
        chain = LLMChain(llm=self.llm, prompt=self.sic_prompt)
        response = chain.invoke(
            {
                "industry_descr": industry_descr,
                "job_title": job_title,
                "job_description": job_description,
            },
            return_only_outputs=True,
        )
        if self.verbose:
            logger.debug("%s", response)
        # Parse the output to desired format with one retry
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SicResponse
        )
        try:
            validated_answer = parser.parse(response["text"])
        except ValueError as parse_error:
            logger.debug(
                "Retrying llm response parsing due to an error: %s", parse_error
            )
            logger.error("Unable to parse llm response: %s", parse_error)

            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
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
            "Chars Lmt: %d Candidate Lmt: %d Activities Lmt: %d Short List Len: %d Code Digits: %d",
            chars_limit,
            candidates_limit,
            activities_limit,
            len(short_list),
            code_digits,
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
                    "Shortening list of candidates to fit token limit from %d to %d",
                    len(sic_candidates),
                    nn,
                )
                sic_candidates = sic_candidates[:nn]

        return "\n".join(sic_candidates)

    def rag_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        expand_search_terms: bool = True,
        code_digits: int = 5,
        candidates_limit: int = 5,
    ) -> tuple[SicResponse, Optional[list[dict[Any, Any]]], Optional[Any]]:
        """Generates a SIC classification based on respondent's data using RAG approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            expand_search_terms (bool, optional): Whether to expand the search terms
                to include job title and description. Defaults to True.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 5.

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

        if self.embed is None:
            try:
                self._load_embedding_handler()
            except ValueError as err:
                logger.exception(err)
                logger.warning("Error: Empty embedding vector store, exit early")
                validated_answer = SicResponse(
                    codable=False,
                    sic_candidates=[],
                    reasoning="Error, Empty embedding vector store, exit early",
                )
                return validated_answer, None, None

        # Retrieve relevant SIC codes and format them for prompt
        if expand_search_terms:
            short_list = self.embed.search_index_multi(  # type: ignore # False positive
                query=[industry_descr or "", job_title or "", job_description or ""]
            )
        else:
            short_list = self.embed.search_index(  # type: ignore # False positive
                query=industry_descr
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
            final_prompt = self.sic_prompt_rag.format(**call_dict)
            logger.debug("%s", final_prompt)

        chain = LLMChain(llm=self.llm, prompt=self.sic_prompt_rag)

        try:
            response = chain.invoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.exception(err)
            logger.warning("Error from LLMChain, exit early")
            validated_answer = SicResponse(
                codable=False,
                sic_candidates=[],
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug("%s", response)

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SicResponse
        )
        try:
            validated_answer = parser.parse(response["text"])
        except ValueError as parse_error:
            logger.exception(parse_error)
            logger.warning("Failed to parse response:\n%s", response["text"])

            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
            )
            validated_answer = SicResponse(
                codable=False,
                sic_candidates=[],
                reasoning=reasoning,
            )

        return validated_answer, short_list, call_dict

    def sa_rag_sic_code(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        expand_search_terms: bool = True,
        code_digits: int = 5,
        candidates_limit: int = 5,
    ) -> tuple[SurveyAssistSicResponse, Optional[list[dict[Any, Any]]], Optional[Any]]:
        """Generates a SIC classification based on respondent's data using RAG approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            expand_search_terms (bool, optional): Whether to expand the search terms
                to include job title and description. Defaults to True.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 5.

        Returns:
            SurveyAssistSicResponse: The generated response to the query.

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

        if self.embed is None:
            try:
                self._load_embedding_handler()
            except ValueError as err:
                logger.exception(err)
                logger.warning("Error: Empty embedding vector store, exit early")
                validated_answer = SurveyAssistSicResponse(
                    followup="Follow-up question not available due to error.",
                    sic_code="N/A",
                    sic_descriptive="N/A",
                    sic_candidates=[],
                    reasoning="Error, Empty embedding vector store, exit early",
                )
                return validated_answer, None, None

        # Retrieve relevant SIC codes and format them for prompt
        if expand_search_terms:
            short_list = self.embed.search_index_multi(  # type: ignore # False positive
                query=[industry_descr or "", job_title or "", job_description or ""]
            )
        else:
            short_list = self.embed.search_index(  # type: ignore # False positive
                query=industry_descr
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
            logger.debug("%s", final_prompt)

        chain = LLMChain(llm=self.llm, prompt=self.sa_sic_prompt_rag)

        try:
            response = chain.invoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.exception(err)
            logger.warning("Error from LLMChain, exit early")
            validated_answer = SurveyAssistSicResponse(
                followup="Follow-up question not available due to error.",
                sic_code="N/A",
                sic_descriptive="N/A",
                sic_candidates=[],
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug("%s", response)

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=SurveyAssistSicResponse
        )
        try:
            validated_answer = parser.parse(response["text"])
        except ValueError as parse_error:
            logger.exception(parse_error)
            logger.warning("Failed to parse response:\n%s", response["text"])

            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
            )
            validated_answer = SurveyAssistSicResponse(
                followup="Follow-up question not available due to error.",
                sic_code="N/A",
                sic_descriptive="N/A",
                sic_candidates=[],
                reasoning=reasoning,
            )

        return validated_answer, short_list, call_dict

    def rag_general_code(
        self,
        respondent_data: dict,
        candidates_limit: int = 7,
    ) -> tuple[RagResponse, Optional[Any]]:
        """Generates a classification answer based on respondent's data
        using RAG and custom index.

        Args:
            respondent_data (dict): A dictionary containing respondent data.
            candidates_limit (int, optional): The maximum number of candidate
                codes to consider. Defaults to 7.

        Returns:
            RagResponse: The generated classification response to the query.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.
        """
        if self.embed is None:
            try:
                self._load_embedding_handler()
            except ValueError as err:
                logger.exception(err)
                logger.warning("Error: Empty embedding vector store, exit early")
                validated_answer = RagResponse(
                    codable=False,
                    alt_candidates=[],
                    reasoning="Error: Empty embedding vector store, exit early",
                )
                return validated_answer, None

        # Retrieve relevant SIC codes and format them for prompt
        query = (
            [str(value) for value in respondent_data.values()]
            if respondent_data
            else [""]
        )
        short_list = self.embed.search_index_multi(query=query)  # type: ignore # False positive

        candidate_codes = (
            "{"
            + "}, /n{".join(
                [
                    "Code: " + x["code"] + ", Description: " + x["title"]
                    for x in short_list[:candidates_limit]
                ]
            )
            + "}"
        )

        if self.verbose:
            final_prompt = self.general_prompt_rag.format(
                respondent_data=str(respondent_data),
                classification_index=candidate_codes,
            )
            logger.debug("%s", final_prompt)

        chain = LLMChain(llm=self.llm, prompt=self.general_prompt_rag)

        try:
            response = chain.invoke(
                {
                    "respondent_data": str(respondent_data),
                    "classification_index": candidate_codes,
                },
                return_only_outputs=True,
            )
        except ValueError as err:
            logger.exception(err)
            logger.warning("Error from LLMChain, exit early")
            validated_answer = RagResponse(
                codable=False,
                alt_candidates=[],
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, short_list

        if self.verbose:
            logger.debug("llm_response=%s", response)

        # Parse the output to desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=RagResponse
        )
        try:
            validated_answer = parser.parse(response["text"])
        except ValueError as parse_error:
            logger.exception(parse_error)
            logger.warning("Failed to parse response:\n%s", response["text"])

            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
            )
            validated_answer = RagResponse(
                codable=False,
                alt_candidates=[],
                reasoning=reasoning,
            )

        return validated_answer, short_list

    def unambiguous_sic_code(
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        reranker_response: Optional[RerankingResponse] = None,
    ) -> tuple[UnambiguousResponse, dict[str, Any]]:
        """Evaluates codability to a 5-digit SIC code based on respondent's
            data and reranker output.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description.
                Defaults to None.
            reranker_response (RerankingResponse, optional): The response
                from the reranker.

        Returns:
            Tuple[UnambiguousResponse, Dict[str, Any]]: The generated response
                to the query and the call dictionary.

        Raises:
            ValueError: If there is an error during the parsing of the response.
            ValueError: If the default embedding handler is required but
                not loaded correctly.

        """

        def prep_call_dict(
            industry_descr, job_title, job_description, reranker_response
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
                "reranker_response": str(reranker_response),
            }
            return call_dict

        call_dict = prep_call_dict(
            industry_descr=industry_descr,
            job_title=job_title,
            job_description=job_description,
            reranker_response=reranker_response,
        )

        if self.verbose:
            final_prompt = self.sic_prompt_unambiguous.format(**call_dict)
            logger.debug("%s", final_prompt)

        chain = LLMChain(llm=self.llm, prompt=self.sic_prompt_unambiguous)

        try:
            response = chain.invoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.exception(err)
            logger.warning("Error from LLMChain, exit early")
            validated_answer = UnambiguousResponse(
                codable=False,
                alt_candidates=[],
                reasoning="Error from LLMChain, exit early",
            )
            return validated_answer, call_dict

        if self.verbose:
            logger.debug("%s", response)

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=UnambiguousResponse
        )
        try:
            validated_answer = parser.parse(response["text"])
        except ValueError as parse_error:
            logger.exception(parse_error)
            logger.warning("Failed to parse response:\n%s", response["text"])

            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
            )
            validated_answer = UnambiguousResponse(
                codable=False,
                alt_candidates=[],
                reasoning=reasoning,
            )

        return validated_answer, call_dict

    def reranker_sic(  # noqa: PLR0913
        self,
        industry_descr: str,
        job_title: Optional[str] = None,
        job_description: Optional[str] = None,
        expand_search_terms: bool = True,
        code_digits: int = 5,
        candidates_limit: int = 7,
        output_limit: int = 5,
    ) -> Union[tuple[Any, Optional[list], Optional[dict[str, Any]]], dict[str, Any]]:
        """Generates a set of relevant SIC codes based on respondent's data
            using reranking approach.

        Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            expand_search_terms (bool, optional): Whether to expand the search terms
                to include job title and description. Defaults to True.
            code_digits (int, optional): The number of digits in the generated
                SIC code. Defaults to 5.
            candidates_limit (int, optional): The maximum number of SIC code candidates
                to consider. Defaults to 7.
            output_limit (int, optional): The maximum number of SIC codes to return.
                Defaults to 5.

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

        if self.embed is None:
            try:
                self._load_embedding_handler()
            except ValueError as err:
                logger.exception(err)
                logger.warning("Error: Empty embedding vector store, exit early")
                validated_answer = RerankingResponse(
                    selected_codes=[],
                    excluded_codes=[],
                    status="Error, Empty embedding vector store, exit early",
                    n_requested=output_limit,
                )
                return validated_answer, None, None

        # Retrieve relevant SIC codes and format them for prompt
        if expand_search_terms:
            short_list = self.embed.search_index_multi(  # type: ignore # False positive
                query=[industry_descr or "", job_title or "", job_description or ""]
            )
        else:
            short_list = self.embed.search_index(  # type: ignore # False positive
                query=industry_descr
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
            logger.debug("%s", final_prompt)

        chain = LLMChain(llm=self.llm, prompt=self.sic_prompt_reranker)

        try:
            response = chain.invoke(call_dict, return_only_outputs=True)
        except ValueError as err:
            logger.exception(err)
            logger.warning("Error from LLMChain, exit early")
            validated_answer = RerankingResponse(
                selected_codes=[],
                excluded_codes=[],
                status="Error from LLMChain, exit early",
                n_requested=output_limit,
            )
            return validated_answer, short_list, call_dict

        if self.verbose:
            logger.debug("%s", response)

        # Parse the output to the desired format
        parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
            pydantic_object=RerankingResponse
        )
        try:
            validated_answer = parser.parse(response["text"])
        except ValueError as parse_error:
            logger.exception(parse_error)
            logger.warning("Failed to parse response:\n%s", response["text"])

            reasoning = (
                f'ERROR parse_error=<{parse_error}>, response=<{response["text"]}>'
            )
            validated_answer = RerankingResponse(
                selected_codes=[],
                excluded_codes=[],
                status=reasoning,
                n_requested=output_limit,
            )

        return validated_answer, short_list, call_dict
