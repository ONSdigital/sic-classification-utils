"""Module for generating prompt templates for SIC classification tasks.

This module provides various prompt templates for tasks related to the classification
of respondent data into UK SIC (Standard Industry Classification) codes. The prompts
are designed to work with the LangChain library and include configurations for
different use cases, such as determining SIC codes, re-ranking SIC codes, and handling
ambiguous classifications.

The module includes:
- Core prompt templates for SIC classification tasks.
- Support for partial variables and format instructions.
- Integration with Pydantic models for structured output parsing.

Attributes:
    SIC_PROMPT_PYDANTIC (PromptTemplate): Template for determining SIC codes based on
        respondent data.
    SIC_PROMPT_RAG (PromptTemplate): Template for determining SIC codes with a relevant
        subset of SIC codes provided.
    SA_SIC_PROMPT_RAG (PromptTemplate): Template for determining a list of most likely
        SIC codes with confidence scores.
    GENERAL_PROMPT_RAG (PromptTemplate): Template for determining custom classification
        codes with a relevant subset of codes provided.
    SIC_PROMPT_UNAMBIGUOUS (PromptTemplate): Template for evaluating if a 5-digit SIC
        code can be assigned with high confidence.
    SIC_PROMPT_RERANKER (PromptTemplate): Template for re-ranking and selecting the most
        relevant SIC codes based on semantic similarity and business context alignment.
"""

# pylint: disable=invalid-name # Need to clean up the code to remove this

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from industrial_classification_utils.embed.embedding import get_config
from industrial_classification_utils.models.response_model import (
    FinalSICAssignment,
    RerankingResponse,
    SicResponse,
    UnambiguousResponse,
)
from industrial_classification_utils.utils.sic_data_access import (
    load_text_from_config,
)

config = get_config()

_core_prompt = """You are a conscientious classification assistant of respondent data
for the use in the UK official statistics. Respondent data may be in English or Welsh,
but you always respond in British English."""

_sic_template = """"Given the respondent's description of the main activity their
company does, their job title and job description, your task is to determine
the UK SIC (Standard Industry Classification) code for this company if it can be
determined to the division (two-digit) level. If the code cannot be determined,
identify the additional information needed to determine it.
Make sure to use the provided 2007 SIC Index.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Output Format===
{format_instructions}

===2007 SIC Index===
{sic_index}
"""

# Load the SIC index from the configuration and convert to file path string
sic_index = load_text_from_config(config["lookups"]["sic_condensed"])

parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=SicResponse
)

SIC_PROMPT_PYDANTIC = PromptTemplate.from_template(
    template=_core_prompt + _sic_template,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "sic_index": sic_index,
    },
)


_sic_template_rag = """"Given the respondent's description of the main activity their
company does, their job title and job description (which may be different then the
main company activity), your task is to determine the UK SIC (Standard Industry
Classification) code for this company if it can be determined.
Make sure to use the provided Relevant subset of UK SIC 2007. If the code cannot be
determined (or is likely not included in the provided subset), identify the additional
information needed to determine it and a list of most likely codes.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Relevant subset of UK SIC 2007===
{sic_index}

===Output Format===
{format_instructions}

===Output===
"""

_sic_template_confidence_rag = """"Given the respondent's description of the main
activity their company does, their job title and job description (which may be
different to the main company activity), your task is to determine the UK SIC
(Standard Industry Classification) code for this company if it can be determined.

The following will be provided to make your decision and send appropriate output:
Respondent Data
Relevant subset of UK SIC 2007 (you must only use this list to classify)
Output Format

You must only use the Relevant subset of UK SIC 2007 provided to determine if you
can match a sic code.
Where the data shows ambuguity, (e.g I teach children, could be classified as
secondary or primary school teacher), you must return the list of possible sic codes
that might match with a confidence score for each.

If the code cannot be determined (or is likely not included in the provided subset),
identify the additional information needed to determine it and a list of most likely
codes.


===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Relevant subset of UK SIC 2007===
{sic_index}

===Output Format===
{format_instructions}

===Output===
"""

# Was sic_template_rag
SIC_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _sic_template_confidence_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)


_sa_sic_template_rag = """"Given the respondent's description of the main activity their
company does, their job title and job description (which may be different to the
main company activity), your task is to determine a list of the most likely UK SIC
(Standard Industry Classification) codes for this company.

The following will be provided to make your decision and send appropriate output:
Respondent Data
Relevant subset of UK SIC 2007 (you must only use this list to classify)
Output Format (the output format MUST be valid JSON)

Only use the subset of UK SIC 2007 provided to determine if you can match the most
likely sic codes, provide a confidence score between 0 and 1 where 0.1 is least
likely and 0.9 is most likely.

You must return the a subset list of possible sic codes (UK SIC 2007 codes provided)
that might match with a confidence score for each.

You must provide a follow up question that would help identify the exact coding based
on the list you respond with.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Relevant subset of UK SIC 2007===
{sic_index}

===Output Format===
{format_instructions}

===Output===
"""

parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=SicResponse
)

SA_SIC_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _sa_sic_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)


_general_template_rag = """"Given the respondent's data, your task is to determine
the classification code. Make sure to use the provided Relevant subset of
classification index and select codes from this list only.
If the code cannot be determined (or not included in the provided subset),
do not provide final code, instead identify the additional information needed
to determine the correct code and suggest few most likely codes.

===Respondent Data===
{respondent_data}

===Relevant subset of classification index===
{classification_index}

===Output Format===
{format_instructions}

===Output===
"""
parser = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=SicResponse
)

GENERAL_PROMPT_RAG = PromptTemplate.from_template(
    template=_core_prompt + _general_template_rag,
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

_sic_template_unambiguous = """"Given:
1. Respondent data (job_title, job_description, industry_descr)
2. Shortlist of UK Standard Industrial Classification (SIC) codes

Your task is to evaluate whether response can be assigned to a single 5-digit SIC code.

===Steps to take===
Approach the task in the following order:
1. Review the shortlist and evaluate each candidate SIC code.
2. Assess the relevance of each candidate SIC code to survey respondent based on semantic similarity and business context alignment. Specifically, this includes:
    a) fundamental alignment between the query and the code's main business activity
    b) matches between query and specific example activities listed under the code
3. For each SIC code candidate provide a confidence score between 0 and 1 where 0.1 is least likely and 0.9 is most likely.
4. Decide if response can be codeded unambiguously to a single 5-digit SIC code with 95 per cent confidence.
5. Provide reasoning for your decision.

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Shortlist===
{sic_candidates}

===Output Format===
{format_instructions}
"""
parser_unambiguous = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=UnambiguousResponse
)

SIC_PROMPT_UNAMBIGUOUS = PromptTemplate.from_template(
    template=_core_prompt + _sic_template_unambiguous,
    partial_variables={
        "format_instructions": parser_unambiguous.get_format_instructions(),
    },
)

_sic_template_reranker = """"You are a precise semantic matching system.
Your task is to re-rank and select the N most relevant UK SIC (Standard Industry
Classification) codes from a provided list of candidates based on their relevance
to the respondent's description.

===Task Description===

Analyze each candidate SIC code's relevance to the query.
Score each candidate on a scale of 0.0 to 1.0 based on semantic similarity and
business context alignment.
Select the top N most relevant codes.
Provide clear reasoning for your scoring decisions.
Your response must be a single JSON object with NO additional text or formatting.

===Scoring Criteria===

Primary Activity Match (0.0-0.4):

Evaluates the fundamental alignment between the query and the code's main
business activity

Scoring guidelines:

0.35-0.4: Perfect match (e.g., "Beer brewery" → "Manufacture of beer")
0.25-0.34: Strong match with minor differences (e.g., "Craft brewery" →
"Manufacture of beer")
0.15-0.24: Related activity in same sector (e.g., "Beer distribution" →
"Manufacture of beer")
0.05-0.14: Tangentially related activity (e.g., "Beer tasting" →
"Manufacture of beer")
0.0-0.04: Minimal or no relation to primary activity


Context Precision (0.0-0.3):

Measures how specifically the code captures the business context of the query
Considers industry position (manufacturing, wholesale, retail, service)
Scoring guidelines:

0.25-0.3: Exact business context match (e.g., manufacturing vs. retail context)
0.15-0.24: Related context with same business model
0.05-0.14: Similar industry but different business model
0.0-0.04: Different business context entirely


Examples:

Query "Beer shop" matching "Retail sale of beverages" (high precision)
Query "Beer shop" matching "Wholesale of beverages" (medium precision)
Query "Beer shop" matching "Manufacture of beer" (low precision)


Example Activity Alignment (0.0-0.3):

Evaluates matches between query and specific example activities listed under the code
Considers both exact matches and semantic similarity

Scoring guidelines:

0.25-0.3: Direct match with example activities
0.15-0.24: Semantic equivalence to example activities
0.05-0.14: Partial overlap with example activities
0.0-0.04: No matching example activities


Special considerations:

Multiple matching examples increase score within range
Industry-specific terminology matches are weighted heavily
Generic matches receive lower scores

===Input Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}
- Number of codes to select (N): {n}

===UK SIC 2007 candidates===
{sic_index}

===Requirements===

Scores must be between 0.0 and 1.0
Selected codes must be exactly N in number
All codes must receive a score and reasoning
Reasoning must reference specific aspects of the code and query
Scores should reflect relative relevance between codes

===Output Format===
{format_instructions}

"""
parser_reranker = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=RerankingResponse
)


SIC_PROMPT_RERANKER = PromptTemplate.from_template(
    template=_core_prompt + _sic_template_reranker,
    partial_variables={
        "format_instructions": parser_reranker.get_format_instructions(),
    },
)


_sic_template_final_assignment = """"You are an expert in industrial classifications.
You are tasked with assigning UK Standard Industrial Classification (SIC) codes to survey
responses with high confidence.

Key objective: You MUST assign a 5-digit SIC code from the candidates provided. Only provide a higher-level
code if multiple candidates have nearly identical confidence scores (within 0.2 of each other) AND no single
can be identified as the clear best match.

Assignment logic:
1. Default behavior: Assign the highest-confidence 5-digit SIC code from the candidates
2. Higher-level code exception: Only if two or more codes have confidence scores within 0.2
 of each other AND you cannot determine a clear winner. Provide the most granular
higher-level code with X padding to 5-digits (e.g., 8610X for 4-digit confidence, 86XXX for
3-digit confidence, 8XXXX for 2-digit confidence).
3. 95% confidence interpretation: This means "more likely than not" given the available evidence -
not absolute certainty

Key principles:
1. Focus on Best Fit: Rather than seeking absolute certainty, identify which code best fits the totality of evidence.
2. Be Decisive: The goal is accurate classification, not perfect certainty. If evidence clearly points to one
code over others, assign it confidently.

Important: When a respondent's closed question answer directly matches or closely aligns with a SIC code
description, this constitutes strong evidence for that code.

Follow these steps in order:
1. Review all available information - respondent data, candidate SIC codes, and follow-up responses
2. Evaluate each candidate SIC code against all available evidence
3. Assign confidence scores - Rate each candidate from 0.1 (least likely) to 0.9 (most likely).
Weight respondent's own descriptions heavily.
4. Apply assignment logic - Select the candidate with the highest confidence score as your primary assignment.
Only consider higher-level coding if multiple candidates have nearly identical scores (within 0.2) and you cannot
differentiate between them
5. Determine final assignment - Assign best fitting 5-digit code or the most specific higher-level code
6. Provide clear reasoning - Explain your decision with specific evidence

===Respondent Data===
- Company's main activity: {industry_descr}
- Job Title: {job_title}
- Job Description: {job_description}

===Short list of UK SIC codes===
{sic_candidates}

===Follow up question 1===
{open_question}
{answer_to_open_question}

===Follow up question 2===
{closed_question}
{answer_to_closed_question}

===Output Format===
{format_instructions}
"""
parser_final_assignment = PydanticOutputParser(  # type: ignore # Suspect langchain ver bug
    pydantic_object=FinalSICAssignment
)

SIC_PROMPT_FINAL_ASSIGNMENT = PromptTemplate.from_template(
    template=_core_prompt + _sic_template_final_assignment,
    partial_variables={
        "format_instructions": parser_final_assignment.get_format_instructions(),
    },
)


# pylint: disable=too-few-public-methods
class PromptTemplates:
    """A collection of predefined prompt templates for various use cases.

    Attributes:
        SIC_PROMPT_PYDANTIC (PromptTemplate): A prompt template for SIC using Pydantic.
        SIC_PROMPT_RAG (PromptTemplate): A prompt template for SIC with RAG
                                            (Retrieval-Augmented Generation).
        SA_SIC_PROMPT_RAG (PromptTemplate): A prompt template for SA SIC with RAG.
        GENERAL_PROMPT_RAG (PromptTemplate): A general-purpose prompt template with RAG.
        SIC_PROMPT_UNAMBIGUOUS (PromptTemplate): A prompt template for unambiguous
                                                    SIC classification.
        SIC_PROMPT_RERANKER (PromptTemplate): A prompt template for SIC reranking.

    Methods:
        get_all_templates() -> list[PromptTemplate]:
            Returns all stored prompt templates as a list.
    """

    def __init__(self):
        self.SIC_PROMPT_PYDANTIC = SIC_PROMPT_PYDANTIC
        self.SIC_PROMPT_RAG = SIC_PROMPT_RAG
        self.SA_SIC_PROMPT_RAG = SA_SIC_PROMPT_RAG
        self.GENERAL_PROMPT_RAG = GENERAL_PROMPT_RAG
        self.SIC_PROMPT_UNAMBIGUOUS = SIC_PROMPT_UNAMBIGUOUS
        self.SIC_PROMPT_RERANKER = SIC_PROMPT_RERANKER

    def get_all_templates(self) -> list[PromptTemplate]:
        """Returns all stored prompt templates as a list."""
        return [
            self.SIC_PROMPT_PYDANTIC,
            self.SIC_PROMPT_RAG,
            self.SA_SIC_PROMPT_RAG,
            self.GENERAL_PROMPT_RAG,
            self.SIC_PROMPT_UNAMBIGUOUS,
            self.SIC_PROMPT_RERANKER,
        ]
