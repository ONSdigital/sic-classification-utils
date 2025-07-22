"""
Tests for industrial_classification_utils.llm.llm.py
"""
import pytest
from industrial_classification.hierarchy.sic_hierarchy import SIC, SicCode, SicNode
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from industrial_classification_utils.llm.llm import ClassificationLLM
from industrial_classification_utils.models.response_model import (  # RerankingResponse,; SurveyAssistSicResponse,; UnambiguousResponse,
    SicResponse,
)


@pytest.mark.parametrize(
    "model, openai_api_key, expected_model",
    [
        ("gemini", None, ChatVertexAI),
        ("text-", None, ChatVertexAI),
        ("gpt", "key", ChatOpenAI),
    ],
)
@pytest.mark.utils
def test_llm_model(model, openai_api_key, expected_model):
    llm_model_type = ClassificationLLM(
        model_name=model, openai_api_key=openai_api_key
    ).llm
    assert isinstance(llm_model_type, expected_model)


@pytest.mark.utils
def test_llm_model_default():
    assert isinstance(ClassificationLLM().llm, ChatVertexAI)


@pytest.mark.utils
def test_model_name():
    assert ClassificationLLM().llm.model_name == "gemini-1.0-pro"


@pytest.mark.utils
def test_sic_get_code_initialise():
    llm_sic_code = ClassificationLLM(model_name="gemini-1.5-flash").get_sic_code(
        industry_descr="", job_description="", job_title=""
    )
    assert isinstance(llm_sic_code, SicResponse)


# pylint: disable=R0801, W0621
@pytest.fixture
def prompt_candidate_sic():
    nodes = [
        SicNode(sic_code=SicCode("A0111x"), description="Bird watching"),
        SicNode(sic_code=SicCode("A0112x"), description="Petting animals"),
    ]
    lookup = {}
    for node in nodes:
        lookup[str(node.sic_code)] = node
        lookup[node.sic_code.alpha_code] = node
        lookup[node.sic_code.alpha_code.replace("x", "")] = node
        if node.sic_code.n_digits > 1:
            lookup[node.sic_code.alpha_code[1:].replace("x", "")] = node

        if node.sic_code.n_digits == 4 and not node.children:  # noqa: PLR2004
            key = node.sic_code.alpha_code[1:5] + "0"
            lookup[key] = node
    sic = SIC(nodes=nodes, code_lookup=lookup)
    llm_class = ClassificationLLM(model_name="gemini-1.5-flash")
    llm_class.sic = sic
    print(llm_class.sic.all_leaf_text())
    return llm_class


@pytest.mark.parametrize(
    "code, activities, expected_output_code, expected_output_activities",
    [
        ("01110", ["observing"], "01110", "observing"),
        ("01120", ["giving belly rubs"], "01120", "giving belly rubs"),
    ],
)
@pytest.mark.utils
def test_prompt_candidate(
    prompt_candidate_sic,
    code,
    activities,
    expected_output_code,
    expected_output_activities,
):
    result = prompt_candidate_sic._prompt_candidate(code=code, activities=activities)
    assert all(x in result for x in [expected_output_code, expected_output_activities])