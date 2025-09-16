# pylint: disable=C0116
"""Tests for industrial_classification_utils.llm.llm.py."""

from unittest import mock

import pytest
import vertexai
from industrial_classification.hierarchy.sic_hierarchy import SIC, SicCode, SicNode
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from industrial_classification_utils.llm.llm import ClassificationLLM
from industrial_classification_utils.models.response_model import SicResponse
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage
import json

MODEL_NAME = "gemini-1.5-flash"
LOCATION = "europe-west2"


def test_setup():
    vertexai.init(project="classifai-sandbox", location=LOCATION)


@pytest.fixture(autouse=True)
def mock_vertex_ai():
    with mock.patch(
        "google.cloud.aiplatform.gapic.PredictionServiceClient"
    ) as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.generate_content.return_value = mock.Mock()
        yield


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
def test_pass_llm_argument():
    llm_model = ClassificationLLM(llm = "model").llm
    assert llm_model == "model"


@pytest.mark.utils
def test_llm_model_default():
    assert isinstance(ClassificationLLM().llm, ChatVertexAI)


@pytest.mark.utils
def test_model_name():
    assert ClassificationLLM().llm.model_name == "gemini-1.0-pro"

@pytest.mark.utils
def test_llm_response_mocked_get_sic_code(mocker):
    mock_object_str = {
        "codable": True,
        "followup": "This is follow-up",
        "sic_code": "12345",
        "sic_descriptive": "description12345",
        "sic_candidates": [
            {
            "sic_code": "23456",
            "sic_descriptive": "description23456",
            "likelihood": 0.5
            },
            {
            "sic_code": "34567",
            "sic_descriptive": "description34567",
            "likelihood": 0.5
            }
        ],
        "reasoning": "reasoning12345"
        }
    mock_object_json = json.dumps(mock_object_str)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mock_patcher = mocker.patch("industrial_classification_utils.llm.llm.ChatVertexAI.invoke", return_value = mock_message)

    result = ClassificationLLM(model_name=MODEL_NAME).get_sic_code(
        industry_descr="", job_description="", job_title=""
    )
    assert isinstance(result, SicResponse)
    

@pytest.mark.utils
def test_sic_get_code_initialise():
    llm_sic_code = ClassificationLLM(model_name=MODEL_NAME).get_sic_code(
        industry_descr="", job_description="", job_title=""
    )
    assert isinstance(llm_sic_code, SicResponse)


# pylint: disable=R0801, W0621
@pytest.fixture
def prompt_candidate_sic():
    nodes = [
        SicNode(sic_code=SicCode("A12345"), description="description12345"),
        SicNode(sic_code=SicCode("A23456"), description="description23456"),
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
    llm_class = ClassificationLLM(model_name=MODEL_NAME)
    llm_class.sic = sic
    return llm_class

@pytest.mark.utils
def test_llm_response_mocked_sa_rag_sic_code(mocker, prompt_candidate_sic):
    short_list = [
        {
            "distance": 0.6,
            "title": "title1",
            "code": "12345",
            "four_digit_code": "1234",
            "two_digit_code": "12",
        },
        {
            "distance": 0.7,
            "title": "title2",
            "code": "23456",
            "four_digit_code": "2345",
            "two_digit_code": "23",
        }
    ]
    mock_object_str = {
        "codable": True,
        "followup": "This is follow-up",
        "sic_code": "12345",
        "sic_descriptive": "description12345",
        "sic_candidates": [
            {
            "sic_code": "23456",
            "sic_descriptive": "description23456",
            "likelihood": 0.5
            },
            {
            "sic_code": "34567",
            "sic_descriptive": "description34567",
            "likelihood": 0.5
            }
        ],
        "reasoning": "reasoning12345"
        }
    mock_object_json = json.dumps(mock_object_str)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mock_patcher = mocker.patch("industrial_classification_utils.llm.llm.ChatVertexAI.invoke", return_value = mock_message)

    result = prompt_candidate_sic.sa_rag_sic_code(
        industry_descr="", job_description="", job_title="", short_list = short_list
    )
    result_list = []
    for i in result[0]:
        result_list.append(i[0])
    for i in result[1][0]:
        result_list.append(i)
    for i in result[2]:
        result_list.append(i)
    assert result_list == ['followup', 'sic_code', 'sic_descriptive', 'sic_candidates', 'reasoning', 'distance', 'title', 'code', 'four_digit_code', 'two_digit_code', 'industry_descr','job_title', 'job_description', 'sic_index']


@pytest.mark.parametrize(
    "code, activities, expected_output_code, expected_output_activities",
    [
        ("12345", ["activity1"], "12345", "activity1"),
        ("23456", ["activity2"], "23456", "activity2"),
    ],
)
@pytest.mark.utils
def test_prompt_candidate_output(
    prompt_candidate_sic,
    code,
    activities,
    expected_output_code,
    expected_output_activities,
):
    result = prompt_candidate_sic._prompt_candidate(  # pylint: disable=W0212
        code=code, activities=activities
    )
    assert all(x in result for x in [expected_output_code, expected_output_activities])


@pytest.mark.parametrize(
    "industry, title, job_description, short_list, expected_job_title",
    [
        (
            "school",
            "",
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "Unknown",
        ),
        (
            "school",
            " ",
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "Unknown",
        ),
        (
            "school",
            None,
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "Unknown",
        ),
        (
            "school",
            "teacher",
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "teacher",
        ),
    ],
)
@pytest.mark.utils
def test_sa_rag_sic_code_prep_call_dict_job_title_correct(
    industry, title, job_description, short_list, expected_job_title
):
    result = ClassificationLLM(model_name=MODEL_NAME).sa_rag_sic_code(
        industry, title, job_description, short_list=short_list
    )[2]["job_title"]
    assert result == expected_job_title


@pytest.mark.parametrize(
    "industry, title, job_description, short_list",
    [
        ("school", "teacher", "educate kids", [{"title": "Education", "code": "01"}]),
    ],
)
@pytest.mark.utils
def test_sa_rag_sic_code_prep_followup_is_str(
    industry, title, job_description, short_list
):
    result = (
        ClassificationLLM(model_name=MODEL_NAME)
        .sa_rag_sic_code(industry, title, job_description, short_list=short_list)[0]
        .followup
    )
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "industry, title, job_description, sic_candidates, expected_job_title",
    [
        (
            "school",
            "",
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "Unknown",
        ),
        (
            "school",
            " ",
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "Unknown",
        ),
        (
            "school",
            None,
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "Unknown",
        ),
        (
            "school",
            "teacher",
            "educate kids",
            [{"title": "Education", "code": "01"}],
            "teacher",
        ),
    ],
)
@pytest.mark.utils
def test_unambiguous_sic_code_call_dict_job_title_correct(
    industry, title, job_description, sic_candidates, expected_job_title
):
    result = ClassificationLLM(model_name=MODEL_NAME).unambiguous_sic_code(
        industry, title, job_description, sic_candidates=sic_candidates
    )[1]["job_title"]
    assert result == expected_job_title


@pytest.mark.utils
def test_unambiguous_sic_code_followup_is_str():
    result = (
        ClassificationLLM(model_name=MODEL_NAME)
        .unambiguous_sic_code(
            "school",
            "teacher",
            "educate kids",
            sic_candidates=[{"title": "Education", "code": "01"}],
        )[0]
        .reasoning
    )
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "industry, title, short_list, expected_job_title",
    [
        ("school", None, [{"title": "Education", "code": "01"}], "Unknown"),
        ("school", "", [{"title": "Education", "code": "01"}], "Unknown"),
        ("school", " ", [{"title": "Education", "code": "01"}], "Unknown"),
        ("school", "teacher", [{"title": "Education", "code": "01"}], "teacher"),
    ],
)
@pytest.mark.utils
def test_reranker_sic_call_dict_job_title_correct(
    industry, title, short_list, expected_job_title
):
    result = ClassificationLLM(model_name=MODEL_NAME).reranker_sic(
        industry, title, short_list=short_list
    )[2]["job_title"]
    assert result == expected_job_title


@pytest.mark.parametrize(
    "industry, short_list",
    [
        ("school", [{"title": "Education", "code": "01"}]),
    ],
)
@pytest.mark.utils
def test_reranker_sic_response_is_str(industry, short_list):
    result = (
        ClassificationLLM(model_name=MODEL_NAME)
        .reranker_sic(industry, short_list=short_list)[0]
        .model_dump()["selected_codes"][0]["reasoning"]
    )
    assert isinstance(result, str)
