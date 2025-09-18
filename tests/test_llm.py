# pylint: disable=C0116
"""Tests for industrial_classification_utils.llm.llm.py."""

import json
from unittest import mock

import pytest
import vertexai
import pandas as pd
from industrial_classification.hierarchy.sic_hierarchy import SIC, SicCode, SicNode
from langchain_core.messages import AIMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from industrial_classification_utils.llm.llm import ClassificationLLM
from industrial_classification.meta.classification_meta import ClassificationMeta
from industrial_classification_utils.models.response_model import (
    FinalSICAssignment,
    SicResponse,
    SurveyAssistSicResponse,
    UnambiguousResponse,
)
from industrial_classification.hierarchy.sic_hierarchy import load_hierarchy

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
    llm_model = ClassificationLLM(llm="model").llm
    assert llm_model == "model"


@pytest.mark.utils
def test_llm_model_default():
    assert isinstance(ClassificationLLM().llm, ChatVertexAI)


@pytest.mark.utils
def test_model_name():
    assert ClassificationLLM().llm.model_name == "gemini-1.0-pro"


@pytest.mark.utils
def test_llm_response_mocked_get_sic_code(mocker):
    mock_object_dict = {
        "codable": True,
        "followup": "This is follow-up",
        "sic_code": "12345",
        "sic_descriptive": "description12345",
        "sic_candidates": [
            {
                "sic_code": "23456",
                "sic_descriptive": "description23456",
                "likelihood": 0.5,
            },
            {
                "sic_code": "34567",
                "sic_descriptive": "description34567",
                "likelihood": 0.5,
            },
        ],
        "reasoning": "reasoning12345",
    }
    mock_object_json = json.dumps(mock_object_dict)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mock_patcher = mocker.patch(  # noqa: F841
        "industrial_classification_utils.llm.llm.ChatVertexAI.invoke",
        return_value=mock_message,
    )

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
        },
    ]
    mock_object_dict = {
        "codable": True,
        "followup": "This is follow-up",
        "sic_code": "12345",
        "sic_descriptive": "description12345",
        "sic_candidates": [
            {
                "sic_code": "23456",
                "sic_descriptive": "description23456",
                "likelihood": 0.5,
            },
            {
                "sic_code": "34567",
                "sic_descriptive": "description34567",
                "likelihood": 0.5,
            },
        ],
        "reasoning": "reasoning12345",
    }
    mock_object_json = json.dumps(mock_object_dict)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mock_patcher = mocker.patch(  # noqa: F841
        "industrial_classification_utils.llm.llm.ChatVertexAI.invoke",
        return_value=mock_message,
    )

    result = prompt_candidate_sic.sa_rag_sic_code(
        industry_descr="", job_description="", job_title="", short_list=short_list
    )
    assert isinstance(result[0], SurveyAssistSicResponse)
    assert isinstance(result[1], list)
    assert isinstance(result[2], dict)


@pytest.mark.utils
def test_llm_response_mocked_unambiguous_sic_code(mocker, prompt_candidate_sic):
    sic_candidates = [
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
        },
    ]
    mock_object_dict = {
        "codable": True,
        "followup": "This is follow-up",
        "sic_code": "12345",
        "sic_descriptive": "description12345",
        "sic_candidates": [
            {
                "sic_code": "23456",
                "sic_descriptive": "description23456",
                "likelihood": 0.5,
            },
            {
                "sic_code": "34567",
                "sic_descriptive": "description34567",
                "likelihood": 0.5,
            },
        ],
        "reasoning": "This is reasoning for the llm answer. Padded to 50 characters (Pydantic)",
    }
    mock_object_json = json.dumps(mock_object_dict)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mock_patcher = mocker.patch(  # noqa: F841
        "industrial_classification_utils.llm.llm.ChatVertexAI.invoke",
        return_value=mock_message,
    )

    result = prompt_candidate_sic.unambiguous_sic_code(
        industry_descr="",
        job_description="",
        job_title="",
        sic_candidates=sic_candidates,
    )
    assert isinstance(result[0], UnambiguousResponse)
    assert isinstance(result[1], dict)


@pytest.mark.utils
def test_llm_response_mocked_final_sic_code(mocker, prompt_candidate_sic):
    sic_candidates = [
        {
            "class_code": "12345",
            "class_descriptive": "description1",
            "likelihood": 0.8,
        },
        {
            "class_code": "23456",
            "class_descriptive": "description2",
            "likelihood": 0.6,
        },
    ]
    mock_object_dict = {
        "codable": True,
        "followup": "This is follow-up",
        "sic_code": "12345",
        "sic_descriptive": "description12345",
        "sic_candidates": [
            {
                "sic_code": "23456",
                "sic_descriptive": "description23456",
                "likelihood": 0.5,
            },
            {
                "sic_code": "34567",
                "sic_descriptive": "description34567",
                "likelihood": 0.5,
            },
        ],
        "reasoning": "This is reasoning for the llm answer. Padded to 50 characters (Pydantic)",
    }
    mock_object_json = json.dumps(mock_object_dict)

    mock_message = mocker.Mock(spec=AIMessage)
    mock_message.content = mock_object_json

    mock_patcher = mocker.patch(  # noqa: F841
        "industrial_classification_utils.llm.llm.ChatVertexAI.invoke",
        return_value=mock_message,
    )

    result = prompt_candidate_sic.final_sic_code(
        industry_descr="",
        job_title="",
        job_description="",
        sic_candidates=str(sic_candidates),
        open_question="",
        answer_to_open_question="",
        closed_question="",
        answer_to_closed_question="",
    )
    assert isinstance(result[0], FinalSICAssignment)
    assert isinstance(result[1], dict)

@pytest.fixture
def mock_sic_meta():
    SICMeta_mock = {}
    SICMeta_mock["Axxxxx"] = {"title": "titleA", "detail":"detailA"}
    SICMeta_mock["A11xxx"] = {"title": "title11", "detail":"detail11"}
    SICMeta_mock["A111xx"] = {"title": "title111", "detail":"detail111"}
    SICMeta_mock["A1111x"] = {"title": "title1111", "detail":"detail1111", "includes": ["includes1111", "includes1111A"]}
    SICMeta_mock["A11111"] = {"title": "title11111", "detail":"detail11111", "excludes": ["excludes11111"]}
    SICMeta_mock["A11112"] = {"title": "title11112", "detail":"detail11112"}
    sic_meta_mock = [
    ClassificationMeta.model_validate({"code": k} | v) for k, v in SICMeta_mock.items()
    ]
    return sic_meta_mock


@pytest.fixture
def mock_sic_meta_patch(mock_sic_meta):
    with mock.patch('industrial_classification.meta.sic_meta.sic_meta', mock_sic_meta):
        yield


@pytest.fixture
def classification_llm_with_sic():
    nodes = [
        SicNode(sic_code=SicCode("Axxxxx"), description="descriptionA"),
        SicNode(sic_code=SicCode("A11xxx"), description="description11"),
        SicNode(sic_code=SicCode("A111xx"), description="description111"),
        SicNode(sic_code=SicCode("A1111x"), description="description1111"),
        SicNode(sic_code=SicCode("A11111"), description="description11111"),
        SicNode(sic_code=SicCode("A11112"), description="description11112"),
    ]
    llm_class = ClassificationLLM(model_name="gemini-1.5-flash")

    index_mock = {'uk_sic_2007': ["11111", "11112"],
    'activity': ["activity1", "activity2"]}
    sic_index_df_mock = pd.DataFrame(index_mock)
    df_mock = {'description':["desc1", "desc2", "desc3", "desc4", "desc5", "desc6"],
    'section': ["A", "A", "A", "A", "A", "A"],
    'most_disaggregated_level': ["11111", "11112", "1111", "111", "11", "A"],
    'level_headings': ["Sub Class", "Sub Class", "Class", "Group", "Division", "SECTION"]}
    sic_df_mock = pd.DataFrame(df_mock)
    sic = load_hierarchy(sic_df_mock, sic_index_df_mock)

    llm_class.sic = sic

    return llm_class

@pytest.mark.utils
def test_prompt_candidate_include_all(mock_sic_meta_patch, classification_llm_with_sic):

    result111 = classification_llm_with_sic._prompt_candidate("111", ["activity"], include_all = True)
    result1111 = classification_llm_with_sic._prompt_candidate("1111", ["activity"], include_all = True)
    result11111 = classification_llm_with_sic._prompt_candidate("11111", ["activity"], include_all = True)

    assert isinstance(result111, str)
    assert isinstance(result1111, str)
    assert isinstance(result11111, str)
    assert all(x in result111 for x in ["Code", "Title", "Details"])
    assert all(x in result1111 for x in ["Code", "Title", "Details", "Includes"])
    assert all(x in result11111 for x in ["Code", "Title", "Details", "Excludes"])
    

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

@pytest.mark.utils
def test_open_api_key_raise_not_implemented_error():
    with pytest.raises(NotImplementedError, match="Need to provide an OpenAI API key"):
        ClassificationLLM(model_name="gpt")


@pytest.mark.utils
def test_model_family_raise_not_implemented_error():
    with pytest.raises(NotImplementedError, match="Unsupported model family"):
        ClassificationLLM(model_name="aaaa")


# @patch('industrial_classification_utils.models.response_model.SicResponse')
# def test_except_value_error_get_sic_code(mock_parser_class):
#     mock_parser_instance = mock_parser_class.return_value
#     mock_parser_instance.parse.side_effect = ValueError("Parsing error")


@pytest.mark.utils
def test_unambiguous_sic_code_sic_candidates_is_none_raise_value_error():
    with pytest.raises(ValueError, match="Short list is None - list provided from embedding search."):
        ClassificationLLM(model_name=MODEL_NAME).unambiguous_sic_code(industry_descr="", job_description="", job_title="")


@pytest.mark.utils
def test_sa_rag_sic_code_short_list_is_none_raise_value_error():
    with pytest.raises(ValueError, match="Short list is None - list provided from embedding search."):
        ClassificationLLM(model_name=MODEL_NAME).sa_rag_sic_code(industry_descr="", job_description="", job_title="")


@pytest.mark.utils
def test_reranker_sic_short_list_is_none_raise_value_error():
    with pytest.raises(ValueError, match="Short list is None - list provided from embedding search."):
        ClassificationLLM(model_name=MODEL_NAME).reranker_sic(industry_descr="", job_description="", job_title="")
