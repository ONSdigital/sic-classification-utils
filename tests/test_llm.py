from industrial_classification_utils.llm.llm import ClassificationLLM
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
import pytest


@pytest.mark.parametrize("model, openai_api_key, expected_model",
    [
        ("gemini", None, ChatVertexAI),
        ("text-", None, ChatVertexAI),
        ("gpt", "key", ChatOpenAI)
    ],
)
@pytest.mark.initialise
def test_llm_model(model, openai_api_key, expected_model):
    llm_model_type = type(ClassificationLLM(model_name=model, openai_api_key=openai_api_key).llm)
    assert llm_model_type == expected_model

@pytest.mark.initialise
def test_llm_model_default():
    assert type(ClassificationLLM().llm) == ChatVertexAI

@pytest.mark.initialise
def test_model_name():
    assert ClassificationLLM().llm.model_name == "gemini-1.0-pro"