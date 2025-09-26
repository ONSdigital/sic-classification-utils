# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).
---
## [0.1.5] - 2025-09-26

## Changed
- updated `__init__` method of ClassificationLLM to pass thinking_budget as 0 to accommodate newer gemini model (2.5-flash) and set region to europe-west1
- updated example scripts to use the new gemini-2.5-flash model as gemini-1.5-flash is now deprecated

## [0.1.4] - 2025-09-18

## Added
- `final_sic_code` method for `ClassificationLLM class in `industrial_classification.llm.llm.py`, to assign SIC using initial response & followup question and response.
- `formulate_open_question` and `formulate_closed_question` methods for  `ClassificationLLM class in `industrial_classification.llm.llm.py`, to construct followup questions.
- `synthetic_responses` module, with a `SyntheticResponder` class capable of emulating user interaction.
- "Evaluation Pipeline" scripts to batch process survey responses, stored in `scripts/` folder.

## Changed
- updated `unambiguous_sic_code` method of ClassificationLLM to accept the raw semantic search output as input, rather than needing it to be pre-formatted.
- updated `unambiguous_sic_code` method and `UnambiguousResponse` response model to check if a response is unambiguously codable to 5-digits.

---
## [0.1.3] - 2025-06-30

## Added
- Tests for `industrail_classification.embed.embedding.py`

## Changed
- Package versions in `pyproject.toml`:
    - Added langchain models to use the most recent versions (langchain-google-genai, langchain-core, langchain-community, langchain-huggingface, langchain-chroma)
    - Updated package versions (langchain-openai, numpy, chromadb)
    - Removed packages that were replaced by their newer counterparts (langchain was replaced with langchain-core, langchain-community, langchain-huggingface, langchain-chroma; langchain-google-vertexai replaced with langchain-google-genai that uses different LLM model)

