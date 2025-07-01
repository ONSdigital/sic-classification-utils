# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).
---
## [0.1.3] - 2025-06-30

## Added
- Tests for `industrail_classification.embed.embedding.py`

## Changed
- Package versions in `pyproject.toml`:
    - Added langchain models to use the most recent versions (langchain-google-genai, langchain-core, langchain-community, langchain-huggingface, langchain-chroma)
    - Updated package versions (langchain-openai, numpy, chromadb)
    - Removed packages that were replaced by their newer counterparts (langchain was replaced with langchain-core, langchain-community, langchain-huggingface, langchain-chroma; langchain-google-vertexai replaced with langchain-google-genai that uses different LLM model)

