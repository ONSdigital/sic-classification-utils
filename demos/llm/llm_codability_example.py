"""This module provides illustration for the use of unambiguous prompt with an LLM."""

# %%
import asyncio
import json
from pathlib import Path
from pprint import pprint

import nest_asyncio

from industrial_classification_utils.llm import ClassificationLLM

# pylint: disable=duplicate-code
# %%
nest_asyncio.apply()
DATA_DIR = Path(__file__).resolve().parent / "data"

LLM_MODEL = "gemini-2.5-flash"
JOB_TITLE = "psychologist"
JOB_DESCRIPTION = "I help adults who have mental health difficulties"
ORG_DESCRIPTION = "adult mental health"

with (DATA_DIR / "adult_mental_health_embed_short_list.json").open(
    encoding="utf-8"
) as handle:
    EXAMPLE_EMBED_SHORT_LIST = json.load(handle)

# %%
uni_chat = ClassificationLLM(model_name=LLM_MODEL, verbose=True)

sa_response = asyncio.run(
    uni_chat.unambiguous_sic_code(
        industry_descr=ORG_DESCRIPTION,
        semantic_search_results=EXAMPLE_EMBED_SHORT_LIST,
        job_title=JOB_TITLE,
        job_description=JOB_DESCRIPTION,
        code_digits=5,
        candidates_limit=7,
    )
)

pprint(sa_response[0].model_dump(), indent=2, width=80)
