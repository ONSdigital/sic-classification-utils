# pylint: disable=invalid-name, protected-access, line-too-long, missing-module-docstring, duplicate-code
import asyncio
import json
from pathlib import Path
from pprint import pprint

from industrial_classification_utils.llm import ClassificationLLM

DATA_DIR = Path(__file__).resolve().parent / "data"

# Inputs for ClassificationLLM methods
JOB_TITLE = "psychologist"
JOB_DESCRIPTION = "I help adults who have mental health difficulties"
ORG_DESCRIPTION = "adult mental health"

with (DATA_DIR / "adult_mental_health_embed_short_list.json").open(
    encoding="utf-8"
) as handle:
    EXAMPLE_EMBED_SHORT_LIST = json.load(handle)

if __name__ == "__main__":
    uni_chat = ClassificationLLM(model_name="gemini-2.5-flash", verbose=True)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    sic_response_unambiguous = loop.run_until_complete(
        uni_chat.unambiguous_sic_code(
            industry_descr=ORG_DESCRIPTION,
            semantic_search_results=EXAMPLE_EMBED_SHORT_LIST,
            job_title=JOB_TITLE,
            job_description=JOB_DESCRIPTION,
        )
    )

    # Formulate Open Question
    sic_followup = loop.run_until_complete(
        uni_chat.formulate_open_question(
            industry_descr=ORG_DESCRIPTION,
            job_title=JOB_TITLE,
            job_description=JOB_DESCRIPTION,
            llm_output=sic_response_unambiguous[0].alt_candidates,  # type: ignore
        )
    )

    print("Open Question answer: Follow-up and Reasoning")
    pprint(sic_followup[0].model_dump(), indent=2, width=80)

    filtered_list = [
        elem.class_code for elem in sic_response_unambiguous[0].alt_candidates
    ]
    filtered_candidates = uni_chat._prompt_candidate_list_filtered(
        EXAMPLE_EMBED_SHORT_LIST,
        filtered_list=filtered_list,
        activities_limit=5,  # type: ignore
    )

    # Formulate Closed Quesiton
    sic_closed_followup = loop.run_until_complete(
        uni_chat.formulate_closed_question(
            industry_descr=ORG_DESCRIPTION,
            job_title=JOB_TITLE,
            job_description=JOB_DESCRIPTION,
            llm_output=filtered_candidates,  # type: ignore
        )
    )
    print(
        """\nClosed Quesiton answer: Follow-up, Reasoning, and List of simplified SIC options to choose from"""
    )
    pprint(sic_closed_followup[0].model_dump(), indent=2, width=80)

    print("\nCandidate list for the prompt")
    pprint(filtered_candidates, indent=2, width=80)
