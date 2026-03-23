#!/usr/bin/env python3
# pylint: disable=duplicate-code, redefined-outer-name
"""This script generates a SIC classification based on respondent's data
using a Retrieval Augmented Generation (RAG) approach and persists the results.
It reloads the output from the previous stage as a DataFrame object,
performs classification for each row using an LLM, writes the results back into
the DataFrame, and then saves the results to CSV, parquet, and JSON metadata
files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""
import asyncio
from argparse import Namespace
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.llm.llm import ClassificationLLM
from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
CODE_DIGITS = 5
CANDIDATES_LIMIT = 10
MAX_ASYNC_BATCH_SIZE = 10

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
SEMANTIC_SEARCH_COL = "semantic_search_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


async def get_rag_response_batch_async(
    batch: pd.DataFrame, c_llm: ClassificationLLM
) -> list[dict[str, Any]]:
    """Processes a batch of rows concurrently to generate RAG SIC results.

    Args:
        batch: Batch of rows. Must include the columns defined by
            `JOB_TITLE_COL`, `JOB_DESCRIPTION_COL`, `INDUSTRY_DESCR_COL`, and
            `SEMANTIC_SEARCH_COL`.
        c_llm: Initialised LLM wrapper used to run the RAG prompt.

    Returns:
        A list of dictionaries (one per row) that can be written directly into
        the output columns (`unambiguously_codable`, `initial_code`,
        `alt_sic_candidates`, `followup_question`).
    """
    tasks = []
    for _, row in batch.iterrows():
        task = asyncio.create_task(
            c_llm.sa_rag_sic_code(
                job_title=row[JOB_TITLE_COL],
                job_description=row[JOB_DESCRIPTION_COL],
                industry_descr=row[INDUSTRY_DESCR_COL],
                code_digits=CODE_DIGITS,
                candidates_limit=CANDIDATES_LIMIT,
                short_list=row[SEMANTIC_SEARCH_COL],
            )
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    results: list[dict[str, Any]] = []
    for sic_response, _, _ in responses:
        results.append(
            {
                "initial_code": sic_response.sic_code or "",
                "followup_question": sic_response.followup or "",
                "unambiguously_codable": (sic_response.sic_code or "") != "",
                "alt_sic_candidates": [
                    {
                        "code": i.sic_code,
                        "likelihood": i.likelihood,
                        "title": i.sic_descriptive,
                    }
                    for i in sic_response.sic_candidates
                ],
            }
        )
    return results


async def main_async(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    start_batch_id: int,
    args: Namespace,
    c_llm: ClassificationLLM,
) -> None:
    """Runs async RAG SIC allocation and persists checkpoints.

    Args:
        df: The input DataFrame containing the survey responses and semantic search results.
        metadata: The metadata dictionary loaded from the input JSON file.
        start_batch_id: The batch ID to start from, determined by the checkpointing logic.
        args: The command-line arguments parsed by `parse_args()`.
        c_llm: An initialised instance of the ClassificationLLM class.
    """
    print("Running RAG SIC allocation...")

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(
                    start_batch_id * metadata["batch_size_async"],
                    len(df),
                    metadata["batch_size_async"],
                ),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            results = await get_rag_response_batch_async(batch, c_llm)

            # Write results directly into output columns (no extra apply helpers)
            df.loc[batch.index, "unambiguously_codable"] = pd.Series(
                [r["unambiguously_codable"] for r in results],
                index=batch.index,
            )
            df.loc[batch.index, "initial_code"] = pd.Series(
                [r["initial_code"] for r in results],
                index=batch.index,
            )
            df.loc[batch.index, "alt_sic_candidates"] = pd.Series(
                [r["alt_sic_candidates"] for r in results],
                index=batch.index,
                dtype="object",
            )
            df.loc[batch.index, "followup_question"] = pd.Series(
                [r["followup_question"] for r in results],
                index=batch.index,
            )

            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("RAG SIC allocation is complete")
    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_args("STG2")
    df, metadata, start_batch_id = set_up_initial_state(args)

    if "unambiguously_codable" not in df.columns:
        df["unambiguously_codable"] = False
    if "initial_code" not in df.columns:
        df["initial_code"] = ""
    if "alt_sic_candidates" not in df.columns:
        df["alt_sic_candidates"] = np.empty((len(df), 0)).tolist()
    if "followup_question" not in df.columns:
        df["followup_question"] = ""

    uni_chat = ClassificationLLM(
        model_name=metadata["model_name"],
        model_location=metadata["model_location"],
        verbose=False,
    )

    asyncio.run(
        main_async(
            df=df,
            metadata=metadata,
            start_batch_id=start_batch_id,
            args=args,
            c_llm=uni_chat,
        )
    )
