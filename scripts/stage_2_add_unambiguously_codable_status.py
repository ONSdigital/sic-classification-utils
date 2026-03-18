#!/usr/bin/env python3
# pylint: disable=duplicate-code, redefined-outer-name
"""This script analyzes a dataset to determine if each record is
"unambiguously codable" for a Standard Industrial Classification (SIC) code.

It reloads the output from the previous stage as a DataFrame object, uses a
Large Language Model (LLM) to assess codability for each row, and adds new
columns for the codability status, an initial SIC code (if one can be
assigned), and a list of alternative SIC candidates. The results are then saved
to CSV, parquet, and JSON metadata files in a user-specified output folder.

The script requires a configured connection to a compatible LLM.

See README_evaluation_pipeline.md for more details on how to run.
"""
import asyncio
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
# Default values and constants:
MODEL_NAME = "gemini-2.5-flash"
MODEL_LOCATION = "europe-west1"
CODE_DIGITS = 5
CANDIDATES_LIMIT = 10

MAX_ASYNC_BATCH_SIZE = 10

JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"

OUTPUT_COLS_INITIAL = {
    "sic_code_col": "initial_code",
    "codable_col": "unambiguously_codable",
    "alt_candidates_col": "alt_sic_candidates",
    "semantic_search_col": "semantic_search_results",
}
OUTPUT_COLS_FINAL = {
    "sic_code_col": "final_code",
    "codable_col": "unambiguously_codable_final",
    "alt_candidates_col": "alt_sic_candidates_final",
    "semantic_search_col": "second_semantic_search_results",
}
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def _update_metadata_with_args_and_defaults(parsed_args, in_metadata):
    """Updates the metadata dictionary with values from the command-line arguments,
    using defaults where necessary.

    Args:
        parsed_args: The command-line arguments parsed by `parse_args()`.
        in_metadata: The initial metadata dictionary loaded from the input JSON file.

    Returns:
        dict: The updated metadata dictionary with values from args and defaults.
    """
    updated_metadata = in_metadata.copy() if in_metadata else {}

    # Persisted checkpoint metadata expects this key.
    updated_metadata["batch_size"] = parsed_args.batch_size

    # Update metadata with values from parsed_args, using defaults where necessary
    updated_metadata["model_name"] = updated_metadata.get("model_name", MODEL_NAME)
    updated_metadata["model_location"] = updated_metadata.get(
        "model_location", MODEL_LOCATION
    )
    updated_metadata["code_digits"] = updated_metadata.get("code_digits", CODE_DIGITS)
    updated_metadata["candidates_limit"] = updated_metadata.get(
        "candidates_limit", CANDIDATES_LIMIT
    )

    if parsed_args.batch_size > MAX_ASYNC_BATCH_SIZE:
        print(f"batch size too large. lower batch size to {MAX_ASYNC_BATCH_SIZE}")

    updated_metadata["batch_size_async"] = min(
        parsed_args.batch_size, MAX_ASYNC_BATCH_SIZE
    )

    return updated_metadata


# This new async function processes a whole batch of rows concurrently.
async def get_unambiguous_sic_batch_async(
    batch: pd.DataFrame,
    semantic_search_col: str,
    c_llm: ClassificationLLM,
    candidates_limit: int,
    code_digits: int,
) -> list[dict[str, Any]]:
    """Processes a batch of rows asynchronously to get unambiguous SIC codability."""
    # 1. Create a task for each row in the batch
    tasks = []
    for _, row in batch.iterrows():
        task = asyncio.create_task(
            c_llm.unambiguous_sic_code(
                industry_descr=row[MERGED_INDUSTRY_DESC_COL],
                semantic_search_results=row[semantic_search_col],
                job_title=row[JOB_TITLE_COL],
                job_description=row[JOB_DESCRIPTION_COL],
                candidates_limit=candidates_limit,
                code_digits=code_digits,
            )
        )
        tasks.append(task)

    # 2. Run all tasks concurrently
    responses = await asyncio.gather(*tasks)

    # 3. Process the results from asyncio.gather()
    results = []
    for sa_response, _ in responses:
        results.append(
            {
                "unambiguously_codable": sa_response.codable,
                "code": sa_response.class_code,
                "alt_candidates": [
                    {
                        "code": i.class_code,
                        "title": i.class_descriptive,
                        "likelihood": i.likelihood,
                    }
                    for i in sa_response.alt_candidates
                ],
            }
        )
    return results


async def main_async(
    df: pd.DataFrame,
    metadata: dict,
    start_batch_id: int,
    args,
    c_llm: ClassificationLLM,
):
    """Runs the unambiguous codability analysis (async batch processing).

    Args:
    df (pd.DataFrame): The input DataFrame containing the dataset to analyze.
    metadata (dict): The metadata dictionary containing configuration values.
    start_batch_id (int): The batch ID to start processing from (when restarting from checkpoint).
    args: The command-line arguments parsed by `parse_args()`.
    c_llm (ClassificationLLM): An instance of the ClassificationLLM class for making LLM calls.

    """
    col_names = OUTPUT_COLS_FINAL if args.second_run else OUTPUT_COLS_INITIAL

    if col_names["codable_col"] not in df.columns:
        df[col_names["codable_col"]] = False
    if col_names["sic_code_col"] not in df.columns:
        df[col_names["sic_code_col"]] = ""
    if col_names["alt_candidates_col"] not in df.columns:
        df[col_names["alt_candidates_col"]] = [[] for _ in range(len(df))]

    print("running unambiguous codability analysis...")

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
            results = await get_unambiguous_sic_batch_async(
                batch,
                semantic_search_col=col_names["semantic_search_col"],
                c_llm=c_llm,
                candidates_limit=metadata["candidates_limit"],
                code_digits=metadata["code_digits"],
            )

            # Write results directly into output columns (no extractor helpers)
            df.loc[batch.index, col_names["codable_col"]] = pd.Series(
                [bool(r.get("unambiguously_codable", False)) for r in results],
                index=batch.index,
            )
            df.loc[batch.index, col_names["sic_code_col"]] = pd.Series(
                [str(r.get("code") or "") for r in results],
                index=batch.index,
            )
            df.loc[batch.index, col_names["alt_candidates_col"]] = pd.Series(
                [(r.get("alt_candidates") or []) for r in results],
                index=batch.index,
                dtype="object",
            )

            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("unambiguous codability analysis is complete")
    if "intermediate_unambig_results" in df.columns:
        df.drop(columns=["intermediate_unambig_results"], inplace=True)
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

    df, metadata, start_batch_id = set_up_initial_state(parsed_args=args)

    metadata = _update_metadata_with_args_and_defaults(args, metadata)

    c_llm = ClassificationLLM(
        model_name=metadata["model_name"],
        model_location=metadata["model_location"],
        verbose=False,
    )
    print("Classification LLM loaded.")

    asyncio.run(
        main_async(
            df=df,
            metadata=metadata,
            start_batch_id=start_batch_id,
            args=args,
            c_llm=c_llm,
        )
    )
