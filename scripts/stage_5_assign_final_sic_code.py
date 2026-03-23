#!/usr/bin/env python3
# pylint: disable=duplicate-code, redefined-outer-name
"""This script assigns final SIC code using intermediate outputs from previous stages.
It reloads the output from the previous stage as a DataFrame object, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

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
# Constants:
INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
SIC_CANDIDATES_COL = "alt_sic_candidates"
OPEN_QUESTION_COL = "followup_question"
ANSWER_TO_OPEN_QUESTION_COL = "followup_answer"
CLOSED_QUESTION = ""
ANSWER_TO_CLOSED_QUESTION = ""

#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


async def get_final_sic_batch_async(
    batch: pd.DataFrame, c_llm: ClassificationLLM
) -> list[dict[str, Any]]:
    """Processes a batch of rows asynchronously to assign final SIC codes."""
    tasks = []
    for _, row in batch.iterrows():
        task = asyncio.create_task(
            c_llm.final_sic_code(
                industry_descr=row[MERGED_INDUSTRY_DESC_COL],
                job_title=row[JOB_TITLE_COL],
                job_description=row[JOB_DESCRIPTION_COL],
                sic_candidates=row[SIC_CANDIDATES_COL],
                open_question=row[OPEN_QUESTION_COL],
                answer_to_open_question=row[ANSWER_TO_OPEN_QUESTION_COL],
                # closed_question=CLOSED_QUESTION,
                # answer_to_closed_question=ANSWER_TO_CLOSED_QUESTION,
            )
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    results: list[dict[str, Any]] = []
    for final_assignment, _ in responses:
        results.append(
            {
                "unambiguously_codable_final": final_assignment.codable,
                "final_sic": final_assignment.unambiguous_code,
                "higher_level_final_sic": final_assignment.higher_level_code,
            }
        )
    return results


def get_unambiguous_status_final(row: pd.Series) -> bool:
    """Gets the final codability status from the intermediate results.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        final codability status (bool).
    """
    if row["intermediate_unambig_results"]["unambiguously_codable_final"] is not None:
        return row["intermediate_unambig_results"]["unambiguously_codable_final"]
    return False


def get_final_sic_code(row: pd.Series) -> str:
    """Gets the assigned SIC code from the intermediate results, if possible.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        final_code (str): the assigned SIC code.
    """
    if row["intermediate_unambig_results"]["final_sic"] is not None:
        return row["intermediate_unambig_results"]["final_sic"]
    return ""


def get_higher_level_sic_code(row: pd.Series) -> str:
    """Gets the higher level SIC code from the intermediate results, if possible.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        higher_level_code (str): the higher level SIC code if final code cannot be assigned
            unambiguously.
    """
    if row["intermediate_unambig_results"]["higher_level_final_sic"] is not None:
        return row["intermediate_unambig_results"]["higher_level_final_sic"]
    return ""


async def main_async(df, metadata, start_batch_id, args, c_llm):
    """Runs final SIC assignment in async batches."""
    print("running final SIC code assignment...")

    for col in [
        "unambiguously_codable_final",
        "final_sic",
        "higher_level_final_sic",
    ]:
        if col not in df.columns:
            df[col] = ""

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
            results = await get_final_sic_batch_async(batch, c_llm)
            batch.loc[batch.index, "intermediate_unambig_results"] = results
            df.loc[batch.index, "unambiguously_codable_final"] = batch.apply(
                get_unambiguous_status_final, axis=1
            )
            df.loc[batch.index, "final_sic"] = batch.apply(get_final_sic_code, axis=1)
            df.loc[batch.index, "higher_level_final_sic"] = batch.apply(
                get_higher_level_sic_code, axis=1
            )
            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("Final SIC code assignment is complete")
    print("deleting temporary DataFrame column...")
    df = df.drop("intermediate_unambig_results", axis=1)

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
    args = parse_args("STG5")

    df, metadata, start_batch_id = set_up_initial_state(parsed_args=args)

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
