#!/usr/bin/env python3
# pylint: disable=duplicate-code, redefined-outer-name
"""This script retrieves followup questions and persists the results.
It reads reloads the output from the previous stage as a DataFrame object,
retireves a follow-up question for each row, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""
import asyncio

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
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"

CANDIDATE_SIC_COL = "alt_sic_candidates"
OUTPUT_COL = "followup_question"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


async def get_open_question_batch_async(
    batch: pd.DataFrame, c_llm: ClassificationLLM
) -> list[str]:
    """Process a batch of rows asynchronously to generate an open follow-up question for each row.

    Args:
        batch (pd.DataFrame): A batch of DataFrame containing rows with columns corresponding
                         to the survey responses, and the semantic search results.
        c_llm (ClassificationLLM): An initialised instance of the ClassificationLLM class.

    Returns: question (str).
    """
    tasks = []
    for _, row in batch.iterrows():
        task = asyncio.create_task(
            c_llm.formulate_open_question(
                industry_descr=row[MERGED_INDUSTRY_DESC_COL],
                job_title=row[JOB_TITLE_COL],
                job_description=row[JOB_DESCRIPTION_COL],
                llm_output=row[CANDIDATE_SIC_COL],  # type: ignore
            )
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    results = []
    for sic_followup_object, _ in responses:
        if sic_followup_object.followup is None:
            results.append("")
        else:
            results.append(sic_followup_object.followup)
    return results


async def main_async(df, metadata, start_batch_id, args, c_llm):
    """Main function to generate follow up questions.
    Deviates from the stage_k template to enable async processing.
    """
    print("getting followup questions ...")

    df_uncodable = df[~df["unambiguously_codable"]]

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df_uncodable,
                np.arange(
                    start_batch_id * metadata["batch_size_async"],
                    len(df_uncodable),
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
            results = await get_open_question_batch_async(batch, c_llm)
            df.loc[batch.index, OUTPUT_COL] = results

            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("Followup question retrieval is complete")

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
    args = parse_args("STG3")

    df, metadata, start_batch_id = set_up_initial_state(args)

    c_llm = ClassificationLLM(
        model_name=metadata["model_name"],
        model_location=metadata["model_location"],
        verbose=False,
    )
    print("Classification LLM loaded.")

    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = ""

    asyncio.run(
        main_async(
            df=df,
            metadata=metadata,
            start_batch_id=start_batch_id,
            args=args,
            c_llm=c_llm,
        )
    )
