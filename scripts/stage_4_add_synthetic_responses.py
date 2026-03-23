#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script answers followup questions and persists the results.
It reads reloads the output from the previous stage as a DataFrame object,
answers the question in each row, creates a new column in the DataFrame
with this information, and then saves the results to CSV, parquet, and
JSON metadata files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""
import numpy as np
from tqdm import tqdm

from industrial_classification_utils.synthetic_responses.synthetic_response_utils import (
    SyntheticResponder,
)
from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
FOLLOWUP_QUESTION_COL = "followup_question"
FOLLOWUP_ANSWER_COL = "followup_answer"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def get_followup_answer(row: dict, one_sr: SyntheticResponder) -> str:
    """Answer followup question using the provided row data.
    Intended for use as a `.apply()` operation to create a new column in a pd.DataFrame object.

    Args:
        row (dict): A dictionary containing the survey responses and the followup question.
        one_sr (SyntheticResponder): An instance of the SyntheticResponder class,
            used to generate the answer.

    Returns:
        str: The generated answer to the followup question.
    """
    payload = {
        "industry_descr": row[MERGED_INDUSTRY_DESC_COL],
        "job_title": row[JOB_TITLE_COL],
        "job_description": row[JOB_DESCRIPTION_COL],
    }
    if not row["unambiguously_codable"]:
        answer_followup_prompt = one_sr.construct_prompt(
            payload, row[FOLLOWUP_QUESTION_COL]
        )
        return one_sr.answer_followup(answer_followup_prompt, payload).answer
    return ""


if __name__ == "__main__":
    args = parse_args("STG4")

    df, metadata, start_batch_id = set_up_initial_state(args)

    sr = SyntheticResponder(
        persona=None,
        get_question_function=None,
        model_name=metadata["model_name"],
        model_location=metadata["model_location"],
    )

    print("getting synthetic responses to followup questions...")
    if FOLLOWUP_ANSWER_COL not in df.columns:
        df[FOLLOWUP_ANSWER_COL] = ""

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(
                    start_batch_id * metadata["batch_size"],
                    len(df),
                    metadata["batch_size"],
                ),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            df.loc[batch.index, FOLLOWUP_ANSWER_COL] = batch.apply(
                lambda row: get_followup_answer(row, one_sr=sr), axis=1
            )
            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("synthetic response generation is complete")

    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")
