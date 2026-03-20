#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script modifies industry description with followup questions and follow up answers,
and persists the results. It reads reloads the output from the previous stage as a DataFrame
object, modifies each row with selected method, overwrites 'merged_industry_desc' column in
the DataFrame with this information, and then saves the results to CSV, parquet, and
JSON metadata files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""
import numpy as np
import pandas as pd
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
EXTENDED_INDUSTRY_METHOD = "concatenate"

FOLLOWUP_QUESTION = "followup_question"
FOLLOWUP_ANSWER = "followup_answer"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"

OUTPUT_COL = "extended_industry_desc"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def get_rephrased_industry_desc(
    row: pd.Series, one_sr: SyntheticResponder, method: str = "concatenate"
) -> str:
    """Rephrase industry description with follow up question and follow up answer as a label.

    Args:
        row (pd.Series): A row from the input DataFrame containing the survey responses,
                         and the followup question.
        one_sr (SyntheticResponder): An instance of the SyntheticResponder class, used to rephrase
              the industry description if method is set to 'rephrase'.
        method (str): method to use for rephrasing. Options are 'concatenate' and 'rephrase'.

    Returns: response (str)
    """
    if row[FOLLOWUP_ANSWER] == "":
        return row[MERGED_INDUSTRY_DESC_COL]
    if method == "concatenate":
        return (
            row[MERGED_INDUSTRY_DESC_COL]
            + """
- Followup Question: """
            + row[FOLLOWUP_QUESTION]
            + """
- Followup Answer: """
            + row[FOLLOWUP_ANSWER]
        )
    if method == "rephrase" and row[FOLLOWUP_QUESTION] != "":
        return one_sr.rephrase_question_and_id(
            row[MERGED_INDUSTRY_DESC_COL],
            row[FOLLOWUP_QUESTION],
            row[FOLLOWUP_ANSWER],
        )[0]
    return row[MERGED_INDUSTRY_DESC_COL]


if __name__ == "__main__":
    args = parse_args("STG5")

    df, metadata, start_batch_id = set_up_initial_state(args)

    metadata["extended_industry_method"] = EXTENDED_INDUSTRY_METHOD

    sr = SyntheticResponder(
        persona=None,
        get_question_function=None,
        model_name=metadata["model_name"],
    )

    print(
        "rephrasing industry description with follow up questions and followup answers..."
    )
    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = ""

    # rephrase new job description
    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(start_batch_id * args.batch_size, len(df), args.batch_size),
            )
        )
    ):
        if batch_id == 0:
            pass
        else:
            df.loc[batch.index, OUTPUT_COL] = batch.apply(
                lambda row: get_rephrased_industry_desc(
                    row, one_sr=sr, method=metadata["extended_industry_method"]
                ),
                axis=1,
            )
            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("industry description modification is complete")
    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")
