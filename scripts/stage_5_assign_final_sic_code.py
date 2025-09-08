#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script assigns final SIC code using intermediate outputs from previous stages.
It reloads the output from the previous stage as a DataFrame object, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_5_assign_final_sic_code.py --help
```

Example Usage:

1. Ensure you have (re-)authenticated with `gcloud` for the current project.

2. Run the script:
   ```bash
   python stage_5_assign_final_sic_code.py \
        -n my_output \
        -b 200 \
        persisted_dataframe.parquet \
        persisted_metadata.json \
        output_folder
   ```
   where:
     - `-n my_output` sets the output filename prefix to "my_output".
     - `-b 200` specifies to process in batches of 200 rows, checkpointing between batches.
     - `persisted_dataframe.parquet` is the saved dataframe output at the previous stage.
     - `persisted_metadata.json` is persisted JSON metadata from the previous stage.
     - `output_folder` is the directory where results will be saved.

3. Verify outputs exist as expected:
    ```bash
   ls output_folder
   ```
   (expect to see my_output.csv, my_output.parquet, and my_output_metadata.json)

"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.llm.llm import ClassificationLLM

from .shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
MODEL_NAME = "gemini-2.0-flash"
MODEL_LOCATION = "europe-west9"

CODE_DIGITS = 5

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


def assign_final_sic_code(row: pd.Series) -> dict:  # pylint: disable=C0103, W0613
    """Using the provided row data, call an LLM to assign a final SIC code.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing the columns corresponding
                         to the survey responses, sic candidates, and follow-up questions and
                         corresponding answers.

    Returns:
        result (dict): final codability, assigned 5 digit SIC code or a higher level code if
            final sic cannot be assigned unambiguously.
    """
    sa_final_sic = c_llm.final_sic_code(
        industry_descr=row[MERGED_INDUSTRY_DESC_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        sic_candidates=row[SIC_CANDIDATES_COL],
        open_question=row[OPEN_QUESTION_COL],
        answer_to_open_question=row[ANSWER_TO_OPEN_QUESTION_COL],
        closed_question=CLOSED_QUESTION,
        answer_to_closed_question=ANSWER_TO_CLOSED_QUESTION,
    )
    result = {
        "unambiguously_codable_final": sa_final_sic[0].codable,
        "final_sic": sa_final_sic[0].unambiguous_code,
        "higher_level_final_sic": sa_final_sic[0].higher_level_code,
    }
    return result


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


c_llm = ClassificationLLM(MODEL_NAME, verbose=False)
print("Classification LLM loaded.")

if __name__ == "__main__":
    args = parse_args()

    df, metadata, start_batch_id, restart_successful = set_up_initial_state(
        args.restart,
        args.output_folder,
        args.output_shortname,
        args.input_parquet_file,
        args.input_metadata_json,
        args.batch_size,
        stage_id="stage_k",
    )

    print("running final SIC code assignment...")
    if (not args.restart) or (not restart_successful):
        df["intermediate_unambig_results"] = {
            "unambiguously_codable_final": False,
            "final_sic": "",
            "higher_level_final_sic": "",
        }
        df["unambiguously_codable_final"] = False
        df["final_sic"] = ""
        df["higher_level_final_sic"] = ""

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(start_batch_id * args.batch_size, len(df), args.batch_size),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            batch.loc[batch.index, "intermediate_unambig_results"] = batch.apply(
                assign_final_sic_code, axis=1
            )
            df.loc[batch.index, "unambiguously_codable_final"] = batch.apply(
                get_unambiguous_status_final, axis=1
            )
            df.loc[batch.index, "final_sic"] = batch.apply(get_final_sic_code, axis=1)
            df.loc[batch.index, "higher_level_final_sic"] = batch.apply(
                get_higher_level_sic_code, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    print("Final SIC code assignment is complete")
    print("deleting temporary DataFrame column...")
    df = df.drop("intermediate_unambig_results", axis=1)

    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
