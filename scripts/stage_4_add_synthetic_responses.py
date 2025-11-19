#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script answers followup questions and persists the results.
It reads reloads the output from the previous stage as a DataFrame object,
answers the question in each row, creates a new column in the DataFrame
with this information, and then saves the results to CSV, parquet, and
JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_4_add_snthetic_responses.py --help
```

Example Usage:

1. Ensure you have (re-)authenticated with `gcloud` for the current project.

2. Run the script:
   ```bash
   python stage_4_add_snthetic_responses.py \
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
MODEL_NAME = "gemini-2.5-flash"
MODEL_LOCATION = "europe-west1"

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def get_followup_answer(row: pd.Series) -> str:  # pylint: disable=C0103, W0613
    """Answer followup question using the provided row data.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing the survey responses,
                         and the followup question.
    Returns: llm_response (str).
    """
    payload = {
        "industry_descr": row[MERGED_INDUSTRY_DESC_COL],
        "job_title": row[JOB_TITLE_COL],
        "job_description": row[JOB_DESCRIPTION_COL],
    }
    if row["followup_question"] is not None:
        answer_followup_prompt = SR.construct_prompt(payload, row["followup_question"])
        llm_response = SR.answer_followup(answer_followup_prompt, payload)
    else:
        llm_response = ""
    return llm_response


# pylint: disable=C0116 # the docstring is below
def get_rephrased_id(row: pd.Series) -> str:
    """Rephrase industry description with follow up question and follow up answer as a label.

    Args:
        row (pd.Series): A row from the input DataFrame containing the survey responses,
                         and the followup question.

    Returns:
        str: response (str)
    """
    return SR.rephrase_question_and_id(row["merged_industry_desc"])[0]


SR = SyntheticResponder(persona=None, get_question_function=None, model_name=MODEL_NAME)

if __name__ == "__main__":
    args = parse_args("STG4")

    job_description_rephrased = []
    df, metadata, start_batch_id, restart_successful = set_up_initial_state(
        args.restart,
        args.output_folder,
        args.output_shortname,
        args.input_parquet_file,
        args.input_metadata_json,
        args.batch_size,
        stage_id="stage_4",
    )
    print("getting synthetic responses to followup questions...")
    if (not args.restart) or (not restart_successful):
        df["followup_answer"] = ""
        df["industry_descriprion_rephrased"] = ""

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
            df.loc[batch.index, "followup_answer"] = batch.apply(
                get_followup_answer, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    df["merged_industry_desc"] = (
        df["merged_industry_desc"]
        .str.rstrip(".")
        .str.cat(df["followup_question"].str.lower(), sep=", Question: ")
        .str.cat(df["followup_answer"].str.lower(), sep=", Answer: ")
        # .str.rstrip(".").str.cat(df["followup_answer"].str.lower(), sep=", ")
    )

    # # rephrase new job description
    # for batch_id, batch in tqdm(
    #     enumerate(
    #         np.split(
    #             df,
    #             np.arange(start_batch_id * args.batch_size, len(df), args.batch_size),
    #         )
    #     )
    # ):
    #     if batch_id == 0:
    #         pass
    #     else:
    #         df.loc[batch.index, "industry_descriprion_rephrased"] = batch.apply(
    #             get_rephrased_id, axis=1
    #         )
    #         persist_results(
    #             df,
    #             metadata,
    #             args.output_folder,
    #             args.output_shortname,
    #             is_final=False,
    #             completed_batches=(batch_id + 1 + start_batch_id),
    #         )
    # df["merged_industry_desc"] = df["industry_descriprion_rephrased"]
    # df.drop(columns=["industry_descriprion_rephrased"], inplace=True)

    print("synthetic response generation is complete")

    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
