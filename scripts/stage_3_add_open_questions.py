#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script retrieves followup questions and persists the results.
It reads reloads the output from the previous stage as a DataFrame object,
retireves a follow-up question for each row, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_3_add_open_questions.py --help
```

Example Usage:

1. Ensure you have (re-)authenticated with `gcloud` for the current project.

2. Run the script:
   ```bash
   python stage_3_add_open_questions.py \
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
from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
MODEL_NAME = "gemini-2.5-flash"
MODEL_LOCATION = "europe-west1"

CODE_DIGITS = 5
CANDIDATES_LIMIT = 10

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def get_open_question(row: pd.Series) -> str:  # pylint: disable=C0103, W0613
    """Using the provided row data, call an LLM to generate an open follow-up question.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing the columns corresponding
                         to the survey responses, and the semantic search results.
    Returns: question (str).
    """
    sic_followup_object, _ = c_llm.formulate_open_question(
        industry_descr=row[MERGED_INDUSTRY_DESC_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        llm_output=row["alt_sic_candidates"],  # type: ignore
    )
    if sic_followup_object.followup is None:
        return ""
    return sic_followup_object.followup


c_llm = ClassificationLLM(MODEL_NAME, verbose=False)
print("Classification LLM loaded.")

if __name__ == "__main__":
    args = parse_args("STG3")

    df, metadata, start_batch_id, restart_successful = set_up_initial_state(
        args.restart,
        args.output_folder,
        args.output_shortname,
        args.input_parquet_file,
        args.input_metadata_json,
        args.batch_size,
        stage_id="stage_3",
    )

    print("getting followup questions ...")
    if (not args.restart) or (not restart_successful):
        df["followup_question"] = ""

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
            df.loc[batch.index, "followup_question"] = batch.apply(
                get_open_question, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    print("Followup question retrieval is complete")

    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
