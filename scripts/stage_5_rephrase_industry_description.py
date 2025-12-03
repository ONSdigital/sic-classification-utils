#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script rephrases industry description with followup questions and follow up answers,
and persists the results. It reads reloads the output from the previous stage as a DataFrame
object, rephrases each row, overwrites 'merged_industry_desc' column in the DataFrame
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

MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
FOLLOWUP_QUESTION = "followup_question"
FOLLOWUP_ANSWER = "followup_answer"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


# pylint: disable=C0116 # the docstring is below


def get_rephrased_id(row: pd.Series, method: str = "concatenate") -> str:
    """Rephrase industry description with follow up question and follow up answer as a label.

    Args:
        row (pd.Series): A row from the input DataFrame containing the survey responses,
                         and the followup question.
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
        return SR.rephrase_question_and_id(
            row[MERGED_INDUSTRY_DESC_COL],
            row[FOLLOWUP_QUESTION],
            row[FOLLOWUP_ANSWER],
        )[0]
    return row[MERGED_INDUSTRY_DESC_COL]


SR = SyntheticResponder(persona=None, get_question_function=None, model_name=MODEL_NAME)

if __name__ == "__main__":
    args = parse_args("STG4")

    job_description_rephrased = []
    df, metadata, start_batch_id, restart_successful, second_run_variables = (
        set_up_initial_state(
            args.restart,
            args.second_run,
            args.output_folder,
            args.output_shortname,
            args.input_parquet_file,
            args.input_metadata_json,
            args.batch_size,
            stage_id="stage_5",
        )
    )
    print(
        "rephrasing industry description with follow up questions and followup answers..."
    )
    if (not args.restart) or (not restart_successful):
        df["industry_description_rephrased"] = ""

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
            df.loc[batch.index, "industry_description_rephrased"] = batch.apply(
                get_rephrased_id, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )
    df[MERGED_INDUSTRY_DESC_COL] = df["industry_description_rephrased"]
    df.drop(columns=["industry_description_rephrased"], inplace=True)

    print("synthetic response generation is complete")

    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
