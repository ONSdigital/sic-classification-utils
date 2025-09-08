#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script analyzes a dataset to determine if each record is
"unambiguously codable" for a Standard Industrial Classification (SIC) code.

It reloads the output from the previous stage as a DataFrame object, uses a
Large Language Model (LLM) to assess codability for each row, and adds new
columns for the codability status, an initial SIC code (if one can be
assigned), and a list of alternative SIC candidates. The results are then saved
to CSV, parquet, and JSON metadata files in a user-specified output folder.

The script requires a configured connection to a compatible LLM.

Clarification On Script Arguments:

```bash
python stage_2_add_unambiguously_codable_status.py --help
```

Example Usage:

1. Ensure you have run `gcloud` (re-)authentication for the current project.

2. Run the script:
   ```bash
   python stage_2_add_unambiguously_codable_status.py \
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
from typing import Any

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
CANDIDATES_LIMIT = 10

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def get_unambiguous_sic(
    row: pd.Series,
) -> dict[str, Any]:  # pylint: disable=C0103, W0613
    """Performs codability analysis using the provided row data.
    If unambiguously codable, provides an initial SIC code.
    Otherwise, provides a list of SIC candidates.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing <required columns>.

    Returns:
        result (dict): the codability, assigned code (or None), and the alternatice SIC candidates.
    """
    short_list = c_llm._prompt_candidate_list(  # pylint: disable=W0212
        row["semantic_search_results"],
        code_digits=CODE_DIGITS,
        candidates_limit=CANDIDATES_LIMIT,
    )

    sa_response = c_llm.unambiguous_sic_code(
        industry_descr=row[MERGED_INDUSTRY_DESC_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        sic_candidates=short_list,
    )

    result = {
        "unambiguously_codable": sa_response[0].codable,
        "code": sa_response[0].class_code,
        "alt_candidates": [
            {"code": i.class_code, "title": i.class_descriptive}
            for i in sa_response[0].alt_candidates
        ],
    }
    return result


def get_unambiguous_status(row: pd.Series) -> bool:
    """Gets the codability status from the intermediate results.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        codability status (bool).
    """
    if row["intermediate_unambig_results"]["unambiguously_codable"] is not None:
        return row["intermediate_unambig_results"]["unambiguously_codable"]
    return False


def get_initial_sic_code(row: pd.Series) -> str:
    """Gets the assigned SIC code from the intermediate results, if possible.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        initial_code (str): the assigned SIC code, or an empty string if not available.
    """
    if row["intermediate_unambig_results"]["code"] is not None:
        return row["intermediate_unambig_results"]["code"]
    return ""


def get_alt_sic_candidates(row: pd.Series) -> list[dict]:
    """Gets the alternative SIC candidates from the intermediate results, if possible.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        alt_sic_codes (list): the alternative SIC candidates, or an empty list if not available.
    """
    if row["intermediate_unambig_results"]["alt_candidates"] is not None:
        return row["intermediate_unambig_results"]["alt_candidates"]
    return []


c_llm = ClassificationLLM(MODEL_NAME, verbose=False)
print("Classification LLM loaded.")

if __name__ == "__main__":
    args = parse_args("STG2")

    df, metadata, start_batch_id, restart_successful = set_up_initial_state(
        args.restart,
        args.output_folder,
        args.output_shortname,
        args.input_parquet_file,
        args.input_metadata_json,
        args.batch_size,
        stage_id="stage_2",
    )

    print("running unamibuous codability analysis...")
    if (not args.restart) or (not restart_successful):
        df["intermediate_unambig_results"] = {
            "unambiguously_codable": False,
            "code": "",
            "alt_candidates": [],
        }
        df["unambiguously_codable"] = False
        df["initial_code"] = ""
        df["alt_sic_candidates"] = np.empty((len(df), 0)).tolist()

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
                get_unambiguous_sic, axis=1
            )
            df.loc[batch.index, "unambiguously_codable"] = batch.apply(
                get_unambiguous_status, axis=1
            )
            df.loc[batch.index, "initial_code"] = batch.apply(
                get_initial_sic_code, axis=1
            )
            df.loc[batch.index, "alt_sic_candidates"] = batch.apply(
                get_alt_sic_candidates, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    print("unambiguous coding analysis is complete")
    print("deleting temporary DataFrame column...")
    df = df.drop("intermediate_unambig_results", axis=1)
    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
