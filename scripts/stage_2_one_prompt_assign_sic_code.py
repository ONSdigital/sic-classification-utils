#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script generates a SIC classification based on respondent's data
using RAG approach and persists the results.
It reloads the output from the previous stage as a DataFrame object,
performs classification for each row of data passed using LLM and creates
a new column in the DataFrame with this information, and then saves the results
to CSV, parquet, and JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_2_assign_sic_code_one_prompt.py --help
```

Example Usage:

1. Ensure the `gcloud` authentication ()

2. Run the script:
   ```bash
   python stage_2_assign_sic_code_one_prompt.py \
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
   (expect to see my_output.csv, my_output.parquet and my_output_metadata.json)
"""
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
VECTOR_STORE_URL_BASE = "http://0.0.0.0:8088"
STATUS_ENDPOINT = "/v1/sic-vector-store/status"
SEARCH_ENDPOINT = "/v1/sic-vector-store/search-index"
MODEL_NAME = "gemini-2.5-flash"

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
SHORT_LIST = "semantic_search_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def get_rag_response(row: pd.Series) -> dict[str, Any]:  # pylint: disable=C0103, W0613
    """Generates sa_rag_response dictionary using the provided row data.
    Intended for use as a `.apply()` operation to create a new colum in
    a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        result (doct[str, Any]): a dictionary with initial_sic,
        and alt_candidates for specified row.
    """
    sa_rag_response = uni_chat.sa_rag_sic_code(  # pylint: disable=E0606
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        industry_descr=row[INDUSTRY_DESCR_COL],
        candidates_limit=10,
        short_list=row[SHORT_LIST],
    )

    result = {
        "initial_sic": sa_rag_response[0].sic_code,
        "followup": sa_rag_response[0].followup,
        "unambiguously_codable": sa_rag_response[0].sic_code not in ("", None),
        "alt_candidates": [
            {
                "code": i.sic_code,
                "title": i.sic_descriptive,
            }
            for i in sa_rag_response[0].sic_candidates
        ],
    }
    return result


def get_codable_status(row: pd.Series) -> str:
    """Generator funciton to access initial_sic for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: a initial_sic for the row.
    """
    return row["sa_rag_sic_response"]["unambiguously_codable"]


def get_followup_question(row: pd.Series) -> str:
    """Generator funciton to access initial_sic for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: a initial_sic for the row.
    """
    return row["sa_rag_sic_response"]["followup"]


def get_initial_sic(row: pd.Series) -> str:
    """Generator funciton to access initial_sic for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: a initial_sic for the row.
    """
    return row["sa_rag_sic_response"]["initial_sic"]


def get_alt_sic_candidates(row: pd.Series) -> str:
    """Generator funciton to access alt_candidates for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: A list of possible sic_code alternatives.
    """
    return row["sa_rag_sic_response"]["alt_candidates"]


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

    print("Running RAG SIC allocation...")
    if (not args.restart) or (not restart_successful):
        df["sa_rag_sic_response"] = {
            "unambiguously_codable": False,
            "initial_sic": "",
            "alt_candidates": [],
            "followup": "",
        }
        df["unambiguously_codable"] = False
        df["initial_code"] = ""
        df["alt_sic_candidates"] = np.empty((len(df), 0)).tolist()
        df["followup_question"] = ""

    uni_chat = ClassificationLLM(model_name=MODEL_NAME)
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
            batch.loc[batch.index, "sa_rag_sic_response"] = batch.apply(
                get_rag_response, axis=1
            )
            df.loc[batch.index, "unambiguously_codable"] = batch.apply(
                get_codable_status, axis=1
            )
            df.loc[batch.index, "initial_code"] = batch.apply(get_initial_sic, axis=1)
            df.loc[batch.index, "alt_sic_candidates"] = batch.apply(
                get_alt_sic_candidates, axis=1
            )
            df.loc[batch.index, "followup_question"] = batch.apply(
                get_followup_question, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )
    df.drop("sa_rag_sic_response", axis=1, inplace=True)
    print("RAG SIC allocation is complete")

    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
