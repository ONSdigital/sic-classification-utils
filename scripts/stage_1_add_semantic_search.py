#!/usr/bin/env python3
"""This script performs semantic search on a dataset using a vector store
and persists the results. It reads in a CSV file as a DataFrame object,
interacts with a vector store to obtain semantic search results for each
row, creates a new column in the DataFrame with this information, and
then saves the results to CSV, parquet, and JSON metadata files in a
user-specified output folder.

The script requires a running vector store service.

Clarification On Script Arguments:

```bash
python stage_1_add_semantic_search.py --help
```

Example Usage:

1. Ensure the vector store is running at http://0.0.0.0:8088.

2. Run the script:
   ```bash
   python stage_1_add_semantic_search.py \
        -n my_output \
        -b 200 \
        -s \
        input.csv \
        initial_metadata.json \
        output_folder
   ```
   where:
     - `-n my_output` sets the output filename prefix to "my_output".
     - `-b 200` specifies to process in batches of 200 rows, checkpointing between batches.
     - `-s` specifies the second run of the stage. If `-s` absent - first run.
     - `input.csv` is the input CSV file.
     - `initial_metadata.json` is the JSON file containing the initial processing metadata.
       An example is shown below.
     - `output_folder` is the directory where results will be saved.

3. Verify outputs exist as expected:
    ```bash
   ls output_folder
   ```
   (expect to see my_output.csv, my_output.parquet, and my_output_metadata.json)

--------------
Example expected Contents of initial_metadata.json:
{
    "original_dataset_name": "DSC_Rep_Sample.csv",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "llm_model_name": "gemini-1.0-pro",
    "llm_location": "eu-west2",
    "sic_index_file": "uksic2007indexeswithaddendumdecember2022.xlsx",
    "sic_structure_file": "publisheduksicsummaryofstructureworksheet.xlsx",
    "sic_condensed_file": "sic_2d_condensed.txt",
    "matches": 20,
    "sic_index_size": 34663,
    "runner_initials": "LR"
}

"""
from re import sub as regex_sub

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException
from tqdm import tqdm

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

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
SELF_EMPLOYED_DESC_COL = "sic2007_self_employed"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"  # created in this script
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def clean_text(text: str) -> str:
    """Cleans a text string by removing newlines, converting arbitrary
    whitespace to a single space, removing -9's and standardizing case.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    if isinstance(text, float):
        text = ""
    text = text.replace("\n", " ")
    text = regex_sub(r"\s+", " ", text)
    text = text.lower()
    text = text.capitalize()
    return text


def make_merged_industry_desc(row: pd.Series) -> str:
    """Merges the main industry description column with the self-employed description column.

    Args:
        row (pd.Series): A row from the input DataFrame containing industry description,
                         self employed description.

    Returns:
        description (str): The merged descriptions.
    """
    ind_desc = (
        row[INDUSTRY_DESCR_COL] if isinstance(row[INDUSTRY_DESCR_COL], str) else ""
    )
    self_emp_desc = (
        row[SELF_EMPLOYED_DESC_COL]
        if isinstance(row[SELF_EMPLOYED_DESC_COL], str)
        else ""
    )

    return f"{ind_desc}{self_emp_desc}"


def clean_text_industry(text: str) -> str:
    """Cleans a text string by removing newlines, converting arbitrary
    whitespace to a single space, removing -9's and standardizing case.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    text = text.replace("\n", " ")
    text = text.replace("-9", "")
    text = regex_sub(r"\s+", " ", text)
    text = text.lower()
    text = text.capitalize()
    return text


def check_vector_store_ready():
    """Checks if the vector store is ready by querying its status endpoint.
    Raises an exception if the service is unavailable or not ready.
    Exits silently if the vector store is ready.
    """
    try:
        response = requests.get(f"{VECTOR_STORE_URL_BASE}{STATUS_ENDPOINT}", timeout=10)
        response.raise_for_status()
        if response.json()["status"] != "ready":
            raise OSError("The vector store is still loading, re-try in a few minutes")
    except (HTTPError, ConnectionError, RequestException):
        print("Bad response from locally-running vector store")
        raise
    except Exception:
        print("Could not interact with locally-running vector store")
        raise


def get_semantic_search_results(row: pd.Series) -> list[dict]:
    """Performs a semantic search using the provided row data.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing the merged industry
                         description, job title, and job description.
    Returns: A list of dictionaries containing the title, code and distance for each search
    result.
    """
    payload = {
        "industry_descr": row[MERGED_INDUSTRY_DESC_COL],
        "job_title": row[JOB_TITLE_COL],
        "job_description": row[JOB_DESCRIPTION_COL],
    }

    # Prevent undefined behaviour from VectorStore by
    # sanitising non-string inputs (e.g. None)
    for k, v in payload.items():
        if not isinstance(v, str):
            payload[k] = ""

    response = requests.post(
        f"{VECTOR_STORE_URL_BASE}{SEARCH_ENDPOINT}", json=payload, timeout=25
    )
    response.raise_for_status()
    response_json = response.json()
    try:
        results = response_json["results"]
    except (KeyError, AttributeError):
        print("results key missing from JSON response from vector store", response_json)
        raise

    reduced_results = [
        {"title": r["title"], "code": r["code"], "distance": r["distance"]}
        for r in results
    ]
    return reduced_results


if __name__ == "__main__":
    args = parse_args("STG1")

    check_vector_store_ready()
    print("Vector store is ready")

    df, metadata, start_batch_id, restart_successful, second_run_variables = (
        set_up_initial_state(
            args.restart,
            args.second_run,
            args.output_folder,
            args.output_shortname,
            args.input_parquet_file,
            args.input_metadata_json,
            args.batch_size,
            stage_id="stage_1",
            is_stage_1=True,
        )
    )

    semantic_search = (  # pylint: disable=C0103
        "second_semantic_search_results"
        if second_run_variables
        else "semantic_search_results"
    )

    # Make a merged industry description column:
    if "merged_industry_desc" not in df:
        df[MERGED_INDUSTRY_DESC_COL] = df.apply(make_merged_industry_desc, axis=1)
    # Clean the Survey Response columns:
    df[INDUSTRY_DESCR_COL] = df[INDUSTRY_DESCR_COL].apply(clean_text)
    df[JOB_DESCRIPTION_COL] = df[JOB_DESCRIPTION_COL].apply(clean_text)
    df[JOB_TITLE_COL] = df[JOB_TITLE_COL].apply(clean_text)
    df[MERGED_INDUSTRY_DESC_COL] = df[MERGED_INDUSTRY_DESC_COL].apply(
        clean_text_industry
    )
    print("Input loaded")

    print("running semantic search...")
    if (not args.restart) or (not restart_successful):
        df[semantic_search] = np.empty((len(df), 0)).tolist()

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
            df.loc[batch.index, semantic_search] = batch.apply(
                get_semantic_search_results, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    print("semantic search complete")
    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
