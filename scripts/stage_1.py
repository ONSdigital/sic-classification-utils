#!/usr/bin/env python3
"""This script performs semantic search on a dataset using a vector store
and persists the results. It reads in a CSV file as a DataFrame object,
interacts with a vector store to obtain semantic search results for each
row, creates a new column in the DataFrame with this information, and
then saves the results to CSV, pickle, and JSON metadata files in a
user-specified output folder.

The script requires a running vector store service.

Clarification On Script Arguments:

```bash
python stage_1.py --help
```

Example Usage:

1. Ensure the vector store is running at http://0.0.0.0:8088.

2. Run the script:
   ```bash
   python stage_1.py input.csv output_folder -n my_output
   ```
   where:
     - `input.csv` is the input CSV file.
     - `output_folder` is the directory where results will be saved.
     - `-n my_output` sets the output filename prefix to "my_output".

3. Verify outputs exist as expected:
    ```bash
   ls output_folder
   ```
   (expect to see my_output.csv, my_output.gz, and my_output_metadata.json)

"""
import json
import os
from argparse import ArgumentParser as AP

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException

#####################################################
# Constants:
METADATA: dict = {}

VECTOR_STORE_URL_BASE = "http://0.0.0.0:8088"
STATUS_ENDPOINT = "/v1/sic-vector-store/status"
SEARCH_ENDPOINT = "/v1/sic-vector-store/search-index"

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
#####################################################


def parse_args():
    """Parses command line arguments for the script."""
    parser = AP()
    parser.add_argument(
        "input_file", type=str, help="relative path to the input CSV dataset"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="relative path to the output folder location (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--output_shortname",
        "-n",
        type=str,
        default="STG1",
        help="output filename prefix for easy identification (optional, default: STG1)",
    )
    return parser.parse_args()


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
        row (pd.Series): A row from the input DataFrame containing industry description,
                         job title, and job description.
    Returns: A list of dictionaries containing the code and distance for each search result.
    """
    payload = {
        "industry_descr": row[INDUSTRY_DESCR_COL],
        "job_title": row[JOB_TITLE_COL],
        "job_description": row[JOB_DESCRIPTION_COL],
    }

    # Prevent undefined behaviour from VectorStore by
    # sanitising non-string inputs (e.g. None)
    for k, v in payload.items():
        if not isinstance(v, str):
            payload[k] = ""

    response = requests.post(
        f"{VECTOR_STORE_URL_BASE}{SEARCH_ENDPOINT}", json=payload, timeout=10
    )
    response.raise_for_status()
    response_json = response.json()
    try:
        results = response_json["results"]
    except (KeyError, AttributeError):
        print("results key missing from JSON response from vector store", response_json)
        raise

    reduced_results = [{"code": r["code"], "distance": r["distance"]} for r in results]
    return reduced_results


def persist_results(
    df_with_search: pd.DataFrame, output_folder: str, output_shortname: str
):
    """Persists the results DataFrame to CSV, pickle, and saves metadata to JSON.

    Args:
        df_with_search (pd.DataFrame): The DataFrame containing the results to be persisted.
        output_folder (str): The path to the output folder where the files will be saved.
        output_shortname (str): The prefix given to each file to be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Saving results to CSV...")
    df_with_search.to_csv(f"{output_folder}/{output_shortname}.csv")
    print("Saving results to pickle...")
    df_with_search.to_pickle(f"{output_folder}/{output_shortname}.gz")
    print("Saving setup metadata to JSON...")
    with open(f"{output_folder}/{output_shortname}_metadata.json", "w", encoding="utf8") as f:
        json.dump(METADATA, f)


if __name__ == "__main__":
    args = parse_args()

    check_vector_store_ready()
    print("Vector store is ready")

    df = pd.read_csv(args.input_file)
    print("Input loaded")

    print("running semantic search...")
    df["semantic_search_results"] = df.apply(get_semantic_search_results, axis=1)
    print("semantic search complete")

    print("persisting results...")
    persist_results(df, args.output_folder, args.output_shortname)
    print("Done!")
