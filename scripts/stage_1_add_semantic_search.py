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
        input.csv \
        initial_metadata.json \
        output_folder
   ```
   where:
     - `-n my_output` sets the output filename prefix to "my_output".
     - `-b 200` specifies to process in batches of 200 rows, checkpointing between batches.
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
import json
import os
from argparse import ArgumentParser as AP
from datetime import UTC, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException
from tqdm import tqdm

#####################################################
# Constants:
VECTOR_STORE_URL_BASE = "http://0.0.0.0:8088"
STATUS_ENDPOINT = "/v1/sic-vector-store/status"
SEARCH_ENDPOINT = "/v1/sic-vector-store/search-index"

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def parse_args():
    """Parses command line arguments for the script."""
    parser = AP()
    parser.add_argument(
        "input_data_file", type=str, help="relative path to the input CSV dataset"
    )
    parser.add_argument(
        "input_metadata_json",
        type=str,
        help="relative path to the initial JSON metadata",
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
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=200,
        help="save the output every X rows, as a checkpoint that can be used to restart the "
        "processing job if needed (optional, default: 200)",
    )
    parser.add_argument(
        "--restart",
        "-r",
        action="store_true",
        default=False,
        help="try to restart a processing job (optional flag)",
    )
    return parser.parse_args()


def try_to_restart(
    output_folder, output_shortname, input_data_file, input_metadata_json, batch_size
):
    """Attempts to restart a processing job by loading checkpoint data.

    This function tries to load a previously saved DataFrame, metadata, and
    checkpoint information from an intermediate output directory. If these files
    are found and loaded successfully, it indicates a successful restart.

    If any file is not found or another exception occurs during loading, it
    reverts to starting the process from scratch. In this case, it loads the
    initial input data and metadata, and prepares new checkpoint information.

    Args:
        output_folder (str): The path to the specified output folder.
        output_shortname (str): The prefix for the output filenames.
        input_data_file (str): The path to the original input CSV file, used
            only if starting from scratch after failure to restart.
        input_metadata_json (str): The path to the initial metadata JSON file,
            used only if starting from scratch after failure to restart.
        batch_size (int): The size of processing batches, used only if starting
            from scratch after failure to restart.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The loaded or newly created DataFrame.
            - dict: The loaded or newly created metadata dictionary.
            - dict: The loaded or newly created checkpoint information.
            - bool: True if the restart from a checkpoint was successful,
              False otherwise.

    Raises:
        FileNotFoundError: If starting from scratch and the initial data
            file (`input_data_file`) or metadata file (`input_metadata_json`)
            cannot be found.
    """
    try:
        df_persisted = pd.read_parquet(
            f"{output_folder}/intermediate_outputs/{output_shortname}.parquet"
        )
        with open(
            f"{output_folder}/intermediate_outputs/{output_shortname}_checkpoint_info.json",
            encoding="utf8",
        ) as checkpoint:
            checkpoint_info_persisted = json.load(checkpoint)
        with open(
            f"{output_folder}/intermediate_outputs/{output_shortname}_metadata.json",
            encoding="utf8",
        ) as meta:
            metadata_persisted = json.load(meta)
        restart_successful = True
        print("Partially-processed data re-loaded succesfully")
        return (
            df_persisted,
            metadata_persisted,
            checkpoint_info_persisted,
            restart_successful,
        )
    except (FileNotFoundError, Exception):  # pylint: disable=W0718:
        print("Could not re-load checkpointed results, restarting from scratch...")
        restart_successful = False
        try:
            with open(input_metadata_json, encoding="utf-8") as input_meta:
                metadata_persisted = json.load(input_meta)
        except FileNotFoundError:
            print(f"Could not find metadata file {input_metadata_json}")
            raise
        metadata_persisted["start_unix_timestamp"] = datetime.now(UTC).timestamp()
        metadata_persisted["batch_size"] = batch_size
        df_persisted = pd.read_csv(input_data_file)
        checkpoint_info_persisted = {
            "completed_batches": 0,
            "batch_size": batch_size,
        }
        return (
            df_persisted,
            metadata_persisted,
            checkpoint_info_persisted,
            restart_successful,
        )


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
    Returns: A list of dictionaries containing the title, code and distance for each search
    result.
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


def persist_results(  # noqa: PLR0913 # pylint: disable=R0913, R0917
    df_with_search: pd.DataFrame,
    metadata: dict,
    output_folder: str,
    output_shortname: str,
    is_final: Optional[bool] = False,
    completed_batches: Optional[int] = 0,
):
    """Persists the results DataFrame to CSV, parquet, and saves metadata to JSON.

    Args:
        df_with_search (pd.DataFrame): The DataFrame containing the results to be persisted.
        metadata (dict): The additional metadata surrounding this processing job.
        output_folder (str): The path to the output folder where the files will be saved.
        output_shortname (str): The prefix given to each file to be saved.
        is_final (bool): Mark the output as the final output.
                         Optional, default False.
        completed_batches (int): Specify the number of completed batches being saved.
                                 Optional, default 0.
    Returns: None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if is_final:
        print("Saving results to CSV...")
        df_with_search.to_csv(f"{output_folder}/{output_shortname}.csv", index=False)
        print("Saving results to parquet...")
        df_with_search.to_parquet(
            f"{output_folder}/{output_shortname}.parquet", index=False
        )
        print("Saving setup metadata to JSON...")
        with open(
            f"{output_folder}/{output_shortname}_metadata.json",
            "w",
            encoding="utf8",
        ) as output_meta:
            json.dump(metadata, output_meta)

    else:
        output_folder = f"{output_folder}/intermediate_outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df_with_search.to_parquet(
            f"{output_folder}/{output_shortname}.parquet", index=False
        )
        with open(
            f"{output_folder}/{output_shortname}_metadata.json", "w", encoding="utf8"
        ) as temp_meta:
            json.dump(metadata, temp_meta)
        with open(
            f"{output_folder}/{output_shortname}_checkpoint_info.json",
            "w",
            encoding="utf8",
        ) as checkpoint:
            json.dump(
                {
                    "completed_batches": completed_batches,
                    "batch_size": metadata["batch_size"],
                },
                checkpoint,
            )


if __name__ == "__main__":
    args = parse_args()

    check_vector_store_ready()
    print("Vector store is ready")
    RESTART_SUCCESS = True

    if args.restart:
        try:
            df, METADATA, checkpoint_info, RESTART_SUCCESS = try_to_restart(
                args.output_folder,
                args.output_shortname,
                args.input_data_file,
                args.input_metadata_json,
                args.batch_size,
            )
        except Exception:
            print(
                "Could not load persisted output, and ran into an issue starting from scratch"
            )
            raise
    else:
        try:
            with open(args.input_metadata_json, encoding="utf-8") as f:
                METADATA = json.load(f)
        except FileNotFoundError:
            print(f"Could not find metadata file {args.input_metadata_json}")
            raise
        METADATA["stage_1_start_timestamp"] = datetime.now(UTC).timestamp()
        METADATA["stage_1_start_time_readable"] = datetime.now(UTC).strftime(
            "%Y/%m/%d_%H:%M:%S"
        )
        METADATA["batch_size"] = args.batch_size
        df = pd.read_csv(args.input_data_file)
        print("Input loaded")

    print("running semantic search...")
    if (not args.restart) or (not RESTART_SUCCESS):
        df["semantic_search_results"] = np.empty((len(df), 0)).tolist()
        START_BATCH_ID = 0
    else:
        START_BATCH_ID = checkpoint_info["completed_batches"]

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(START_BATCH_ID * args.batch_size, len(df), args.batch_size),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            df.loc[batch.index, "semantic_search_results"] = batch.apply(
                get_semantic_search_results, axis=1
            )
            persist_results(
                df,
                METADATA,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1),
            )

    print("semantic search complete")

    print("persisting results...")
    persist_results(
        df, METADATA, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
