#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script generates a SIC classification based on respondent's data
using RAG approach and persists the results.
It reloads the output from the previous stage as a DataFrame object,
performs classification for each row of data passed using LLM and creates
a new column in the DataFrame with this information, and then saves the results
to CSV, pickle, and JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_2_get_rag_sic_code.py --help
```

Example Usage:

1. Ensure the `gcloud` authentication ()

2. Run the script:
   ```bash
   python stage_2_get_rag_sic_code.py \
        -n my_output \
        -b 200 \
        persisted_dataframe.gz \
        persisted_metadata.json \
        output_folder
   ```
   where:
     - `-n my_output` sets the output filename prefix to "my_output".
     - `-b 200` specifies to process in batches of 200 rows, checkpointing between batches.
     - `persisted_dataframe.gz` is the pickled dataframe output at the previous stage.
     - `persisted_metadata.json` is persisted JSON metadata from the previous stage.
     - `output_folder` is the directory where results will be saved.

3. Verify outputs exist as expected:
    ```bash
   ls output_folder
   ```
   (expect to see my_output_<timestamp>.csv, my_output_<timestamp>.gz,
    and my_output_metadata_<timestamp>.json)
"""
import json
import os
from argparse import ArgumentParser as AP
from datetime import UTC, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.llm.llm import ClassificationLLM

#####################################################
# Constants:
VECTOR_STORE_URL_BASE = "http://0.0.0.0:8088"
STATUS_ENDPOINT = "/v1/sic-vector-store/status"
SEARCH_ENDPOINT = "/v1/sic-vector-store/search-index"
MODEL_NAME = "gemini-2.0-flash"

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
SHORT_LIST = "semantic_search_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def parse_args():
    """Parses command line arguments for the script."""
    parser = AP()
    parser.add_argument(
        "input_pickle_file",
        type=str,
        help="relative path to the persisted DataFrame from previous stage",
    )
    parser.add_argument(
        "input_metadata_json",
        type=str,
        help="relative path to the persisted metadata from previous stage",
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
        default="STGK",
        help="output filename prefix for easy identification (optional, default: STGK)",
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
    output_folder, output_shortname, input_pickle_file, input_metadata_json, batch_size
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
        input_pickle_file (str): The path to the (previous stage's) persisted dataframe file, used
            only if starting from scratch after failure to restart.
        input_metadata_json (str): The path to the (previous stage's) metadata JSON file,
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
            file (`input_pickle_file`) or metadata file (`input_metadata_json`)
            cannot be found.
    """
    try:
        df_persisted = pd.read_pickle(  # noqa: S301
            f"{output_folder}/intermediate_outputs/{output_shortname}.gz"
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
    except (FileNotFoundError, Exception):  # pylint: disable=W0718
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
        df_persisted = pd.read_pickle(input_pickle_file)  # noqa: S301
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


def get_rag_response(row: pd.Series) -> dict[str, Any]:  # pylint: disable=C0103, W0613
    """Generates sa_rag_response dictionary using the provided row data.
    Intended for use as a `.apply()` operation to create a new colum in
    a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        result (doct[str, Any]): a dictionary with final_sic_code,
        sic_candidates, and sic_descriptive for specified row.
    """
    sa_rag_response = uni_chat.sa_rag_sic_code(  # pylint: disable=E0606
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        industry_descr=row[INDUSTRY_DESCR_COL],
        candidates_limit=10,
        short_list=row[SHORT_LIST],
    )

    result = {
        "final_sic_code": sa_rag_response[0].sic_code,
        "sic_descriptive": sa_rag_response[0].sic_descriptive,
        "sic_candidates": [
            {"sic_code": i.sic_code,
            "likelihood": i.likelihood,
            "sic_description": i.sic_descriptive}
            for i in sa_rag_response[0].sic_candidates
        ],
    }
    return result


def get_final_sic_code(row: pd.Series) -> str:
    """Generator funciton to access final_sic_code for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: a final_sic_code for the row.
    """
    return row["sa_rag_sic_response"]["final_sic_code"]


def get_sic_candidates(row: pd.Series) -> str:
    """Generator funciton to access sic_candidates for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: A list of possible sic_code alternatives.
    """
    return row["sa_rag_sic_response"]["sic_candidates"]


def get_sic_descriptive(row: pd.Series) -> str:
    """Generator funciton to access sic_descriptive for the specified row.

    Args:
        row (pd.Series): A row from the input DataFrame containing
        semantic_search_results column.

    Returns:
        str: A list of SIC descriptions.
    """
    return row["sa_rag_sic_response"]["sic_descriptive"]


def persist_results(  # noqa: PLR0913 # pylint: disable=R0913, R0917
    df_with_search: pd.DataFrame,
    metadata: dict,
    output_folder: str,
    output_shortname: str,
    is_final: Optional[bool] = False,
    completed_batches: Optional[int] = 0,
):
    """Persists the results DataFrame to CSV, pickle, and saves metadata to JSON.

    Args:
        df_with_search (pd.DataFrame): The DataFrame containing the results to be persisted.
        metadata (dict): The additional metadata surrounding this processing job.
        output_folder (str): The path to the output folder where the files will be saved.
        output_shortname (str): The prefix given to each file to be saved.
        is_final (bool): Mark the output as the final output and timestamp filenames.
                         Optional, default False.
        completed_batches (int): Specify the number of completed batches being saved.
                                 Optional, default 0.
    Returns: None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if is_final:
        time_suffix = datetime.now(UTC).strftime("%Y_%m_%d_%H")
        print("Saving results to CSV...")
        df_with_search.to_csv(f"{output_folder}/{output_shortname}_{time_suffix}.csv")
        print("Saving results to pickle...")
        df_with_search.to_pickle(f"{output_folder}/{output_shortname}_{time_suffix}.gz")
        print("Saving setup metadata to JSON...")
        with open(
            f"{output_folder}/{output_shortname}_metadata_{time_suffix}.json",
            "w",
            encoding="utf8",
        ) as output_meta:
            json.dump(metadata, output_meta)

    else:
        output_folder = f"{output_folder}/intermediate_outputs"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df_with_search.to_pickle(f"{output_folder}/{output_shortname}.gz")
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

    RESTART_SUCCESS = True

    if args.restart:
        try:
            df, METADATA, checkpoint_info, RESTART_SUCCESS = try_to_restart(
                args.output_folder,
                args.output_shortname,
                args.input_pickle_file,
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
        METADATA["start_unix_timestamp"] = datetime.now(UTC).timestamp()
        METADATA["batch_size"] = args.batch_size
        df = pd.read_pickle(args.input_pickle_file)  # noqa: S301
        print("Input loaded")

    print("Running RAG SIC allocation...")
    if (not args.restart) or (not RESTART_SUCCESS):
        df["sa_rag_sic_response"] = {
            "final_sic_code": "",
            "sic_candidates": [],
            "sic_descriptive": "",
        }
        df["final_sic_code"] = ""
        df["sic_candidates"] = np.empty((len(df), 0)).tolist()
        df["sic_descriptive"] = ""
        START_BATCH_ID = 0
    else:
        START_BATCH_ID = checkpoint_info["completed_batches"]

    uni_chat = ClassificationLLM(model_name=MODEL_NAME)
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
            batch.apply(print)
            batch.loc[batch.index, "sa_rag_sic_response"] = batch.apply(
                get_rag_response, axis=1
            )
            df.loc[batch.index, "final_sic_code"] = batch.apply(
                get_final_sic_code, axis=1
            )
            df.loc[batch.index, "sic_candidates"] = batch.apply(
                get_sic_candidates, axis=1
            )
            df.loc[batch.index, "sic_descriptive"] = batch.apply(
                get_sic_descriptive, axis=1
            )
            persist_results(
                df,
                METADATA,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1),
            )
    df.drop("sa_rag_sic_response", axis=1, inplace=True)
    print("RAG SIC allocation is complete")

    print("persisting results...")
    persist_results(
        df, METADATA, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
