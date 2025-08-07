#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script performs X on a dataset and persists the results.
It reads reloads the output from the previous stage as a DataFrame object,
performs X for each row, creates a new column in the DataFrame with this
information, and then saves the results to CSV, pickle, and JSON metadata
files in a user-specified output folder.

The script requires Y.

Clarification On Script Arguments:

```bash
python stage_k_template.py --help
```

Example Usage:

1. Ensure the vector store is running at http://0.0.0.0:8088.

2. Run the script:
   ```bash
   python stage_k_template.py \
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

-----------------------------------------------------------------------------------------
What to change, to adapt it to a given stage's requirements:

* update `check_y` function to check whatever is required for your new stage.
  (e.g. connection to LLM established, or Vector Store ready)
* update `get_x()` function to achieve whatever is required for your new column.
* create second `get_x2()` function if more than one new column is required.
* update the `if __name__=="__main__" block to use the new function names, and
  repeat the creation of the empty new column and batch.apply() if you are adding
  more thn one new column.
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
MODEL_NAME = "gemini-2.0-flash"
MODEL_LOCATION = "eu-west9"

CODE_DIGITS = 5
CANDIDATES_LIMIT = 10

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


def check_y():
    """Checks if Y.
    Raises an exception if NOT Y.
    Exits silently if Y.
    """
    try:
        pass
    except Exception:
        print("Y was not met")
        raise


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
        industry_descr=row[INDUSTRY_DESCR_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        sic_candidates=short_list,
    )

    result = {
        "unambiguously_codable": sa_response[0].codable,
        "code": sa_response[0].class_code,
        "alt_candidates": [i.class_code for i in sa_response[0].alt_candidates],
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


def get_alt_sic_candidates(row: pd.Series) -> list:
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


c_llm = ClassificationLLM(MODEL_NAME, verbose=False)

if __name__ == "__main__":
    args = parse_args()
    # check_y()
    # print("Requirement Y is met")
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
        METADATA["stage2_classification_llm_model"] = MODEL_NAME
        METADATA["stage2_classification_llm_location"] = MODEL_LOCATION
        df = pd.read_pickle(args.input_pickle_file)  # noqa: S301
        print("Input loaded")

    print("running unamibuous codability analysis...")
    if (not args.restart) or (not RESTART_SUCCESS):
        df["intermediate_unambig_results"] = {
            "unambiguously_codable": False,
            "code": "",
            "alt_candidates": [],
        }
        df["unambiguously_codable"] = False
        df["initial_code"] = ""
        df["alt_sic_candidates"] = np.empty((len(df), 0)).tolist()
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
                METADATA,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1),
            )

    print("unambiguous coding analysis is complete")
    print("deleting temporary DataFrame column...")
    df = df.drop("intermediate_unambig_results", axis=1)
    print("persisting results...")
    persist_results(
        df, METADATA, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
