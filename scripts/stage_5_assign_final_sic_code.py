#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script assigns final SIC code using intermediate outputs from previous stages.
It reloads the output from the previous stage as a DataFrame object, creates a new column in the
DataFrame with this information, and then saves the results to CSV, pickle,
and JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_5_assign_final_sic_code.py --help
```

Example Usage:

1. Ensure you have (re-)authenticated with `gcloud` for the current project.

2. Run the script:
   ```bash
   python stage_5_assign_final_sic_code.py \
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
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.llm.llm import ClassificationLLM

#####################################################
# Constants:
MODEL_NAME = "gemini-2.0-flash"
MODEL_LOCATION = "europe-west9"

CODE_DIGITS = 5

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
SIC_CANDIDATES_COL = "alt_sic_candidates"
OPEN_QUESTION_COL = "followup_question"
ANSWER_TO_OPEN_QUESTION_COL = "followup_answer"
CLOSED_QUESTION = ""
ANSWER_TO_CLOSED_QUESTION = ""

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


def assign_final_sic_code(row: pd.Series) -> dict:  # pylint: disable=C0103, W0613
    """Using the provided row data, call an LLM to assign a final SIC code.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing the columns corresponding
                         to the survey responses, sic candidates, and follow-up questions and
                         corresponding answers.

    Returns:
        result (dict): final codability, assigned 5 digit SIC code or a higher level code if
            final sic cannot be assigned unambiguously.
    """
    sa_final_sic = c_llm.final_sic_code(
        industry_descr=row[INDUSTRY_DESCR_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        sic_candidates=row[SIC_CANDIDATES_COL],
        open_question=row[OPEN_QUESTION_COL],
        answer_to_open_question=row[ANSWER_TO_OPEN_QUESTION_COL],
        closed_question=CLOSED_QUESTION,
        answer_to_closed_question=ANSWER_TO_CLOSED_QUESTION,
    )
    result = {
        "unambiguously_codable_final": sa_final_sic[0].codable,
        "final_sic": sa_final_sic[0].unambiguous_code,
        "higher_level_final_sic": sa_final_sic[0].higher_level_code,
    }
    return result


def get_unambiguous_status_final(row: pd.Series) -> bool:
    """Gets the final codability status from the intermediate results.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        final codability status (bool).
    """
    if row["intermediate_unambig_results"]["unambiguously_codable_final"] is not None:
        return row["intermediate_unambig_results"]["unambiguously_codable_final"]
    return False


def get_final_sic_code(row: pd.Series) -> str:
    """Gets the assigned SIC code from the intermediate results, if possible.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        final_code (str): the assigned SIC code.
    """
    if row["intermediate_unambig_results"]["final_sic"] is not None:
        return row["intermediate_unambig_results"]["final_sic"]
    return ""


def get_higher_level_sic_code(row: pd.Series) -> str:
    """Gets the higher level SIC code from the intermediate results, if possible.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing "intermediate_unambig_results".

    Returns:
        higher_level_code (str): the higher level SIC code if final code cannot be assigned
            unambiguously.
    """
    if row["intermediate_unambig_results"]["higher_level_final_sic"] is not None:
        return row["intermediate_unambig_results"]["higher_level_final_sic"]
    return ""


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
print("Classification LLM loaded.")

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
        METADATA["stage5_classification_llm_model"] = MODEL_NAME
        METADATA["stage5_classification_llm_location"] = MODEL_LOCATION
        df = pd.read_pickle(args.input_pickle_file)  # noqa: S301
        print("Input loaded")

    print("running final SIC code assignment...")
    if (not args.restart) or (not RESTART_SUCCESS):
        df["intermediate_unambig_results"] = {
            "unambiguously_codable_final": False,
            "final_sic": "",
            "higher_level_final_sic": "",
        }
        df["unambiguously_codable_final"] = False
        df["final_sic"] = ""
        df["higher_level_final_sic"] = ""
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
                assign_final_sic_code, axis=1
            )
            df.loc[batch.index, "unambiguously_codable_final"] = batch.apply(
                get_unambiguous_status_final, axis=1
            )
            df.loc[batch.index, "final_sic"] = batch.apply(get_final_sic_code, axis=1)
            df.loc[batch.index, "higher_level_final_sic"] = batch.apply(
                get_higher_level_sic_code, axis=1
            )
            persist_results(
                df,
                METADATA,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1),
            )

    print("Final SIC code assignment is complete")
    print("deleting temporary DataFrame column...")
    df = df.drop("intermediate_unambig_results", axis=1)

    print("persisting results...")
    persist_results(
        df, METADATA, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
