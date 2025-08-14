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
        "input_parquet_file",
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
    output_folder, output_shortname, input_parquet_file, input_metadata_json, batch_size
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
        input_parquet_file (str): The path to the (previous stage's) persisted dataframe file,
            used only if starting from scratch after failure to restart.
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
            file (`input_parquet_file`) or metadata file (`input_metadata_json`)
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
    except (FileNotFoundError, Exception):  # pylint: disable=W0718
        print("Could not re-load checkpointed results, restarting from scratch...")
        restart_successful = False
        try:
            with open(input_metadata_json, encoding="utf-8") as input_meta:
                metadata_persisted = json.load(input_meta)
        except FileNotFoundError:
            print(f"Could not find metadata file {input_metadata_json}")
            raise
        metadata_persisted["stage_3_start_timestamp"] = datetime.now(UTC).timestamp()
        metadata_persisted["stage_3_start_time_readable"] = datetime.now(UTC).strftime(
            "%Y/%m/%d_%H:%M:%S"
        )
        metadata_persisted["batch_size"] = batch_size
        df_persisted = pd.read_parquet(input_parquet_file)
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


def get_open_question(row: pd.Series) -> str:  # pylint: disable=C0103, W0613
    """Using the provided row data, call an LLM to generate an open follow-up question.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing the columns corresponding
                         to the survey responses, and the semantic search results.
    Returns: question (str).
    """
    sa_sic_rag = c_llm.sa_rag_sic_code(
        industry_descr=row[INDUSTRY_DESCR_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        candidates_limit=10,
        short_list=row["semantic_search_results"],
    )

    sic_followup_object, _ = c_llm.formulate_open_question(
        industry_descr=row[INDUSTRY_DESCR_COL],
        job_title=row[JOB_TITLE_COL],
        job_description=row[JOB_DESCRIPTION_COL],
        llm_output=sa_sic_rag[0].sic_candidates,  # type: ignore
    )
    if sic_followup_object.followup is None:
        return ""
    return sic_followup_object.followup


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
        is_final (bool): Mark the output as the final output. Optional, default False.
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
                args.input_parquet_file,
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
        METADATA["stage_3_start_timestamp"] = datetime.now(UTC).timestamp()
        METADATA["stage_3_start_time_readable"] = datetime.now(UTC).strftime(
            "%Y/%m/%d_%H:%M:%S"
        )
        METADATA["batch_size"] = args.batch_size
        METADATA["stage3_classification_llm_model"] = MODEL_NAME
        METADATA["stage3_classification_llm_location"] = MODEL_LOCATION
        df = pd.read_parquet(args.input_parquet_file)
        print("Input loaded")

    print("getting followup questions ...")
    if (not args.restart) or (not RESTART_SUCCESS):
        df["followup_question"] = ""
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
            df.loc[batch.index, "followup_question"] = batch.apply(
                get_open_question, axis=1
            )
            persist_results(
                df,
                METADATA,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + START_BATCH_ID),
            )

    print("Followup question retrieval is complete")

    print("persisting results...")
    persist_results(
        df, METADATA, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
