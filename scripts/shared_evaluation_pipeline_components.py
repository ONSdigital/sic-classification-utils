#!/usr/bin/env python
"""Shared components for the SIC classification pipeline scripts.

This module provides a set of common, reusable functions for the various
stages of the data processing pipeline. These functions are designed to
handle common tasks such as parsing command-line arguments, managing
checkpointing and restarting of processing jobs, and persisting results
to various file formats.

The main components provided are:
- `parse_args`: A function to parse common command-line arguments for
  pipeline stage scripts.
- `_try_to_restart`: A function to handle loading data from a checkpoint
  or starting a processing stage from scratch.
- `persist_results`: A function to save DataFrames and metadata to
  intermediate or final output files.
- `set_up_initial_state`: A high-level function to orchestrate the
  initial setup of a pipeline stage, determining whether to restart
  from a checkpoint and loading the necessary data.
"""
import json
import os
from argparse import ArgumentParser as AP
from argparse import Namespace
from datetime import UTC, datetime
from typing import Optional

import pandas as pd


def parse_args(default_output_shortname: str = "STGK") -> Namespace:
    """Parses command line arguments for the script.

    Args:
        default_output_shortname (str): The default prefix for output filenames.

    Returns:
        Namespace: The parsed command-line arguments.
    """
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
        default=default_output_shortname,
        help="output filename prefix for easy identification (optional, default: "
        f"{default_output_shortname})",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=50,
        help="save the output every X rows, as a checkpoint that can be used to restart the "
        "processing job if needed (optional, default: 50)",
    )
    parser.add_argument(
        "--restart",
        "-r",
        action="store_true",
        default=False,
        help="try to restart a processing job (optional flag)",
    )
    return parser.parse_args()


def _try_to_restart(  # noqa: PLR0913 # pylint: disable=R0913, R0917
    output_folder: str,
    output_shortname: str,
    input_parquet_file: str,
    input_metadata_json: str,
    batch_size: int,
    stage_id: Optional[str] = "stage_k",
):
    """Attempts to restart a processing job by loading checkpoint data.

    This function tries to load a previously saved DataFrame, metadata, and
    checkpoint information from an intermediate output directory. If these files
    are found and loaded successfully, it marks it as a successful restart.

    If any file is missing, or another exception occurs during loading, it
    reverts to starting the process from scratch. In this case, it loads the
    initial input data and metadata, and prepares new checkpoint information.

    Args:
        output_folder (str): The path to the specified output folder.
        output_shortname (str): The prefix for the output filenames.
        input_parquet_file (str): The path to the (previous stage's) persisted dataframe file, used
            only if starting from scratch after failure to restart.
        input_metadata_json (str): The path to the (previous stage's) metadata JSON file,
            used only if starting from scratch after failure to restart.
        batch_size (int): The size of processing batches, used only if starting
            from scratch after failure to restart.
        stage_id (str): A prefix used for per-stage fields in the metadata.

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
        metadata_persisted[f"{stage_id}_start_timestamp"] = datetime.now(
            UTC
        ).timestamp()
        metadata_persisted[f"{stage_id}_start_time_readable"] = datetime.now(
            UTC
        ).strftime("%Y/%m/%d_%H:%M:%S")
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
        is_final (bool): Mark the output as the final output and timestamp filenames.
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


def set_up_initial_state(  # noqa: PLR0913 # pylint: disable=R0913, R0917
    restart: bool,
    output_folder: str,
    output_shortname: str,
    input_parquet_file: str,
    input_metadata_json: str,
    batch_size: int,
    stage_id: Optional[str] = "stage_k",
) -> tuple[pd.DataFrame, dict, int, bool]:
    """Sets up the initial state for a pipeline stage.

    This function handles the logic for starting a processing job, either by
    restarting from a previously saved checkpoint or by loading the initial
    data from scratch. It populates the initial DataFrame, metadata, and
    determines the starting point for batch processing.

    Args:
        restart (bool): Flag to indicate whether to attempt a restart from a
            checkpoint.
        output_folder (str): The path to the specified output folder.
        output_shortname (str): The prefix for the output filenames.
        input_parquet_file (str): The path to the persisted dataframe file from
            the previous stage.
        input_metadata_json (str): The path to the persisted metadata JSON file
            from the previous stage.
        batch_size (int): The size of processing batches.
        stage_id (str): A prefix used for per-stage fields in the metadata.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The loaded or newly created DataFrame.
            - dict: The loaded or newly created metadata dictionary.
            - int: The starting batch ID for processing.
            - bool: True if a restart from a checkpoint was successful,
              False otherwise.
    """
    restart_successful = True
    if restart:
        try:
            df, metadata, checkpoint_info, restart_successful = _try_to_restart(  # type: ignore
                output_folder,
                output_shortname,
                input_parquet_file,
                input_metadata_json,
                batch_size,
                stage_id=stage_id,
            )
        except Exception:
            print(
                "Could not load persisted output, and ran into an issue starting from scratch"
            )
            raise
    else:
        try:
            with open(input_metadata_json, encoding="utf-8") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Could not find metadata file {input_metadata_json}")
            raise
        metadata[f"{stage_id}_start_timestamp"] = datetime.now(UTC).timestamp()
        metadata[f"{stage_id}_start_time_readable"] = datetime.now(UTC).strftime(
            "%Y/%m/%d_%H:%M:%S"
        )
        metadata["batch_size"] = batch_size
        df = pd.read_parquet(input_parquet_file)
        print("Input loaded")

    start_batch_id = (
        0
        if (not restart) or (not restart_successful)
        else checkpoint_info["completed_batches"]  # type: ignore
    )

    return df, metadata, start_batch_id, restart_successful
