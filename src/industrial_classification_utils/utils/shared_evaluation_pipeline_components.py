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
import shutil
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from typing import Optional

import gcsfs
import pandas as pd


def parse_args(default_output_shortname: str = "STGK") -> Namespace:
    """Parses command line arguments for the script.

    Args:
        default_output_shortname (str): The default prefix for output filenames.

    Returns:
        Namespace: The parsed command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        help="relative path to the persisted DataFrame from previous stage",
    )
    parser.add_argument(
        "--metadata_json",
        "-m",
        type=str,
        default=None,
        help="relative path to the persisted metadata from previous stage",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
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
        default=100,
        help="save the output every X rows, as a checkpoint that can be used to restart the "
        "processing job if needed (optional, default: 100)",
    )
    parser.add_argument(
        "--restart",
        "-r",
        action="store_true",
        default=False,
        help="try to restart a processing job (optional flag)",
    )
    parser.add_argument(
        "--second_run",
        "-s",
        action="store_true",
        default=False,
        help="""Select if running this stage for the second time.
            For STG1 adds second_semantic_search_results, for STG2 runs final classification.""",
    )
    return parser.parse_args()


def _try_to_restart(
    parsed_args: Namespace,
):
    """Attempts to restart a processing job by loading checkpoint data.

    This function tries to load a previously saved DataFrame, metadata, and
    checkpoint information from an intermediate output directory. If these files
    are found and loaded successfully, it marks it as a successful restart.

    If any file is missing, or another exception occurs during loading, it
    reverts to starting the process from scratch. In this case, it loads the
    initial input data and metadata, and prepares new checkpoint information.

    Args:
        parsed_args (Namespace): The namespace of parsed command-line arguments.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The loaded or newly created DataFrame.
            - dict: The loaded or newly created metadata dictionary.
            - dict: The loaded or newly created checkpoint information.

    Raises:
        FileNotFoundError: If starting from scratch and the initial data
            file (`input_file`) or metadata file (`input_metadata_json`)
            cannot be found.
    """
    output_folder = parsed_args.output_folder
    output_shortname = parsed_args.output_shortname

    df_persisted = pd.read_parquet(
        f"{output_folder}/intermediate_outputs/{output_shortname}.parquet"
    )
    checkpoint_info_persisted = _read_json(
        f"{output_folder}/intermediate_outputs/{output_shortname}_checkpoint_info.json"
    )
    metadata_persisted = _read_json(
        f"{output_folder}/intermediate_outputs/{output_shortname}_metadata.json"
    )

    print("Partially-processed data re-loaded succesfully")
    return (
        df_persisted,
        metadata_persisted,
        checkpoint_info_persisted,
    )


def persist_results(  # noqa:PLR0913, pylint: disable=too-many-arguments
    *,
    df: pd.DataFrame,
    metadata: dict,
    output_folder: str,
    output_shortname: str,
    is_final: Optional[bool] = False,
    completed_batches: Optional[int] = 0,
):
    """Persists the results DataFrame to CSV, parquet, and saves metadata to JSON.

    Args:
        df (pd.DataFrame): The DataFrame containing the results to be persisted.
        metadata (dict): The additional metadata surrounding this processing job.
        output_folder (str): The path to the output folder where the files will be saved.
        output_shortname (str): The prefix given to each file to be saved.
        is_final (bool): Mark the output as the final output and timestamp filenames.
                         Optional, default False.
        completed_batches (int): Specify the number of completed batches being saved.
                                 Optional, default 0.
    Returns: None
    """
    if is_final:
        print("Saving setup metadata to JSON...")
        _write_json(
            metadata,
            f"{output_folder}/{output_shortname}_metadata.json",
        )
        print("Saving results to parquet...")
        df.to_parquet(f"{output_folder}/{output_shortname}.parquet", index=False)
        print("Saving results to CSV...")
        df.to_csv(f"{output_folder}/{output_shortname}.csv", index=False)

        print("Removing intermediate outputs...")
        _delete_folder_contents(f"{output_folder}/intermediate_outputs")

    else:
        output_folder = f"{output_folder}/intermediate_outputs"
        _write_json(
            {
                "completed_batches": completed_batches,
                "batch_size": metadata["batch_size"],
                "batch_size_async": metadata.get("batch_size_async"),
            },
            f"{output_folder}/{output_shortname}_checkpoint_info.json",
        )
        _write_json(
            metadata,
            f"{output_folder}/{output_shortname}_metadata.json",
        )
        df.to_parquet(f"{output_folder}/{output_shortname}.parquet", index=False)


def _read_json(file_path: str) -> dict:
    """Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path: The path to the input JSON file.
            Can be a local path or a GCP bucket path (starting with "gs://").

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    if file_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(file_path, "r", encoding="utf8") as f:
            obj = json.load(f)
    else:
        with open(file_path, encoding="utf8") as f:
            obj = json.load(f)
    return obj


def _write_json(obj: dict, file_path: str) -> None:
    """Writes a dictionary to a JSON file - either local or in gcp bucket.

    Args:
        obj: The dictionary to be written to the JSON file.
        file_path: The path to the output JSON file.


    Returns:
        None
    """
    if file_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(file_path, "w", encoding="utf8") as f:
            f.write(json.dumps(obj))
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf8") as f:
            json.dump(obj, f)


def _delete_folder_contents(folder_path: str) -> None:
    """Deletes all files in the specified folder.

    Args:
        folder_path (str): The path to the folder whose contents are to be deleted.

    Returns:
        None
    """
    if folder_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        fs.rm(folder_path, recursive=True)
    else:
        shutil.rmtree(folder_path)


def set_up_initial_state(
    parsed_args: Namespace,
) -> tuple[pd.DataFrame, dict, int, bool]:
    """Sets up the initial state for a pipeline stage.

    This function handles the logic for starting a processing job, either by
    restarting from a previously saved checkpoint or by loading the initial
    data from scratch. It populates the initial DataFrame, metadata, and
    determines the starting point for batch processing.

    Args:
        parsed_args (Namespace): The namespace of parsed command-line arguments.
        stage_id (str): A prefix used for per-stage fields in the metadata.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The loaded or newly created DataFrame.
            - dict: The loaded or newly created metadata dictionary.
            - int: The starting batch ID for processing.
            - bool: True if a restart from a checkpoint was successful,
              False otherwise.
    """
    if parsed_args.restart:
        try:
            df, metadata, checkpoint_info = _try_to_restart(  # type: ignore
                parsed_args,
            )
            return (
                df,
                metadata,
                checkpoint_info["completed_batches"],
                parsed_args.second_run,
            )
        except FileNotFoundError:
            print("Could not load persisted output, starting from scratch")

    try:
        metadata = _read_json(parsed_args.metadata_json)
    except FileNotFoundError:
        print(
            f"Could not find metadata file {parsed_args.metadata_json}"
            ", will use default values for metadata fields"
        )
        metadata = {}

    metadata[f"{parsed_args.output_shortname}_input_file"] = parsed_args.input_file
    metadata[f"{parsed_args.output_shortname}_start_timestamp"] = datetime.now(
        UTC
    ).timestamp()
    metadata[f"{parsed_args.output_shortname}_start_time_readable"] = datetime.now(
        UTC
    ).strftime("%Y/%m/%d_%H:%M:%S")
    df = (
        pd.read_csv(parsed_args.input_file)
        if parsed_args.input_file.endswith(".csv")
        else pd.read_parquet(parsed_args.input_file)
    )

    print("Input loaded")

    return df, metadata, 0, parsed_args.second_run
