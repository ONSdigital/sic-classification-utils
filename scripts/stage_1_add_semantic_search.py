#!/usr/bin/env python3
"""This script performs semantic search on a dataset using a local vector store
and persists the results. It reads in a CSV/PARQUET file as a DataFrame object,
uses :class:`industrial_classification_utils.embed.embedding.EmbeddingHandler`
to obtain semantic search results for each row, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""
from re import sub as regex_sub

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Default values and constants:
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_DIR = "src/industrial_classification_utils/data/vector_store"
K_MATCHES = 20
EMBEDDING_SIC_INDEX_FILE = "extended_SIC_index.xlsx"
EMBEDDING_SIC_STRUCTURE_FILE = "publisheduksicsummaryofstructureworksheet.xlsx"

JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
INDUSTRY_DESCR_COL = "sic2007_employee"
SELF_EMPLOYED_DESC_COL = "sic2007_self_employed"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"

OUTPUT_COL_INITIAL = "semantic_search_results"

# Constants for second run (final codes):
MERGED_INDUSTRY_DESC_COL_FINAL = "extended_industry_desc"
OUTPUT_COL_FINAL = "second_semantic_search_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def _update_metadata_with_args_and_defaults(parsed_args, in_metadata):
    """Updates the metadata dictionary with values from the command-line arguments,
    using defaults where necessary.

    Args:
        parsed_args: The command-line arguments parsed by `parse_args()`.
        in_metadata: The initial metadata dictionary loaded from the input JSON file.

    Returns:
        dict: The updated metadata dictionary with values from args and defaults.
    """
    updated_metadata = in_metadata.copy() if in_metadata else {}

    if updated_metadata.get("original_dataset_name") != parsed_args.input_file:
        print(
            f"""Warning: The original dataset name in the input metadata ({
                updated_metadata.get('original_dataset_name')}) """
            f"does not match the input file specified in the arguments ({parsed_args.input_file}). "
            "The metadata will be updated with the input file name."
        )
        updated_metadata["original_dataset_name"] = parsed_args.input_file

    if "original_dataset_name" not in updated_metadata:
        updated_metadata["original_dataset_name"] = parsed_args.input_file
    elif updated_metadata.get("original_dataset_name") != parsed_args.input_file:
        print(
            f"""Warning: The original dataset name in the input metadata ({
                in_metadata.get('original_dataset_name')}) """
            f"does not match the input file specified in the arguments ({parsed_args.input_file}). "
            "The metadata will be updated with the input file name."
        )
        updated_metadata["original_dataset_name"] = parsed_args.input_file

    if "batch_size" not in updated_metadata:
        updated_metadata["batch_size"] = parsed_args.batch_size
    elif updated_metadata.get("batch_size") != parsed_args.batch_size:
        print(
            f"Warning: The batch size in the input metadata ({in_metadata.get('batch_size')}) "
            f"does not match the batch size specified in the arguments ({parsed_args.batch_size}). "
            "The metadata will be updated with the batch size."
        )
        updated_metadata["batch_size"] = parsed_args.batch_size

    updated_metadata["embedding_model_name"] = updated_metadata.get(
        "embedding_model_name", EMBEDDING_MODEL_NAME
    )
    updated_metadata["db_dir"] = updated_metadata.get("db_dir", DB_DIR)
    updated_metadata["k_matches"] = updated_metadata.get("k_matches", K_MATCHES)
    updated_metadata["sic_index_file"] = updated_metadata.get(
        "sic_index_file", EMBEDDING_SIC_INDEX_FILE
    )
    updated_metadata["sic_structure_file"] = updated_metadata.get(
        "sic_structure_file", EMBEDDING_SIC_STRUCTURE_FILE
    )

    return updated_metadata


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
    text = text.replace(
        "- followup",
        """
- Followup""",
    )
    return text


def _make_embedding_handler(in_metadata: dict) -> EmbeddingHandler:
    """Create an :class:`EmbeddingHandler` using settings from metadata where possible."""
    new_embedding_handler = EmbeddingHandler(
        embedding_model_name=in_metadata.get(
            "embedding_model_name", "all-MiniLM-L6-v2"
        ),
        db_dir=in_metadata.get(
            "db_dir", "src/industrial_classification_utils/data/vector_store"
        ),
        k_matches=in_metadata.get("k_matches", 20),
    )

    new_embedding_handler.embed_index(
        from_empty=True,
        sic_index_file=(
            "industrial_classification_utils.data.sic_index",
            in_metadata.get("sic_index_file", "extended_SIC_index.xlsx"),
        ),
        sic_structure_file=(
            "industrial_classification_utils.data.sic_index",
            in_metadata.get(
                "sic_structure_file", "publisheduksicsummaryofstructureworksheet.xlsx"
            ),
        ),
    )

    return new_embedding_handler


def _get_semantic_search_results(
    row: pd.Series, one_embedding_handler: EmbeddingHandler, second_run_flag: bool
) -> list[dict]:
    """Performs a semantic search using the provided row data.

    Intended for use as a `.apply()` operation to create a new column in a
    pd.DataFrame.

    Returns:
        list[dict]: List of dictionaries with `title`, `code`, `distance`.
        one_embedding_handler (EmbeddingHandler): An instance of the EmbeddingHandler class
            to perform the search.
        second_run_flag (bool): Flag indicating whether this is the second run (final codes)
            or not, which determines which column to use for the search query.

    """
    industry_descr = (
        row[MERGED_INDUSTRY_DESC_COL_FINAL]
        if second_run_flag
        else row[MERGED_INDUSTRY_DESC_COL]
    )
    industry_descr = industry_descr if isinstance(industry_descr, str) else ""

    job_title = row[JOB_TITLE_COL] if isinstance(row.get(JOB_TITLE_COL), str) else ""
    job_description = (
        row[JOB_DESCRIPTION_COL]
        if isinstance(row.get(JOB_DESCRIPTION_COL), str)
        else ""
    )

    results = one_embedding_handler.search_index_multi(
        [industry_descr, job_title, job_description]
    )

    reduced_results = [
        {
            "title": r.get("title", ""),
            "code": r.get("code", ""),
            "distance": float(r.get("distance", 0.0)),
        }
        for r in results
    ]
    return reduced_results


if __name__ == "__main__":
    args = parse_args("STG1")

    df, metadata, start_batch_id = set_up_initial_state(args)

    metadata = _update_metadata_with_args_and_defaults(args, metadata)

    embedding_handler = _make_embedding_handler(metadata)
    print(
        f"Vector store ready (in-process): {embedding_handler._index_size} entries"  # pylint: disable=protected-access
    )

    # Clean the Survey Response columns:
    df[JOB_DESCRIPTION_COL] = df[JOB_DESCRIPTION_COL].apply(clean_text)
    df[JOB_TITLE_COL] = df[JOB_TITLE_COL].apply(clean_text)
    # Make a merged industry description column:
    if args.second_run:
        df[MERGED_INDUSTRY_DESC_COL_FINAL] = df[MERGED_INDUSTRY_DESC_COL_FINAL].apply(
            clean_text_industry
        )
    else:
        df[INDUSTRY_DESCR_COL] = df[INDUSTRY_DESCR_COL].apply(clean_text)
        df[SELF_EMPLOYED_DESC_COL] = df[SELF_EMPLOYED_DESC_COL].apply(clean_text)
        df[MERGED_INDUSTRY_DESC_COL] = df.apply(make_merged_industry_desc, axis=1)
        df[MERGED_INDUSTRY_DESC_COL] = df[MERGED_INDUSTRY_DESC_COL].apply(
            clean_text_industry
        )
    print("Input loaded")

    OUTPUT_COL = OUTPUT_COL_FINAL if args.second_run else OUTPUT_COL_INITIAL
    if OUTPUT_COL not in df:
        df[OUTPUT_COL] = np.empty((len(df), 0)).tolist()

    print("running semantic search...")
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
            df.loc[batch.index, OUTPUT_COL] = batch.apply(
                lambda row: _get_semantic_search_results(
                    row, embedding_handler, args.second_run
                ),
                axis=1,
            )
            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("semantic search complete")
    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")
