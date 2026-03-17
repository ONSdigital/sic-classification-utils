#!/usr/bin/env python3
"""This script performs semantic search on a dataset using a local vector store
and persists the results. It reads in a CSV file as a DataFrame object,
uses :class:`industrial_classification_utils.embed.embedding.EmbeddingHandler`
to obtain semantic search results for each row, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

Clarification On Script Arguments:

```bash
python stage_1_add_semantic_search.py --help
```

Example Usage:

1. Run the script:
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
from tqdm import tqdm

from industrial_classification_utils.embed.embedding import EmbeddingHandler
from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_DIR = "src/industrial_classification_utils/data/vector_store"
K_MATCHES = 20
EMBEDDING_SIC_INDEX_FILE = "extended_SIC_index.xlsx"
EMBEDDING_SIC_STRUCTURE_FILE = "publisheduksicsummaryofstructureworksheet.xlsx"


INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
SELF_EMPLOYED_DESC_COL = "sic2007_self_employed"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"  # created in this script
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
    updated_metadata = in_metadata.copy() if in_metadata else {}

    if updated_metadata.get("original_dataset_name") != parsed_args.input_file:
        print(
            f"""Warning: The original dataset name in the input metadata ({
                in_metadata.get('original_dataset_name')}) """
            f"does not match the input file specified in the arguments ({parsed_args.input_file}). "
            "The metadata will be updated with the input file name."
        )
        updated_metadata["original_dataset_name"] = parsed_args.input_file

    if updated_metadata.get("batch_size") != parsed_args.batch_size:
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


def _make_semantic_search_fn(one_embedding_handler: EmbeddingHandler):
    def _get_semantic_search_results(row: pd.Series) -> list[dict]:
        """Performs a semantic search using the provided row data.

        Intended for use as a `.apply()` operation to create a new column in a
        pd.DataFrame.

        Returns:
            list[dict]: List of dictionaries with `title`, `code`, `distance`.
        """
        industry_descr = (
            row[MERGED_INDUSTRY_DESC_COL]
            if isinstance(row.get(MERGED_INDUSTRY_DESC_COL), str)
            else ""
        )
        job_title = (
            row[JOB_TITLE_COL] if isinstance(row.get(JOB_TITLE_COL), str) else ""
        )
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

    return _get_semantic_search_results


if __name__ == "__main__":
    args = parse_args("STG1")

    df, metadata, start_batch_id, second_run_variables = set_up_initial_state(args)

    metadata = _update_metadata_with_args_and_defaults(args, metadata)

    embedding_handler = _make_embedding_handler(metadata)
    print(
        f"Vector store ready (in-process): {embedding_handler._index_size} entries"  # pylint: disable=protected-access
    )
    get_semantic_search_results = _make_semantic_search_fn(embedding_handler)

    semantic_search_output_col = (  # pylint: disable=C0103
        "second_semantic_search_results"
        if second_run_variables
        else "semantic_search_results"
    )

    # Clean the Survey Response columns:
    df[JOB_DESCRIPTION_COL] = df[JOB_DESCRIPTION_COL].apply(clean_text)
    df[JOB_TITLE_COL] = df[JOB_TITLE_COL].apply(clean_text)
    # Make a merged industry description column:
    if MERGED_INDUSTRY_DESC_COL not in df:
        df[INDUSTRY_DESCR_COL] = df[INDUSTRY_DESCR_COL].apply(clean_text)
        df[SELF_EMPLOYED_DESC_COL] = df[SELF_EMPLOYED_DESC_COL].apply(clean_text)
        df[MERGED_INDUSTRY_DESC_COL] = df.apply(make_merged_industry_desc, axis=1)
    df[MERGED_INDUSTRY_DESC_COL] = df[MERGED_INDUSTRY_DESC_COL].apply(
        clean_text_industry
    )
    print("Input loaded")

    print("running semantic search...")
    if semantic_search_output_col not in df:
        df[semantic_search_output_col] = np.empty((len(df), 0)).tolist()

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
            df.loc[batch.index, semantic_search_output_col] = batch.apply(
                get_semantic_search_results, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("semantic search complete")
    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
