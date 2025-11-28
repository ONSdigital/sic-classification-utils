#!/usr/bin/env python3
# pylint: disable=duplicate-code
# pylint: disable=invalid-name
"""This script analyzes a dataset to determine if each record is
"unambiguously codable" for a Standard Industrial Classification (SIC) code.

It reloads the output from the previous stage as a DataFrame object, uses a
Large Language Model (LLM) to assess codability for each row, and adds new
columns for the codability status, an initial SIC code (if one can be
assigned), and a list of alternative SIC candidates. The results are then saved
to CSV, parquet, and JSON metadata files in a user-specified output folder.

The script requires a configured connection to a compatible LLM.

Clarification On Script Arguments:

```bash
python stage_2_add_unambiguously_codable_status.py --help
```

Example Usage:

1. Ensure you have run `gcloud` (re-)authentication for the current project.

2. Run the script:
   ```bash
   python stage_2_add_unambiguously_codable_status.py \
        -n my_output \
        -b 10 \
        -s \
        persisted_dataframe.parquet \
        persisted_metadata.json \
        output_folder
   ```
   where:
     - `-n my_output` sets the output filename prefix to "my_output".
     - `-b 10` specifies to process in batches of 10 rows, checkpointing between batches.
        Note: the max batch size is limited to 10 here, due to LLM concurrency constraints.
     - `-s` get final_sic (if `-s` absent, run initial stage).
     - `persisted_dataframe.parquet` is the saved dataframe output at the previous stage.
     - `persisted_metadata.json` is persisted JSON metadata from the previous stage.
     - `output_folder` is the directory where results will be saved.

3. Verify outputs exist as expected:
    ```bash
   ls output_folder
   ```
   (expect to see my_output.csv, my_output.parquet, and my_output_metadata.json)

"""
import asyncio
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.llm.llm import ClassificationLLM
from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
MODEL_NAME = "gemini-2.5-flash"
MODEL_LOCATION = "europe-west9"

CODE_DIGITS = 5
CANDIDATES_LIMIT = 10

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


# This new async function processes a whole batch of rows concurrently.
async def get_unambiguous_sic_batch_async(
    batch: pd.DataFrame, semantic_search
) -> list[dict[str, Any]]:
    """Processes a batch of rows asynchronously to get unambiguous SIC codability."""
    # 1. Create a task for each row in the batch
    tasks = []
    for _, row in batch.iterrows():
        task = asyncio.create_task(
            c_llm.unambiguous_sic_code(
                industry_descr=row[MERGED_INDUSTRY_DESC_COL],
                semantic_search_results=row[semantic_search],
                job_title=row[JOB_TITLE_COL],
                job_description=row[JOB_DESCRIPTION_COL],
                candidates_limit=CANDIDATES_LIMIT,
                code_digits=CODE_DIGITS,
            )
        )
        tasks.append(task)

    # 2. Run all tasks concurrently
    responses = await asyncio.gather(*tasks)

    # 3. Process the results from asyncio.gather()
    results = []
    for sa_response, _ in responses:
        results.append(
            {
                "unambiguously_codable": sa_response.codable,
                "code": sa_response.class_code,
                "alt_candidates": [
                    {
                        "code": i.class_code,
                        "title": i.class_descriptive,
                        "likelihood": i.likelihood,
                    }
                    for i in sa_response.alt_candidates
                ],
            }
        )
    return results


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


def get_sic_code(row: pd.Series) -> str:
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


def get_alt_sic_candidates(row: pd.Series) -> list[dict]:
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


async def main():
    """Main function to run the unambiguous codability analysis.
    Deviates from the stage_k template to enable async processing.
    """
    global c_llm, args  # noqa: PLW0603 # pylint: disable=global-statement, global-variable-undefined
    c_llm = ClassificationLLM(MODEL_NAME, verbose=False)
    print("Classification LLM loaded.")
    args = parse_args("STG2")

    df, metadata, start_batch_id, restart_successful, second_run_variables = (
        set_up_initial_state(
            args.restart,
            args.second_run,
            args.output_folder,
            args.output_shortname,
            args.input_parquet_file,
            args.input_metadata_json,
            args.batch_size,
            stage_id="stage_2",
        )
    )

    print("running unamibuous codability analysis...")

    if second_run_variables:
        SIC_CODE = "final_code"
        CODABLE = "unambiguously_codable_final"
        ALT_CANDIDATES = "alt_sic_candidates_final"
        semantic_search = "second_semantic_search_results"
    else:
        SIC_CODE = "initial_code"
        CODABLE = "unambiguously_codable"
        ALT_CANDIDATES = "alt_sic_candidates"
        semantic_search = "semantic_search_results"

    if (not args.restart) or (not restart_successful):
        df["intermediate_unambig_results"] = {
            CODABLE: False,
            SIC_CODE: "",
            ALT_CANDIDATES: [],
        }
        df[CODABLE] = False
        df[SIC_CODE] = ""
        df[ALT_CANDIDATES] = np.empty((len(df), 0)).tolist()

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
            results = await get_unambiguous_sic_batch_async(batch, semantic_search)
            batch.loc[batch.index, "intermediate_unambig_results"] = results
            df.loc[batch.index, CODABLE] = batch.apply(get_unambiguous_status, axis=1)
            df.loc[batch.index, SIC_CODE] = batch.apply(get_sic_code, axis=1)
            df.loc[batch.index, ALT_CANDIDATES] = batch.apply(
                get_alt_sic_candidates, axis=1
            )
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    print("unambiguous coding analysis is complete")
    print("deleting temporary DataFrame column...")
    df = df.drop("intermediate_unambig_results", axis=1)
    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
