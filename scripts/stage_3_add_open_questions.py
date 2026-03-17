#!/usr/bin/env python3
# pylint: disable=duplicate-code, redefined-outer-name
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
        -b 10 \
        persisted_dataframe.parquet \
        persisted_metadata.json \
        output_folder
   ```
   where:
     - `-n my_output` sets the output filename prefix to "my_output".
     - `-b 10` specifies to process in batches of 10 rows, checkpointing between batches.
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
# Default values and constants:
MODEL_NAME = "gemini-2.5-flash"
MODEL_LOCATION = "europe-west1"

CODE_DIGITS = 5
CANDIDATES_LIMIT = 10
MAX_BATCH_SIZE = 10

JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
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

    # Update metadata with values from parsed_args, using defaults where necessary
    updated_metadata["model_name"] = updated_metadata.get("model_name", MODEL_NAME)
    updated_metadata["model_location"] = updated_metadata.get(
        "model_location", MODEL_LOCATION
    )
    updated_metadata["code_digits"] = updated_metadata.get("code_digits", CODE_DIGITS)
    updated_metadata["candidates_limit"] = updated_metadata.get(
        "candidates_limit", CANDIDATES_LIMIT
    )

    if parsed_args.batch_size > MAX_BATCH_SIZE:
        print(f"batch size too large. lower batch size to {MAX_BATCH_SIZE}")

    updated_metadata["batch_size_async"] = min(parsed_args.batch_size, MAX_BATCH_SIZE)

    return updated_metadata


async def get_open_question_batch_async(
    batch: pd.DataFrame, c_llm: ClassificationLLM
) -> list[str]:
    """Process a batch of rows asynchronously to generate an open follow-up question for each row.

    Args:
        batch (pd.DataFrame): A batch of DataFrame containing rows with columns corresponding
                         to the survey responses, and the semantic search results.
        c_llm (ClassificationLLM): An initialised instance of the ClassificationLLM class.

    Returns: question (str).
    """
    tasks = []
    for _, row in batch.iterrows():
        task = asyncio.create_task(
            c_llm.formulate_open_question(
                industry_descr=row[MERGED_INDUSTRY_DESC_COL],
                job_title=row[JOB_TITLE_COL],
                job_description=row[JOB_DESCRIPTION_COL],
                llm_output=row["alt_sic_candidates"],  # type: ignore
            )
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    results = []
    for sic_followup_object, _ in responses:
        if sic_followup_object.followup is None:
            results.append("")
        else:
            results.append(sic_followup_object.followup)
    return results


async def main_async(df, metadata, start_batch_id, args, c_llm):
    """Main function to generate follow up questions.
    Deviates from the stage_k template to enable async processing.
    """
    print("getting followup questions ...")

    df_uncodable = df[~df["unambiguously_codable"]]

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df_uncodable,
                np.arange(
                    start_batch_id * metadata["batch_size_async"],
                    len(df_uncodable),
                    metadata["batch_size_async"],
                ),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            results = await get_open_question_batch_async(batch, c_llm)
            df.loc[batch.index, "followup_question"] = results

            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("Followup question retrieval is complete")

    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_args("STG3")

    df, metadata, start_batch_id, _ = set_up_initial_state(args)

    metadata = _update_metadata_with_args_and_defaults(args, metadata)

    c_llm = ClassificationLLM(
        model_name=metadata["model_name"],
        model_location=metadata["model_location"],
        verbose=False,
    )
    print("Classification LLM loaded.")

    if "followup_question" not in df.columns:
        df["followup_question"] = ""

    asyncio.run(
        main_async(
            df=df,
            metadata=metadata,
            start_batch_id=start_batch_id,
            args=args,
            c_llm=c_llm,
        )
    )
