#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This is a template script for a stage in the evaluation pipeline.

-----------------------------------------------------------------------------------------
What to change, to adapt it to a given stage's requirements:

* update `update_metadata_with_args_and_defaults()` function to add any relevant values from
  the command-line arguments and to set any defaults for missing metadata values.
* update `get_x()` function to achieve whatever is required for your new column.
* update the `if __name__=="__main__"` block to use the new function names, and
  repeat the creation of the empty new column and batch.apply() if you are adding
  more than one new column.
* if asynchronous calls are needed, move the batch processing loop into an async function
  and use `asyncio.run()` for batches.  Respect the MAX_ASYNC_BATCH_SIZE constant.
"""

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from industrial_classification_utils.utils.shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Default values and constants:
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
INDUSTRY_DESCR_COL = "sic2007_employee"
SELF_EMPLOYED_DESC_COL = "sic2007_self_employed"
MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"

OUTPUT_COL = "stage_k_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


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


def get_x(row: pd.Series) -> Any:  # pylint: disable=C0103, W0613
    """Performs X using the provided row data.
    Intended for use as a `.apply()` operation to create a new colum in a pd.DataFrame object.

    Args:
        row (pd.Series): A row from the input DataFrame containing <required columns>.
    Returns: X.
    """
    return 1


if __name__ == "__main__":
    args = parse_args("STGK")

    check_y()
    print("Requirement Y is met")

    df, metadata, start_batch_id = set_up_initial_state(parsed_args=args)

    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = 0

    print("running X...")
    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(
                    start_batch_id * metadata["batch_size"],
                    len(df),
                    metadata["batch_size"],
                ),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            df.loc[batch.index, OUTPUT_COL] = batch.apply(get_x, axis=1)
            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("X is complete")

    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")
