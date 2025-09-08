#!/usr/bin/env python3
# pylint: disable=duplicate-code
"""This script performs X on a dataset and persists the results.
It reads reloads the output from the previous stage as a DataFrame object,
performs X for each row, creates a new column in the DataFrame with this
information, and then saves the results to CSV, parquet, and JSON metadata
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
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .shared_evaluation_pipeline_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
VECTOR_STORE_URL_BASE = "http://0.0.0.0:8088"
STATUS_ENDPOINT = "/v1/sic-vector-store/status"
SEARCH_ENDPOINT = "/v1/sic-vector-store/search-index"

INDUSTRY_DESCR_COL = "sic2007_employee"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
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

    df, metadata, start_batch_id, restart_successful = set_up_initial_state(
        args.restart,
        args.output_folder,
        args.output_shortname,
        args.input_parquet_file,
        args.input_metadata_json,
        args.batch_size,
        stage_id="stage_k"
    )

    print("running X...")
    if (not args.restart) or (not restart_successful):
        df["new_column"] = 0

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
            df.loc[batch.index, "new_column"] = batch.apply(get_x, axis=1)
            persist_results(
                df,
                metadata,
                args.output_folder,
                args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + 1 + start_batch_id),
            )

    print("X is complete")

    print("persisting results...")
    persist_results(
        df, metadata, args.output_folder, args.output_shortname, is_final=True
    )
    print("Done!")
