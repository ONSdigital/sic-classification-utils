"""Provides data access for key files.

This module contains utility functions to load and process data from
SIC-related Excel files. The filepaths for these files are defined in
the configuration function in `embedding.py`.
"""

import pandas as pd


def load_sic_index(filepath: str) -> pd.DataFrame:
    """Loads the SIC index from an Excel file.

    The SIC index provides a list of around 15,000 activities and their
    associated 5-digit SIC codes.

    Args:
        filepath (str): The path to the Excel file containing the SIC index.

    Returns:
        pd.DataFrame: A DataFrame containing the SIC index with columns
        `uk_sic_2007` and `activity`.
    """
    sic_index_df = pd.read_excel(
        filepath,
        sheet_name="Alphabetical Index",
        skiprows=2,
        usecols=["UK SIC 2007", "Activity"],
        dtype=str,
    )

    sic_index_df.columns = [
        col.lower().replace(" ", "_") for col in sic_index_df.columns
    ]

    return sic_index_df


def load_sic_structure(filepath: str) -> pd.DataFrame:
    """Loads the SIC structure from an Excel file.

    This function loads a worksheet containing all the levels and names
    of the UK SIC 2007 hierarchy.

    Args:
        filepath (str): The path to the Excel file containing the SIC structure.

    Returns:
        pd.DataFrame: A DataFrame containing the SIC structure with columns
        `description`, `section`, `most_disaggregated_level`, and `level_headings`.
    """
    sic_df = pd.read_excel(
        filepath,
        sheet_name="reworked structure",
        usecols=[
            "Description",
            "SECTION",
            "Most disaggregated level",
            "Level headings",
        ],
        dtype=str,
    )

    sic_df.columns = [col.lower().replace(" ", "_") for col in sic_df.columns]

    for col in sic_df.columns:
        sic_df[col] = sic_df[col].str.strip()

    return sic_df
