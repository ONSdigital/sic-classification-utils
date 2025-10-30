"""Provides data access for key files.

This module contains utility functions to load and process data from
SIC-related Excel files. The filepaths for these files are defined in
the configuration function in `embedding.py`.
"""

import logging
from importlib.resources import files

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sic_index(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SIC index from an Excel file.

    The SIC index provides a list of around 15,000 activities and their
    associated 5-digit SIC codes.

    Args:
        resource_ref (tuple): The path to the Excel file containing the SIC index.

    Returns:
        pd.DataFrame: A DataFrame containing the SIC index with columns
        `uk_sic_2007` and `activity`.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)

    logger.debug("Loading SIC index from %s", file_path)

    sic_index_df = pd.read_excel(
        file_path,
        sheet_name="Alphabetical Index",
        skiprows=2,
        usecols=["UK SIC 2007", "Activity"],
        dtype=str,
    )

    sic_index_df.columns = [
        col.lower().replace(" ", "_") for col in sic_index_df.columns
    ]

    return sic_index_df


def load_sic_structure(resource_ref: tuple[str, str]) -> pd.DataFrame:
    """Loads the SIC structure from an Excel file.

    This function loads a worksheet containing all the levels and names
    of the UK SIC 2007 hierarchy.

    Args:
        resource_ref (tuple): The path to the Excel file containing the SIC structure.

    Returns:
        pd.DataFrame: A DataFrame containing the SIC structure with columns
        `description`, `section`, `most_disaggregated_level`, and `level_headings`.
    """
    pkg, filename = resource_ref
    file_path = files(pkg).joinpath(filename)

    logger.debug("Loading SIC structure from %s", file_path)

    sic_df = pd.read_excel(
        file_path,
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


def load_text_from_config(config_section: tuple[str, str]) -> str:
    """Loads text content from a configuration file.

    This function reads the content of a text file specified by the given
    configuration section and returns it as a string.

    Args:
        config_section (tuple[str, str]): A tuple containing the package name
            and the filename of the configuration file.

    Returns:
        str: The content of the configuration file as a string.

    """
    pkg, filename = config_section
    file_path = files(pkg).joinpath(filename)

    logger.debug("Loading text from %s", file_path)

    with file_path.open(encoding="utf-8") as f:
        return f.read()
