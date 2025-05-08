"""Unit tests for the SIC data access utility functions.

This module contains tests for the `load_sic_index` and `load_sic_structure`
functions from the `industrial_classification_utils.utils.sic_data_access` module.
"""

from unittest.mock import ANY, patch

import pandas as pd
import pytest

from industrial_classification_utils.utils.sic_data_access import (
    load_sic_index,
    load_sic_structure,
)

# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code


@pytest.fixture
def mock_sic_index_data():
    """Fixture for mock SIC index data.

    Returns:
        pd.DataFrame: A DataFrame containing mock SIC index data.
    """
    return pd.DataFrame(
        {"uk_sic_2007": ["12345", "67890"], "activity": ["Manufacturing", "Retail"]}
    )


@pytest.fixture
def mock_sic_structure_data():
    """Fixture for mock SIC structure data.

    Returns:
        pd.DataFrame: A DataFrame containing mock SIC structure data.
    """
    return pd.DataFrame(
        {
            "description": ["Section A", "Section B"],
            "section": ["A", "B"],
            "most_disaggregated_level": ["Level 1", "Level 2"],
            "level_headings": ["Heading 1", "Heading 2"],
        }
    )


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_sic_index(mock_read_excel, mock_sic_index_data):
    """Test the `load_sic_index` function.

    Args:
        mock_read_excel (MagicMock): Mocked `pandas.read_excel` function.
        mock_sic_index_data (pd.DataFrame): Mock SIC index data.

    Asserts:
        - The `pandas.read_excel` function is called with the correct arguments.
        - The returned DataFrame matches the mock SIC index data.
    """
    mock_read_excel.return_value = mock_sic_index_data
    result = load_sic_index(("industrial_classification_utils.data.sic_index", "uksic2007indexeswithaddendumdecember2022.xlsx"))

    mock_read_excel.assert_called_once_with(
        ANY,
        sheet_name="Alphabetical Index",
        skiprows=2,
        usecols=["UK SIC 2007", "Activity"],
        dtype=str,
    )

    # Verify the path used in the call
    called_args, _ = mock_read_excel.call_args
    assert str(called_args[0]).endswith("uksic2007indexeswithaddendumdecember2022.xlsx")
    assert result.equals(mock_sic_index_data)


@pytest.mark.utils
@patch("pandas.read_excel")
def test_load_sic_structure(mock_read_excel, mock_sic_structure_data):
    """Test the `load_sic_structure` function.

    Args:
        mock_read_excel (MagicMock): Mocked `pandas.read_excel` function.
        mock_sic_structure_data (pd.DataFrame): Mock SIC structure data.

    Asserts:
        - The `pandas.read_excel` function is called with the correct arguments.
        - The returned DataFrame matches the mock SIC structure data.
    """
    mock_read_excel.return_value = mock_sic_structure_data
    result = load_sic_structure(("industrial_classification_utils.data.sic_index", "publisheduksicsummaryofstructureworksheet.xlsx"))
    mock_read_excel.assert_called_once_with(
        ANY,
        sheet_name="reworked structure",
        usecols=[
            "Description",
            "SECTION",
            "Most disaggregated level",
            "Level headings",
        ],
        dtype=str,
    )

    # Verify the path used in the call
    called_args, _ = mock_read_excel.call_args
    assert str(called_args[0]).endswith("publisheduksicsummaryofstructureworksheet.xlsx")

    assert result.equals(mock_sic_structure_data)
