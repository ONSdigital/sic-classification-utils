"""Tests for creating a respondent data dictionary."""

import pytest

from industrial_classification_utils.utils.prep_respondent_data import (
    respondent_data_to_dict,
    respondent_data_to_multiline_string,
)


@pytest.mark.parametrize(
    "education, jd, jt, ind, expected",
    [
        (
            "ed1",
            "jd1",
            "jt1",
            "ind1",
            {
                "Level of education": "ed1",
                "Job description": "jd1",
                "Job title": "jt1",
                "Company main activity": "ind1",
            },
        ),
        (
            "5",
            "jd2",
            "jt2",
            "ind2",
            {
                "Level of education": "5",
                "Job description": "jd2",
                "Job title": "jt2",
                "Company main activity": "ind2",
            },
        ),
    ],
)
def test_all_fields(education, jd, jt, ind, expected):
    """Test case, where all fields are provided.
    Expecting creating two dictionaries with four keys each.
    """
    result = respondent_data_to_dict(
        industry_descr=ind,
        job_title=jt,
        job_description=jd,
        level_of_education=education,
    )

    assert expected == result


@pytest.mark.parametrize(
    "education, jd, jt, ind, expected",
    [
        (None, None, None, None, {}),
        ("unknown", "unknown", "unknown", "unknown", {}),
        ("", "", "", "", {}),
        (" ", " ", " ", " ", {}),
        (" ", "-8", "-8", "-8", {}),
        ("", "-9", "-9", "-9", {}),
    ],
)
def test_answer_not_provided(education, jd, jt, ind, expected):
    """Test case with no answers provided (marked as None, "unknown", "", " ", "-8", or "-9",
    for education None, "unknown", "", " ").
    Expecting returning empty dictionaries.
    """
    result = respondent_data_to_dict(
        industry_descr=ind,
        job_title=jt,
        job_description=jd,
        level_of_education=education,
    )

    assert expected == result


@pytest.mark.parametrize(
    "education, jd, jt, ind, expected",
    [
        ("edu1", "-9", "", "unknown", {"Level of education": "edu1"}),
        ("", "jd2", "", "unknown", {"Job description": "jd2"}),
        ("", "", "jt3", "unknown", {"Job title": "jt3"}),
        (None, "", "unknown", "ind4", {"Company main activity": "ind4"}),
    ],
)
def test_some_answers_provided(education, jd, jt, ind, expected):
    """Test cases, where some responses are provided.
    Expecting dictionaries with only fields that were provided.
    """
    result = respondent_data_to_dict(
        industry_descr=ind,
        job_title=jt,
        job_description=jd,
        level_of_education=education,
    )

    assert expected == result


@pytest.mark.parametrize(
    "respondent_dictionary, expected",
    [
        (
            {
                "Level of education": "ed1",
                "Job description": "jd1",
                "Job title": "jt1",
                "Company main activity": "ind1",
            },
            """    - Level of education: ed1
    - Job description: jd1
    - Job title: jt1
    - Company main activity: ind1""",
        ),
        ({"Level of education": "edu1"}, "    - Level of education: edu1"),
        ({}, ""),
    ],
)
def test_converts_to_string(respondent_dictionary, expected):
    """Test if outputs are strings."""
    result = respondent_data_to_multiline_string(respondent_dictionary)

    assert expected == result
    assert isinstance(result, str)
