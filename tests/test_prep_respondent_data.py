"""Tests for creating a respondent data dictionary."""

import pytest

from industrial_classification_utils.utils.prep_respondent_data import (
    respondent_data_to_dict,
)


@pytest.mark.parametrize(
    "education, jd, jt, ind, expected",
    [
        (
            -9,
            "jd1",
            "jt1",
            "ind1",
            {
                "Level of education": -9,
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
        ("-8", "-8", "-8", "-8", {}),
        ("-9", "-9", "-9", "-9", {}),
    ],
)
def test_answer_not_provided(education, jd, jt, ind, expected):
    """Test case with no answers provided (marked as None, "unknown", "", "-8", or "-9").
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
        ("-9", "jd2", "", "unknown", {"Job description": "jd2"}),
        ("-9", "", "jt3", "unknown", {"Job title": "jt3"}),
        ("-9", "", "unknown", "ind4", {"Company main activity": "ind4"}),
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
