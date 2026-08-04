"""Prepare respondent data."""


def respondent_data_to_dict(
    industry_descr: str | None,
    job_title: str | None = None,
    job_description: str | None = None,
    level_of_education: str | None = None,
) -> dict:
    """Prepares a dictionary with collected respondent data.

    Args:
            industry_descr (str): The description of the industry.
            job_title (str, optional): The job title. Defaults to None.
            job_description (str, optional): The job description. Defaults to None.
            level_of_education (str, optional): The level of education. Defaults to None.

    Returns:
        dict: A dictionary containing responses collected from respondent.
    """
    respondent_data = {}

    if not (
        industry_descr is None or industry_descr in {"", " ", "unknown", "-8", "-9"}
    ):
        respondent_data["Company main activity"] = industry_descr
    if not (job_title is None or job_title in {"", " ", "unknown", "-8", "-9"}):
        respondent_data["Job title"] = job_title
    if not (
        job_description is None or job_description in {"", " ", "unknown", "-8", "-9"}
    ):
        respondent_data["Job description"] = job_description
    if not (level_of_education is None or level_of_education in {"", " ", "unknown"}):
        respondent_data["Level of education"] = level_of_education

    return respondent_data


def respondent_data_to_multiline_string(respondent_data: dict) -> str:
    """Allows to convert a dictionary with respondent data to a multiline string,
    such as:
        - Job title: Nurse
        - Job description: I provide medical help to patients in a hospital.

    Args:
        respondent_data (dict): A dictionary containing data collected from the user.

    Returns:
        str: dictionary converted to a multiline string.
    """
    return "\n".join(f"    - {k}: {v}" for k, v in respondent_data.items())
