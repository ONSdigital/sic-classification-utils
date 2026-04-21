"""Module for common constant definitions.

This module contains constants used across the industrial classification utilities.
"""

from industrial_classification_utils.models.config_model import FullConfig

MAX_ALT_CANDIDATES = 10
DEFAULT_TRUNCATE_LEN = 8


def truncate_identifier(value: str | None, max_len: int = DEFAULT_TRUNCATE_LEN) -> str:
    """Return a truncated string safely, handling None and short values.

    Used for logging to preserve privacy while providing enough context.

    Args:
        value (str | None): The string to truncate.
        max_len (int): Maximum length before truncation. Defaults to 8.

    Returns:
        str: Empty string if value is None/empty, otherwise truncated string
            with "..." suffix if longer than max_len.
    """
    if not value:
        return ""
    return value if len(value) <= max_len else value[:max_len] + "..."


def get_default_config() -> FullConfig:
    """Returns the configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "embedding": {
            "embedding_model_name": "all-MiniLM-L6-v2",  # text-embedding-004
            "db_dir": "src/industrial_classification_utils/data/vector_store",
            "k_matches": 20,
        },
        "llm": {
            "llm_model_name": "gemini-2.5-flash",
            "model_location": "europe-west2",
            "code_digits": 5,
            "candidates_limit": 10,
        },
        "lookups": {
            "sic_index": (
                "industrial_classification_utils.data.sic_index",
                # "extended_SIC_index.xlsx",
                "uksic2007indexeswithaddendumdecember2022.xlsx",
            ),
            "sic_structure": (
                "industrial_classification_utils.data.sic_index",
                "publisheduksicsummaryofstructureworksheet.xlsx",
            ),
            "sic_condensed": (
                "industrial_classification_utils.data.example",
                "sic_2d_condensed.txt",
            ),
        },
    }
