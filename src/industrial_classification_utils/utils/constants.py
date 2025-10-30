"""Module for common constant definitions.

This module contains constants used across the industrial classification utilities.
"""

MAX_ALT_CANDIDATES = 10
DEFAULT_TRUNCATE_LEN = 12


def truncate_identifier(value: str | None, max_len: int = DEFAULT_TRUNCATE_LEN) -> str:
    """Return a truncated string safely, handling None and short values.

    Used for logging to preserve privacy while providing enough context.

    Args:
        value (str | None): The string to truncate.
        max_len (int): Maximum length before truncation. Defaults to 12.

    Returns:
        str: Empty string if value is None/empty, otherwise truncated string
            with "..." suffix if longer than max_len.
    """
    if not value:
        return ""
    return value if len(value) <= max_len else value[:max_len] + "..."
