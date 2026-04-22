"""Tests for SAYT configuration validation."""

import pytest
from pydantic import ValidationError

from industrial_classification_utils.sayt.sayt_config import SaytConfig


@pytest.mark.parametrize(
    "kwargs, exc_type",
    [
        ({"min_chars": 2}, ValidationError),
        ({"min_chars": True}, ValidationError),
        ({"max_suggestions": 0}, ValidationError),
        ({"max_suggestions": 101}, ValidationError),
        ({"ngram_enable": True, "ngram_n": 1}, ValidationError),
        ({"ngram_enable": True, "ngram_n": 6}, ValidationError),
        ({"ngram_enable": True, "ngram_max_df": 0.0}, ValidationError),
        ({"ngram_enable": True, "ngram_max_df": 1.1}, ValidationError),
        ({"ngram_enable": True, "ngram_weight": 0.0}, ValueError),
    ],
)
def test_config_validation(kwargs, exc_type):
    """Reject unsupported SAYT config values and types."""
    with pytest.raises(exc_type):
        SaytConfig.model_validate(kwargs)
