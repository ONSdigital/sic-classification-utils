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


def test_disabled_modules_clear_module_specific_values():
    """Disabled modules should store None for their module-specific config."""
    config = SaytConfig.model_validate(
        {
            "ngram_enable": False,
            "semantic_enable": False,
        }
    )

    assert config.prefix_weight == pytest.approx(1.0)
    assert config.ngram_weight == pytest.approx(0.0)
    assert config.ngram_n is None
    assert config.ngram_max_df is None
    assert config.semantic_weight == pytest.approx(0.0)
    assert config.semantic_model is None


def test_disabled_modules_allow_none_inputs_for_non_weight_values():
    """Explicit None values are valid for disabled module-specific non-weight values."""
    config = SaytConfig.model_validate(
        {
            "ngram_enable": False,
            "ngram_n": None,
            "ngram_max_df": None,
            "semantic_enable": False,
            "semantic_model": None,
        }
    )

    assert config.prefix_weight == pytest.approx(1.0)
    assert config.ngram_weight == pytest.approx(0.0)
    assert config.ngram_n is None
    assert config.ngram_max_df is None
    assert config.semantic_weight == pytest.approx(0.0)
    assert config.semantic_model is None


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ngram_enable": True, "ngram_n": None}, "ngram_n must be set"),
        ({"ngram_enable": True, "ngram_max_df": None}, "ngram_max_df must be set"),
        (
            {"ngram_enable": True, "ngram_max_df": 0.2, "corpus_size": 4},
            "ngram_max_df is too low",
        ),
        (
            {
                "ngram_enable": True,
                "ngram_weight": 0.0,
                "ngram_max_df": 1.0,
                "corpus_size": 1,
            },
            "ngram_weight must be > 0",
        ),
        (
            {
                "ngram_enable": False,
                "semantic_enable": True,
                "semantic_weight": 0.0,
            },
            "semantic_weight must be > 0",
        ),
        (
            {
                "prefix_enable": False,
                "ngram_enable": False,
                "semantic_enable": False,
            },
            "At least one of prefix, ngram, or semantic must be enabled",
        ),
    ],
)
def test_config_additional_validation_errors(kwargs, match):
    """Reject unsupported edge cases for optional modules and weights."""
    with pytest.raises(ValueError, match=match):
        SaytConfig.model_validate(kwargs)


def test_disabled_prefix_and_ngram_zero_their_weights():
    """Disabled modules should always normalise to zero weight."""
    config = SaytConfig.model_validate(
        {
            "prefix_enable": False,
            "prefix_weight": 99.0,
            "ngram_enable": False,
            "ngram_weight": 99.0,
            "semantic_enable": True,
            "semantic_weight": 2.0,
        }
    )

    assert config.prefix_weight == pytest.approx(0.0)
    assert config.ngram_weight == pytest.approx(0.0)
    assert config.semantic_weight == pytest.approx(1.0)


def test_prefix_weight_must_be_positive_when_prefix_enabled():
    """Reject non-positive prefix weight while prefix retrieval is enabled."""
    with pytest.raises(ValueError, match="prefix_weight must be > 0"):
        SaytConfig.model_validate(
            {
                "prefix_enable": True,
                "prefix_weight": 0.0,
                "ngram_enable": False,
            }
        )
