"""Configuration for Search-as-you-type (SAYT) utilities.

This module provides a pydantic model `SaytConfig` that defines and validates the configuration
parameters for SAYT suggesters. It includes type coercion for various field types and range
validation to ensure that the configuration is valid before being used in a suggester
implementation.
"""

# ruff: noqa: PLR2004

from classifai.vectorisers import VectoriserBase
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)


class SaytConfig(BaseModel):
    """Validated configuration for a SAYT suggester instance.

    The model accepts a small amount of input coercion for constructor kwargs
    and enforces the value ranges used by the suggester implementation.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    min_chars: int = 4
    max_suggestions: int = 10
    prefix_enable: bool = True
    prefix_weight: float = 1.0
    ngram_enable: bool = True
    ngram_weight: float = 1.0
    ngram_n: int = 3
    ngram_max_df: float = 0.2
    semantic_enable: bool = True
    semantic_weight: float = 1.0
    semantic_handler: VectoriserBase | str | None = None

    @field_validator("min_chars", "max_suggestions", "ngram_n", mode="before")
    @classmethod
    def _validate_int_fields(cls, value: object, info: ValidationInfo) -> int:
        """Coerce configured integer fields while rejecting bool inputs."""
        return _coerce_int(value, key=info.field_name or "value")

    @field_validator(
        "prefix_weight",
        "ngram_weight",
        "ngram_max_df",
        "semantic_weight",
        mode="before",
    )
    @classmethod
    def _validate_float_fields(cls, value: object, info: ValidationInfo) -> float:
        """Coerce configured float fields from numeric-like inputs."""
        return _coerce_float(value, key=info.field_name or "value")

    @field_validator(
        "prefix_enable",
        "ngram_enable",
        "semantic_enable",
        mode="before",
    )
    @classmethod
    def _validate_bool_fields(cls, value: object, info: ValidationInfo) -> bool:
        """Coerce configured boolean fields from common truthy and falsy values."""
        return _coerce_bool(value, key=info.field_name or "value")

    @model_validator(mode="after")
    def _validate_ranges(self) -> "SaytConfig":
        """Enforce the supported numeric ranges for SAYT settings."""
        if self.min_chars < 3:
            raise ValueError("min_chars must be >= 3")
        if not 1 <= self.max_suggestions <= 100:
            raise ValueError("max_suggestions must be between 1 and 100")
        if self.ngram_enable and not 2 <= self.ngram_n <= 5:
            raise ValueError("ngram_n must be between 2 and 5")
        if self.ngram_enable and not 0.0 < self.ngram_max_df <= 1.0:
            raise ValueError("ngram_max_df must be in (0, 1]")
        return self


def _coerce_int(value: object, *, key: str) -> int:
    """Convert a numeric-like input to an integer for config validation."""
    # bool is a subclass of int; treat it as invalid here.
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an int, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    raise TypeError(f"{key} must be an int")


def _coerce_float(value: object, *, key: str) -> float:
    """Convert a numeric-like input to a float for config validation."""
    if isinstance(value, bool):
        raise TypeError(f"{key} must be a float, not bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:  # pragma: no cover
            raise TypeError(f"{key} must be a float") from exc
    raise TypeError(f"{key} must be a float")


def _coerce_bool(value: object, *, key: str) -> bool:
    """Convert common boolean-like inputs to a strict bool value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    raise TypeError(f"{key} must be a bool")
