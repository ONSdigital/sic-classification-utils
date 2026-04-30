"""Configuration for Search-as-you-type (SAYT) utilities.

This module provides a pydantic model `SaytConfig` that defines and validates the configuration
parameters for SAYT suggesters. It includes type coercion for various field types and range
validation to ensure that the configuration is valid before being used in a suggester
implementation.
"""

# ruff: noqa: PLR2004

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)


class SaytConfig(BaseModel):
    """Validated configuration for a SAYT suggester instance.

    The model relies on Pydantic's native parsing for constructor kwargs and
    enforces the value ranges used by the suggester implementation.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    min_chars: int = 4
    max_suggestions: int = 10
    prefix_enable: bool = True
    prefix_weight: float = 1.0
    ngram_enable: bool = True
    ngram_weight: float = 1.0
    ngram_n: int | None = 3
    ngram_max_df: float | None = 0.2
    semantic_enable: bool = True
    semantic_weight: float = 1.0
    semantic_model: str | None = "all-MiniLM-L6-v2"
    corpus_size: int = 0

    @model_validator(mode="after")
    def _validate_ranges(self) -> "SaytConfig":
        """Enforce the supported numeric ranges for SAYT settings."""
        if self.min_chars < 3:
            raise ValueError("min_chars must be >= 3")
        if not 1 <= self.max_suggestions <= 100:
            raise ValueError("max_suggestions must be between 1 and 100")

        if not self.ngram_enable:
            self.ngram_n = None
        elif not self.ngram_n:
            raise ValueError("ngram_n must be set when ngram is enabled")
        elif not 2 <= self.ngram_n <= 5:
            raise ValueError("ngram_n must be between 2 and 5")

        if not self.ngram_enable:
            self.ngram_max_df = None
        elif not self.ngram_max_df:
            raise ValueError("ngram_max_df must be set when ngram is enabled")
        elif not 0.0 < self.ngram_max_df <= 1.0:
            raise ValueError("ngram_max_df must be in (0, 1]")
        elif self.ngram_max_df * self.corpus_size < 1:
            raise ValueError("ngram_max_df is too low for the given corpus_size")

        return self

    @model_validator(mode="after")
    def _validate_weights(self) -> "SaytConfig":
        if self.prefix_enable and self.prefix_weight <= 0:
            raise ValueError("prefix_weight must be > 0")
        if not self.prefix_enable:
            self.prefix_weight = 0.0

        if self.ngram_enable and self.ngram_weight <= 0:
            raise ValueError("ngram_weight must be > 0")
        if not self.ngram_enable:
            self.ngram_weight = 0.0

        if self.semantic_enable and self.semantic_weight <= 0:
            raise ValueError("semantic_weight must be > 0")
        if not self.semantic_enable:
            self.semantic_weight = 0.0
            self.semantic_model = None

        total_weight = self.prefix_weight + self.ngram_weight + self.semantic_weight
        if total_weight == 0.0:
            raise ValueError(
                "At least one of prefix, ngram, or semantic must be enabled with a positive weight"
            )
        self.prefix_weight /= total_weight
        self.ngram_weight /= total_weight
        self.semantic_weight /= total_weight
        return self
