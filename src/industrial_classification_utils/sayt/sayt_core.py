"""Core data structures and validation for SAYT."""

# ruff: noqa: PLR2004

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z ]+")


def _normalise(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _row_uid(index: int, search_text: str, display_text: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{index}\0{search_text}\0{display_text}"))


class CleanCorpus(BaseModel):
    """Validated cleaned corpus and derived lookup tables for SAYT."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    corpus: object
    rows: list[tuple[str, str, str]] = Field(default_factory=list)
    id_to_search: dict[str, str] = Field(default_factory=dict)
    id_to_display: dict[str, str] = Field(default_factory=dict)
    display_text_count: dict[str, int] = Field(default_factory=dict)
    size: int = 0

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, data: object) -> object:
        if isinstance(data, cls | dict):
            return data
        return {"corpus": data}

    @model_validator(mode="after")
    def _build_indexes(self) -> "CleanCorpus":
        self.rows = self._clean_corpus(
            cast(Iterable[str] | Iterable[tuple[str, str]], self.corpus)
        )
        self.id_to_search = {rid: search for rid, search, _ in self.rows}
        self.id_to_display = {rid: display for rid, _, display in self.rows}
        self.display_text_count = {}
        for _, _, display in self.rows:
            self.display_text_count[display] = (
                self.display_text_count.get(display, 0) + 1
            )
        self.size = len(self.rows)
        return self

    @staticmethod
    def _clean_corpus(
        corpus: Iterable[str] | Iterable[tuple[str, str]],
    ) -> list[tuple[str, str, str]]:
        if not isinstance(corpus, Iterable):
            raise TypeError(
                "corpus must be an iterable of strings or (string, original) tuples"
            )
        cleaned: list[tuple[str, str]] = []
        for item in corpus:
            item_tuple = item if isinstance(item, tuple) else (item, item)
            text = _normalise(item_tuple[0])
            if not text or text == "-9":
                warnings.warn(
                    f"Skipping empty or invalid corpus item: {item!r}",
                    stacklevel=2,
                )
                continue
            display = str(item_tuple[1]).strip()
            if pd.isna(item_tuple[1]) or not display:
                warnings.warn(
                    f"Empty display value for item: {item!r}, using search text as display",
                    stacklevel=2,
                )
                display = str(item_tuple[0]).strip()
            cleaned.append((text, display))
        if not cleaned:
            raise ValueError("corpus is empty after filtering")
        return [
            (_row_uid(i, norm, display), norm, display)
            for i, (norm, display) in enumerate(cleaned)
        ]


class SaytConfig(BaseModel):
    """Validated configuration for a SAYT suggester instance."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    min_chars: int = 4
    max_suggestions: int = 10

    @model_validator(mode="after")
    def _validate_ranges(self) -> "SaytConfig":
        """Enforce the supported numeric ranges for SAYT settings."""
        if self.min_chars < 3:
            raise ValueError("min_chars must be >= 3")
        if not 1 <= self.max_suggestions <= 100:
            raise ValueError("max_suggestions must be between 1 and 100")
        return self


@dataclass(frozen=True, slots=True)
class _Suggestion:
    """Internal suggestion record with score and row metadata."""

    display_text: str
    score: float
    search_text: str = ""
    row_id: str = ""


def _take_with_ties(
    items: list[tuple[str, float]],
    limit: int,
) -> list[tuple[str, float]]:
    """Return the first ``limit`` items and any later items tied on score."""
    if limit < 1 or not items:
        return []

    items = sorted(
        items,
        key=lambda kv: (-kv[1],),
    )

    if limit >= len(items):
        return items

    cutoff_score = float(items[limit - 1][1])
    end = limit
    while end < len(items) and float(items[end][1]) == cutoff_score:
        end += 1
    return items[:end]
