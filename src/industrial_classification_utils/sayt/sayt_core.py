"""Core SAYT data models, corpus cleaning, and ranking helpers."""

# ruff: noqa: PLR2004

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z ]+")


def _normalise(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _row_uid(index: int, search_text: str, display_text: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{index}\0{search_text}\0{display_text}"))


@dataclass(frozen=True, slots=True)
class PersistedCorpusRow:
    """Represent a persisted SAYT corpus row restored from artifact storage."""

    row_id: str
    search_text: str
    display_text: str


class CleanCorpus(BaseModel):
    """Store cleaned SAYT rows and their derived lookup tables.

    Instances are created from raw strings or ``(search_text, display_text)``
    pairs and retain stable row identifiers for downstream score aggregation.
    """

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
            cast(Iterable[str] | Iterable[tuple[object, object]], self.corpus)
        )
        return self._populate_indexes()

    def _populate_indexes(self) -> "CleanCorpus":
        """Rebuild lookup tables from the current cleaned rows."""
        self.id_to_search = {rid: search for rid, search, _ in self.rows}
        self.id_to_display = {rid: display for rid, _, display in self.rows}
        self.display_text_count = {}
        for _, _, display in self.rows:
            self.display_text_count[display] = (
                self.display_text_count.get(display, 0) + 1
            )
        self.size = len(self.rows)
        return self

    @classmethod
    def from_persisted_rows(
        cls,
        rows: Iterable[PersistedCorpusRow | tuple[str, str, str]],
    ) -> "CleanCorpus":
        """Restore a cleaned corpus from persisted row identifiers and text.

        Args:
            rows: Persisted ``(row_id, search_text, display_text)`` triples or
                ``PersistedCorpusRow`` objects.

        Returns:
            A ``CleanCorpus`` whose row identifiers and lookup maps match the
            persisted artifact data exactly.

        Raises:
            ValueError: If no persisted rows are supplied.
        """
        restored_rows = [cls._coerce_persisted_row(row) for row in rows]
        if not restored_rows:
            raise ValueError("corpus is empty after filtering")

        corpus = cls.model_construct(
            corpus=[
                (search_text, display_text)
                for _, search_text, display_text in restored_rows
            ],
            rows=restored_rows,
            id_to_search={},
            id_to_display={},
            display_text_count={},
            size=0,
        )
        return corpus._populate_indexes()

    @staticmethod
    def _coerce_persisted_row(
        row: PersistedCorpusRow | tuple[str, str, str],
    ) -> tuple[str, str, str]:
        """Convert persisted row data into the internal tuple format."""
        if isinstance(row, PersistedCorpusRow):
            return (row.row_id, row.search_text, row.display_text)
        row_id, search_text, display_text = row
        return (str(row_id), str(search_text), str(display_text))

    @staticmethod
    def _clean_corpus(
        corpus: Iterable[str] | Iterable[tuple[object, object]],
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


def _coerce_sayt_int_setting(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"{field_name} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def validate_min_chars(value: object) -> int:
    """Validate the global SAYT minimum query length setting."""
    min_chars = _coerce_sayt_int_setting(value, field_name="min_chars")
    if min_chars < 3:
        raise ValueError("min_chars must be >= 3")
    return min_chars


def validate_max_suggestions(value: object) -> int:
    """Validate the global SAYT maximum suggestion count setting."""
    max_suggestions = _coerce_sayt_int_setting(value, field_name="max_suggestions")
    if not 1 <= max_suggestions <= 100:
        raise ValueError("max_suggestions must be between 1 and 100")
    return max_suggestions


class SaytGlobalSettings(BaseModel):
    """Describe suggester-wide runtime settings."""

    model_config = ConfigDict(extra="forbid")

    min_chars: int
    max_suggestions: int


class SaytCorpusSummary(BaseModel):
    """Summarise the cleaned corpus bound to a suggester."""

    model_config = ConfigDict(extra="forbid")

    size: int
    unique_display_texts: int
    max_duplication: int


class SaytRetrieverArtifactProvenance(BaseModel):
    """Capture persisted artifact details for one retriever entry."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SaytArtifactProvenance(BaseModel):
    """Describe the artifact source of a suggester restored from disk."""

    model_config = ConfigDict(extra="forbid")

    artifact_dir: str
    artifact_type: str
    artifact_version: int
    corpus_file: str
    corpus_size: int


class SaytRetrieverSummary(BaseModel):
    """Summarise one configured retriever within a suggester."""

    model_config = ConfigDict(extra="forbid")

    name: str
    spec_type: str
    retriever_type: str
    configured_weight: float
    normalised_weight: float
    config: dict[str, Any] = Field(default_factory=dict)
    artifact_provenance: SaytRetrieverArtifactProvenance | None = None


class SaytConfiguration(BaseModel):
    """Return a rich runtime summary of a configured suggester."""

    model_config = ConfigDict(extra="forbid")

    settings: SaytGlobalSettings
    corpus: SaytCorpusSummary
    retrievers: list[SaytRetrieverSummary] = Field(default_factory=list)
    artifact_provenance: SaytArtifactProvenance | None = None


@dataclass(frozen=True, slots=True)
class Suggestion:
    """Represent a SAYT match with score and row metadata.

    The meaning of ``score`` depends on the producer. Concrete retrievers emit
    strategy-local scores, while ``SAYTSuggester.suggest_with_scores`` returns
    the combined weighted score.
    """

    display_text: str
    score: float
    search_text: str = ""
    row_id: str = ""


def take_with_ties(
    items: list[tuple[str, float]],
    limit: int,
) -> list[tuple[str, float]]:
    """Return the first ``limit`` items and any later items tied on score.

    Args:
        items: Scored ``(key, score)`` pairs to rank.
        limit: Maximum number of leading items before tie extension is applied.

    Returns:
        The highest-scoring items up to ``limit``, plus any later items that are
        tied with the cutoff score.
    """
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
