# pylint: disable=too-few-public-methods

"""Public retriever protocols and configuration objects for SAYT."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .core import CleanCorpus, Suggestion
from .indexes import (
    build_ngram_index,
    build_semantic_index,
    load_ngram_index,
    load_semantic_index,
)
from .retrievers import NgramRetriever, PrefixRetriever, SemanticRetriever

_MIN_NGRAM_SIZE = 2
_MAX_NGRAM_SIZE = 5


class Retriever(Protocol):
    """Query contract used by the SAYT orchestrator."""

    def suggest_with_scores(
        self, q_norm: str, num_suggestions: int
    ) -> list[Suggestion]:
        """Return scored suggestions for a normalised query string.

        Args:
            q_norm: Normalised query text.
            num_suggestions: Maximum number of scored suggestions to return.

        Returns:
            Ranked ``Suggestion`` objects for the query.
        """


class RetrieverSpec(Protocol):
    """Configuration plus builder for a corpus-bound retriever instance."""

    @property
    def name(self) -> str:
        """Return the stable identifier for this retriever configuration."""

    @property
    def weight(self) -> float:
        """Return the finite positive weight applied during score combination."""

    def build(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> Retriever:
        """Build a corpus-bound retriever instance from this configuration.

        Args:
            corpus: Cleaned corpus to bind to the retriever.
            min_chars: Minimum query length required before retrieval runs.
            filespace_path: Optional persisted filespace directory used when a
                caller wants the build outputs written to disk.
            overwrite: Whether an existing filespace may be replaced when
                ``filespace_path`` is provided.

        Returns:
            A configured retriever instance bound to ``corpus``.
        """


@runtime_checkable
class ArtifactRetrieverSpec(RetrieverSpec, Protocol):
    """Optional artifact-loading capability for persisted retriever specs."""

    def load_from_artifact(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None,
    ) -> Retriever:
        """Restore a runtime retriever from persisted artifact state."""


def _validate_retriever_weight(weight: float) -> None:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("retriever weight must be a finite value > 0")


def _require_filespace_path(
    filespace_path: str | Path | None,
    *,
    spec_name: str,
) -> str | Path:
    if filespace_path is None:
        raise ValueError(f"Retriever '{spec_name}' does not have a stored filespace")
    return filespace_path


@dataclass(frozen=True, slots=True)
class PrefixRetrieverSpec:
    """Configuration for building a prefix retriever."""

    weight: float = 1.0
    name: str = field(init=False, default="prefix")

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialisation."""
        _validate_retriever_weight(self.weight)

    def build(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> Retriever:
        """Build a prefix retriever for the provided cleaned corpus.

        Args:
            corpus: Cleaned corpus to search.
            min_chars: Minimum query length required before retrieval runs.
            filespace_path: Optional persisted filespace path. Prefix
                retrievers ignore this because they do not store dense state.
            overwrite: Ignored for prefix retrievers.

        Returns:
            A configured ``PrefixRetriever``.
        """
        _ = (filespace_path, overwrite)
        return PrefixRetriever(corpus, min_chars=min_chars)

    def load_from_artifact(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None,
    ) -> Retriever:
        """Restore a prefix retriever directly from its runtime config."""
        _ = filespace_path
        return self.build(corpus, min_chars=min_chars)


@dataclass(frozen=True, slots=True)
class NgramRetrieverSpec:
    """Configuration for building a character n-gram retriever."""

    weight: float = 1.0
    n: int = 3
    max_df: float = 0.2
    name: str = field(init=False, default="ngram")

    def __post_init__(self) -> None:
        """Validate n-gram configuration values after initialisation."""
        _validate_retriever_weight(self.weight)
        if not _MIN_NGRAM_SIZE <= self.n <= _MAX_NGRAM_SIZE:
            raise ValueError("ngram n must be between 2 and 5")
        if not 0.0 < self.max_df <= 1.0:
            raise ValueError("ngram max_df must be in (0, 1]")

    def build(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> Retriever:
        """Build a character n-gram retriever for the provided corpus.

        Args:
            corpus: Cleaned corpus to search.
            min_chars: Minimum query length required before retrieval runs.
            filespace_path: Optional filespace directory. When provided, the
                built dense index is persisted there.
            overwrite: Whether an existing filespace may be replaced when
                ``filespace_path`` is provided.

        Returns:
            A configured ``NgramRetriever``.

        Raises:
            ValueError: If ``max_df`` would remove every n-gram feature from the
                provided corpus.
        """
        if self.max_df * corpus.size < 1:
            raise ValueError("ngram max_df is too low for the given corpus")
        if filespace_path is None:
            return NgramRetriever(
                corpus,
                n=self.n,
                max_df=self.max_df,
                min_chars=min_chars,
            )

        index = build_ngram_index(
            corpus,
            n=self.n,
            max_df=self.max_df,
            output_dir=_require_filespace_path(filespace_path, spec_name=self.name),
            overwrite=overwrite,
        )
        return NgramRetriever.from_index(
            corpus,
            min_chars=min_chars,
            index=index,
        )

    def load_from_artifact(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None,
    ) -> Retriever:
        """Restore an n-gram retriever from its persisted dense filespace."""
        index = load_ngram_index(
            corpus,
            n=self.n,
            max_df=self.max_df,
            folder_path=_require_filespace_path(filespace_path, spec_name=self.name),
        )
        return NgramRetriever.from_index(
            corpus,
            min_chars=min_chars,
            index=index,
        )


@dataclass(frozen=True, slots=True)
class SemanticRetrieverSpec:
    """Configuration for building a semantic retriever."""

    weight: float = 1.0
    model: str = "all-MiniLM-L6-v2"
    name: str = field(init=False, default="semantic")

    def __post_init__(self) -> None:
        """Validate semantic retriever configuration after initialisation."""
        _validate_retriever_weight(self.weight)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("semantic model must be a non-empty string")

    def build(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> Retriever:
        """Build a semantic retriever for the provided cleaned corpus.

        Args:
            corpus: Cleaned corpus to search.
            min_chars: Minimum query length required before retrieval runs.
            filespace_path: Optional filespace directory. When provided, the
                built dense index is persisted there.
            overwrite: Whether an existing filespace may be replaced when
                ``filespace_path`` is provided.

        Returns:
            A configured ``SemanticRetriever``.
        """
        if filespace_path is None:
            return SemanticRetriever(corpus, model=self.model, min_chars=min_chars)

        index = build_semantic_index(
            corpus,
            model=self.model,
            output_dir=_require_filespace_path(filespace_path, spec_name=self.name),
            overwrite=overwrite,
        )
        return SemanticRetriever.from_index(
            corpus,
            min_chars=min_chars,
            index=index,
        )

    def load_from_artifact(
        self,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        filespace_path: str | Path | None,
    ) -> Retriever:
        """Restore a semantic retriever from its persisted dense filespace."""
        index = load_semantic_index(
            corpus,
            model=self.model,
            folder_path=_require_filespace_path(filespace_path, spec_name=self.name),
        )
        return SemanticRetriever.from_index(
            corpus,
            min_chars=min_chars,
            index=index,
        )


def default_retriever_specs() -> list[RetrieverSpec]:
    """Return the standard runtime retriever set used by SAYT.

    Returns:
        The default prefix, character n-gram, and semantic retriever specs.
    """
    return [
        PrefixRetrieverSpec(),
        NgramRetrieverSpec(),
        SemanticRetrieverSpec(),
    ]
