# pylint: disable=too-few-public-methods

"""Retriever composition surface for SAYT."""

from dataclasses import dataclass, field
from typing import Protocol

from .sayt_core import CleanCorpus, Suggestion
from .sayt_retrievers import NgramRetriever, PrefixRetriever, SemanticRetriever

_MIN_NGRAM_SIZE = 2
_MAX_NGRAM_SIZE = 5


class Retriever(Protocol):
    """Query contract used by the SAYT orchestrator."""

    def suggest_with_scores(
        self, q_norm: str, num_suggestions: int
    ) -> list[Suggestion]:
        """Return scored suggestions for a normalised query string."""


class RetrieverSpec(Protocol):
    """Configuration plus builder for a corpus-bound retriever instance."""

    @property
    def name(self) -> str:
        """Return the stable identifier for this retriever configuration."""

    @property
    def weight(self) -> float:
        """Return the contribution weight applied during score combination."""

    def build(self, corpus: CleanCorpus, *, min_chars: int) -> Retriever:
        """Build a corpus-bound retriever instance from this configuration."""


def _validate_retriever_weight(weight: float) -> None:
    if weight <= 0:
        raise ValueError("retriever weight must be > 0")


@dataclass(frozen=True, slots=True)
class PrefixRetrieverSpec:
    """Configuration for building a prefix retriever."""

    weight: float = 1.0
    name: str = field(init=False, default="prefix")

    def __post_init__(self) -> None:
        """Validate configuration values after dataclass initialisation."""
        _validate_retriever_weight(self.weight)

    def build(self, corpus: CleanCorpus, *, min_chars: int) -> Retriever:
        """Build a prefix retriever for the provided cleaned corpus."""
        return PrefixRetriever(corpus, min_chars=min_chars)


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

    def build(self, corpus: CleanCorpus, *, min_chars: int) -> Retriever:
        """Build a character n-gram retriever for the provided corpus."""
        if self.max_df * corpus.size < 1:
            raise ValueError("ngram max_df is too low for the given corpus")
        return NgramRetriever(
            corpus,
            n=self.n,
            max_df=self.max_df,
            min_chars=min_chars,
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

    def build(self, corpus: CleanCorpus, *, min_chars: int) -> Retriever:
        """Build a semantic retriever for the provided cleaned corpus."""
        return SemanticRetriever(corpus, model=self.model, min_chars=min_chars)


def default_retriever_specs() -> list[RetrieverSpec]:
    """Return the standard runtime retriever set used by SAYT."""
    return [
        PrefixRetrieverSpec(),
        NgramRetrieverSpec(),
        SemanticRetrieverSpec(),
    ]
