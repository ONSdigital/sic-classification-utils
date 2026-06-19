"""Shared bootstrap helpers for corpus-bound SAYT classes."""

import os
from collections.abc import Iterable, Sequence

from .core import CleanCorpus, validate_max_suggestions, validate_min_chars
from .retriever_specs import RetrieverSpec, default_retriever_specs
from .storage import load_corpus_from_csv


class BaseCorpusBound:  # pylint: disable=too-few-public-methods
    """Shared corpus/retriever bootstrap for SAYT runtime classes."""

    _corpus: CleanCorpus
    _min_chars: int
    _max_suggestions: int
    _retriever_specs: tuple[RetrieverSpec, ...]

    def __init__(
        self,
        corpus: Iterable[tuple[object, object]] | Iterable[str],
        *,
        retrievers: Sequence[RetrieverSpec] | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> None:
        """Validate and store the shared corpus-bound SAYT configuration."""
        self._corpus = CleanCorpus.model_validate(corpus)
        self._min_chars = validate_min_chars(min_chars)
        self._max_suggestions = validate_max_suggestions(max_suggestions)
        self._retriever_specs = tuple(
            default_retriever_specs() if retrievers is None else retrievers
        )

    @classmethod
    def from_csv[  # pylint: disable=too-many-arguments  # noqa: PLR0913
        CorpusBoundT: "BaseCorpusBound"
    ](
        cls: type[CorpusBoundT],
        file_path: str | os.PathLike,
        *,
        search_text_col: str = "title",
        display_text_col: str | None = None,
        retrievers: Sequence[RetrieverSpec] | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> CorpusBoundT:
        """Build a corpus-bound SAYT object from CSV input."""
        corpus_rows = load_corpus_from_csv(
            file_path,
            search_text_col=search_text_col,
            display_text_col=display_text_col,
        )
        return cls(
            corpus_rows,
            retrievers=retrievers,
            min_chars=min_chars,
            max_suggestions=max_suggestions,
        )
