"""Retriever implementations used by SAYT suggesters."""

# pylint: disable=too-few-public-methods, R0801

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher

from .core import CleanCorpus, Suggestion, take_with_ties
from .indexes import DenseVectorIndex, build_ngram_index, build_semantic_index

_FUZZY_PREFIX_MIN_RATIO = 0.75


@dataclass(frozen=True, slots=True)
class _PrefixIndex:
    """Precomputed prefix lookup structures for prefix matching."""

    sorted_terms: list[tuple[str, str]]
    prefix_terms: list[str]
    token_index: dict[str, set[str]]


class PrefixRetriever:
    """Retrieve suggestions using exact, token, and fuzzy prefix matching."""

    def __init__(self, corpus: CleanCorpus, *, min_chars: int) -> None:
        """Initialise a prefix retriever.

        Args:
            corpus: Cleaned corpus to search.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = self._build_index(corpus)

    @staticmethod
    def _build_index(corpus: CleanCorpus) -> _PrefixIndex:
        """Precompute sorted prefix terms and token-prefix lookup tables."""
        sorted_terms: list[tuple[str, str]] = []
        token_index: dict[str, set[str]] = {}

        for row_id, search_norm, _ in corpus.rows:
            sorted_terms.append((search_norm, row_id))
            for token in search_norm.split():
                for i in range(1, min(len(token), len(search_norm)) + 1):
                    prefix_str = token[:i]
                    token_index.setdefault(prefix_str, set()).add(row_id)

        sorted_terms.sort(key=lambda x: x[0])
        prefix_terms = [s for s, _ in sorted_terms]
        return _PrefixIndex(
            sorted_terms=sorted_terms,
            prefix_terms=prefix_terms,
            token_index=token_index,
        )

    def suggest_with_scores(
        self, q_norm: str, num_suggestions: int
    ) -> list[Suggestion]:
        """Return ranked prefix-based suggestions for a normalised query.

        Args:
            q_norm: Normalised query text.
            num_suggestions: Maximum number of scored suggestions to return
                before tie expansion.

        Returns:
            Ranked ``Suggestion`` objects scored by prefix heuristics.
        """
        if len(q_norm) < self._min_chars:
            return []

        scores: dict[str, float] = {}

        left = bisect_left(self._index.prefix_terms, q_norm)
        right = bisect_right(self._index.prefix_terms, q_norm + "\uffff")
        for _, row_id in self._index.sorted_terms[left:right]:
            scores[row_id] = scores.get(row_id, 0.0) + 3.0

        for row_id in self._index.token_index.get(q_norm, set()):
            scores[row_id] = scores.get(row_id, 0.0) + 2.5

        for search_norm, row_id in self._index.sorted_terms:
            prefix = search_norm[: len(q_norm)]
            if not prefix:
                continue
            ratio = SequenceMatcher(a=q_norm, b=prefix).ratio()
            if ratio >= _FUZZY_PREFIX_MIN_RATIO:
                scores[row_id] = scores.get(row_id, 0.0) + (2.4 * ratio)

        ranked = take_with_ties(list(scores.items()), limit=num_suggestions)
        return [
            Suggestion(
                display_text=self._corpus.id_to_display.get(row_id, ""),
                score=float(score),
                search_text=self._corpus.id_to_search.get(row_id, ""),
                row_id=row_id,
            )
            for row_id, score in ranked
        ]


class _DenseRetriever:
    """Shared cosine-similarity retrieval over an in-memory dense index."""

    _corpus: CleanCorpus
    _min_chars: int
    _index: DenseVectorIndex

    @classmethod
    def from_index(
        cls,
        corpus: CleanCorpus,
        *,
        min_chars: int,
        index: DenseVectorIndex,
    ) -> "_DenseRetriever":
        """Restore a dense retriever from an already-built dense index."""
        retriever = cls.__new__(cls)
        retriever._corpus = corpus
        retriever._min_chars = min_chars
        retriever._index = index
        return retriever

    def suggest_with_scores(
        self, q_norm: str, num_suggestions: int
    ) -> list[Suggestion]:
        """Return dense-vector matches after applying retriever-level gating."""
        if len(q_norm) < self._min_chars:
            return []
        return [
            Suggestion(
                display_text=self._corpus.id_to_display.get(row_id, ""),
                score=score,
                search_text=self._corpus.id_to_search.get(row_id, ""),
                row_id=row_id,
            )
            for row_id, score in self._index.query(q_norm, num_suggestions)
        ]


class NgramRetriever(_DenseRetriever):
    """Retrieve suggestions using character n-gram similarity."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        n: int,
        max_df: float,
        min_chars: int,
    ) -> None:
        """Initialise a character n-gram retriever.

        Args:
            corpus: Cleaned corpus to search.
            n: Character n-gram size.
            max_df: Maximum document frequency passed to the n-gram vectoriser.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = build_ngram_index(
            corpus=corpus,
            n=n,
            max_df=max_df,
        )


class SemanticRetriever(_DenseRetriever):
    """Retrieve suggestions using sentence-transformer embeddings."""

    def __init__(
        self,
        corpus: CleanCorpus,
        *,
        model: str,
        min_chars: int,
    ) -> None:
        """Initialise a semantic retriever.

        Args:
            corpus: Cleaned corpus to search.
            model: Sentence-transformer model name without the repository
                prefix.
            min_chars: Minimum query length required before retrieval runs.
        """
        self._corpus = corpus
        self._min_chars = min_chars
        self._index = build_semantic_index(
            corpus=corpus,
            model=model,
        )
