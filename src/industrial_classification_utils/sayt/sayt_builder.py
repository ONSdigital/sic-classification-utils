"""Offline artifact builder for persisted SAYT runtime assets."""

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from .sayt_core import CleanCorpus, SaytConfig
from .sayt_retriever_specs import (
    RetrieverSpec,
    default_retriever_specs,
)
from .sayt_storage import (
    build_artifact_manifest,
    build_retriever_artifact,
    load_corpus_from_csv,
    prepare_artifact_dir,
    write_artifact_corpus,
    write_artifact_manifest,
)


class SAYTBuilder:
    """Build a persisted SAYT artifact for later runtime loading."""

    def __init__(
        self,
        corpus: Iterable[tuple[object, object]] | Iterable[str],
        *,
        retrievers: Sequence[RetrieverSpec] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialise an artifact builder from raw corpus input."""
        self._corpus = CleanCorpus.model_validate(corpus)
        self._config = SaytConfig.model_validate(kwargs)
        self._retriever_specs = tuple(
            default_retriever_specs() if retrievers is None else retrievers
        )

    @classmethod
    def from_csv(
        cls,
        file_path: str | os.PathLike,
        *,
        search_text_col: str = "title",
        display_text_col: str | None = None,
        retrievers: Sequence[RetrieverSpec] | None = None,
        **kwargs: object,
    ) -> "SAYTBuilder":
        """Initialise an artifact builder from CSV input."""
        corpus_rows = load_corpus_from_csv(
            file_path,
            search_text_col=search_text_col,
            display_text_col=display_text_col,
        )
        return cls(corpus_rows, retrievers=retrievers, **kwargs)

    def build_artifact(
        self,
        output_dir: str | os.PathLike,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist the current SAYT configuration and dense stores to disk."""
        artifact_dir = prepare_artifact_dir(output_dir, overwrite=overwrite)
        manifest = build_artifact_manifest(
            corpus=self._corpus,
            config=self._config,
            retriever_specs=self._retriever_specs,
        )

        write_artifact_corpus(self._corpus, artifact_dir=artifact_dir)
        for stored_retriever in manifest.retrievers:
            build_retriever_artifact(
                corpus=self._corpus,
                stored_retriever=stored_retriever,
                artifact_dir=artifact_dir,
            )

        write_artifact_manifest(manifest, artifact_dir=artifact_dir)
        return artifact_dir
