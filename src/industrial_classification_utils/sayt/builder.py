"""Offline artifact builder for persisted SAYT runtime assets."""

# pylint: disable=duplicate-code

import os
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from uuid import uuid4

from .core import CleanCorpus, validate_max_suggestions, validate_min_chars
from .retriever_specs import (
    RetrieverSpec,
    default_retriever_specs,
)
from .storage import (
    build_artifact_manifest,
    build_retriever_artifact,
    load_corpus_from_csv,
    write_artifact_corpus,
    write_artifact_manifest,
)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


class SAYTBuilder:
    """Build a persisted SAYT artifact for later runtime loading."""

    def __init__(
        self,
        corpus: Iterable[tuple[object, object]] | Iterable[str],
        *,
        retrievers: Sequence[RetrieverSpec] | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> None:
        """Initialise an artifact builder from raw corpus input."""
        self._corpus = CleanCorpus.model_validate(corpus)
        self._min_chars = validate_min_chars(min_chars)
        self._max_suggestions = validate_max_suggestions(max_suggestions)
        self._retriever_specs = tuple(
            default_retriever_specs() if retrievers is None else retrievers
        )

    @classmethod
    def from_csv(  # pylint: disable=too-many-arguments  # noqa: PLR0913
        cls,
        file_path: str | os.PathLike,
        *,
        search_text_col: str = "title",
        display_text_col: str | None = None,
        retrievers: Sequence[RetrieverSpec] | None = None,
        min_chars: int = 4,
        max_suggestions: int = 10,
    ) -> "SAYTBuilder":
        """Initialise an artifact builder from CSV input."""
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

    def build_artifact(
        self,
        output_dir: str | os.PathLike,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist the current SAYT configuration and dense stores to disk."""
        artifact_dir = Path(output_dir)
        if artifact_dir.exists() and not overwrite:
            raise FileExistsError("Artifact directory already exists")

        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{artifact_dir.name}.tmp-",
                dir=artifact_dir.parent,
            )
        )

        try:
            manifest = build_artifact_manifest(
                corpus=self._corpus,
                min_chars=self._min_chars,
                max_suggestions=self._max_suggestions,
                retriever_specs=self._retriever_specs,
            )

            write_artifact_corpus(self._corpus, artifact_dir=staged_dir)
            for stored_retriever in manifest.retrievers:
                build_retriever_artifact(
                    corpus=self._corpus,
                    min_chars=self._min_chars,
                    stored_retriever=stored_retriever,
                    artifact_dir=staged_dir,
                )

            write_artifact_manifest(manifest, artifact_dir=staged_dir)

            if artifact_dir.exists():
                backup_dir = (
                    artifact_dir.parent / f".{artifact_dir.name}.bak-{uuid4().hex}"
                )
                artifact_dir.rename(backup_dir)
                try:
                    staged_dir.rename(artifact_dir)
                except Exception:
                    backup_dir.rename(artifact_dir)
                    raise
                _remove_path(backup_dir)
            else:
                staged_dir.rename(artifact_dir)
        except Exception:
            _remove_path(staged_dir)
            raise

        return artifact_dir
