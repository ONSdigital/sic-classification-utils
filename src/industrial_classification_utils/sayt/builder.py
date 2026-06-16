"""Offline artifact builder for persisted SAYT runtime assets."""

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from ._base import BaseCorpusBound
from .storage import (
    build_artifact_manifest,
    build_retriever_artifact,
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


class SAYTBuilder(BaseCorpusBound):
    """Build a persisted SAYT artifact for later runtime loading."""

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
