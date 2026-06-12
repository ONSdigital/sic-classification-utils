"""Artifact and storage helpers for SAYT builder and loader paths."""

import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .core import (
    CleanCorpus,
    PersistedCorpusRow,
    validate_max_suggestions,
    validate_min_chars,
)
from .indexes import (
    build_ngram_index,
    build_semantic_index,
    load_ngram_index,
    load_semantic_index,
)
from .retriever_specs import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    Retriever,
    RetrieverSpec,
    SemanticRetrieverSpec,
)
from .retrievers import NgramRetriever, SemanticRetriever

SAYT_ARTIFACT_TYPE = "sayt"
SAYT_ARTIFACT_VERSION = 2
MANIFEST_FILE_NAME = "manifest.json"
CORPUS_FILE_NAME = "corpus.csv"
_ARTIFACT_CORPUS_FIELDS = ["row_id", "search_text", "display_text"]
_RETRIEVERS_DIR_NAME = "retrievers"


@dataclass(frozen=True, slots=True)
class StoredRetrieverSpec:
    """Persisted retriever spec plus its optional filespace path."""

    artifact_type: str
    spec: RetrieverSpec
    config: dict[str, object]
    path: str | None = None


@dataclass(frozen=True, slots=True)
class SaytArtifactManifest:
    """Structured manifest data for a persisted SAYT artifact."""

    min_chars: int
    max_suggestions: int
    corpus_file: str
    corpus_size: int
    retrievers: tuple[StoredRetrieverSpec, ...]


def load_corpus_from_csv(
    file_path: str | os.PathLike[str],
    *,
    search_text_col: str = "title",
    display_text_col: str | None = None,
) -> list[tuple[object, object]]:
    """Load raw corpus tuples from a CSV file.

    Args:
        file_path: Path to the CSV file containing suggestion rows.
        search_text_col: Column containing the searchable text.
        display_text_col: Optional column containing display text. When
            omitted, the search column is reused for display values.

    Returns:
        Raw ``(search_text, display_text)`` tuples suitable for ``CleanCorpus``.

    Raises:
        ValueError: If the requested search or display column is missing.
    """
    df = pd.read_csv(file_path)
    if search_text_col not in df.columns:
        raise ValueError(f"Column '{search_text_col}' not found in CSV")
    if display_text_col is None:
        display_text_col = search_text_col
    if display_text_col not in df.columns:
        raise ValueError(f"Column '{display_text_col}' not found in CSV")
    return list(zip(df[search_text_col], df[display_text_col], strict=False))


def prepare_artifact_dir(
    artifact_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Create or replace the output directory for a SAYT artifact."""
    path = Path(artifact_dir)
    if path.exists():
        if not overwrite:
            raise FileExistsError("Artifact directory already exists")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_artifact_corpus(corpus: CleanCorpus, *, artifact_dir: str | Path) -> Path:
    """Persist cleaned SAYT rows as the artifact corpus source of truth."""
    output_path = Path(artifact_dir) / CORPUS_FILE_NAME
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_ARTIFACT_CORPUS_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "row_id": row_id,
                "search_text": search_text,
                "display_text": display_text,
            }
            for row_id, search_text, display_text in corpus.rows
        )
    return output_path


def read_artifact_corpus(
    *,
    artifact_dir: str | Path,
    corpus_file: str = CORPUS_FILE_NAME,
) -> list[PersistedCorpusRow]:
    """Read persisted corpus rows from a SAYT artifact."""
    corpus_path = Path(artifact_dir) / corpus_file
    if not corpus_path.exists():
        raise FileNotFoundError(f"Artifact corpus file not found: {corpus_path}")

    with open(corpus_path, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            PersistedCorpusRow(
                row_id=row["row_id"],
                search_text=row["search_text"],
                display_text=row["display_text"],
            )
            for row in reader
        ]


def build_artifact_manifest(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    max_suggestions: int,
    retriever_specs: tuple[RetrieverSpec, ...],
) -> SaytArtifactManifest:
    """Build the structured manifest payload for a SAYT artifact."""
    return SaytArtifactManifest(
        min_chars=min_chars,
        max_suggestions=max_suggestions,
        corpus_file=CORPUS_FILE_NAME,
        corpus_size=corpus.size,
        retrievers=tuple(
            _build_stored_retriever(index, spec)
            for index, spec in enumerate(retriever_specs)
        ),
    )


def write_artifact_manifest(
    manifest: SaytArtifactManifest,
    *,
    artifact_dir: str | Path,
) -> Path:
    """Write the manifest for a SAYT artifact."""
    manifest_path = Path(artifact_dir) / MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(_serialise_manifest(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest_path


def read_artifact_manifest(*, artifact_dir: str | Path) -> SaytArtifactManifest:
    """Read and validate a SAYT artifact manifest."""
    manifest_path = Path(artifact_dir) / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != SAYT_ARTIFACT_TYPE:
        raise ValueError("Unsupported artifact type")
    if payload.get("artifact_version") != SAYT_ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version")

    try:
        return SaytArtifactManifest(
            min_chars=validate_min_chars(payload["min_chars"]),
            max_suggestions=validate_max_suggestions(payload["max_suggestions"]),
            corpus_file=str(payload["corpus_file"]),
            corpus_size=int(payload["corpus_size"]),
            retrievers=tuple(
                _deserialise_stored_retriever(item) for item in payload["retrievers"]
            ),
        )
    except KeyError as exc:
        raise ValueError(f"Malformed artifact manifest: missing {exc.args[0]}") from exc


def retriever_filespace_path(
    artifact_dir: str | Path,
    stored_retriever: StoredRetrieverSpec,
) -> Path:
    """Resolve the persisted filespace for a dense retriever entry."""
    if stored_retriever.path is None:
        raise ValueError(
            f"Retriever '{stored_retriever.spec.name}' does not have a stored filespace"
        )
    return Path(artifact_dir) / stored_retriever.path


def build_retriever_artifact(
    *,
    corpus: CleanCorpus,
    stored_retriever: StoredRetrieverSpec,
    artifact_dir: str | Path,
) -> None:
    """Persist built-in retriever assets required by a SAYT artifact."""
    spec = stored_retriever.spec
    if isinstance(spec, PrefixRetrieverSpec):
        return

    if isinstance(spec, NgramRetrieverSpec):
        build_ngram_index(
            corpus,
            n=spec.n,
            max_df=spec.max_df,
            output_dir=retriever_filespace_path(artifact_dir, stored_retriever),
            overwrite=True,
        )
        return

    if isinstance(spec, SemanticRetrieverSpec):
        build_semantic_index(
            corpus,
            model=spec.model,
            output_dir=retriever_filespace_path(artifact_dir, stored_retriever),
            overwrite=True,
        )
        return

    raise TypeError(
        "Only built-in retriever specs can be persisted; " f"got {type(spec).__name__}"
    )


def load_retriever_from_artifact(
    *,
    corpus: CleanCorpus,
    min_chars: int,
    stored_retriever: StoredRetrieverSpec,
    artifact_dir: str | Path,
) -> Retriever:
    """Restore a built-in runtime retriever from persisted artifact state."""
    spec = stored_retriever.spec
    if isinstance(spec, PrefixRetrieverSpec):
        return spec.build(corpus, min_chars=min_chars)

    if isinstance(spec, NgramRetrieverSpec):
        index = load_ngram_index(
            corpus,
            n=spec.n,
            max_df=spec.max_df,
            folder_path=retriever_filespace_path(artifact_dir, stored_retriever),
        )
        return NgramRetriever.from_index(
            corpus,
            min_chars=min_chars,
            index=index,
        )

    if isinstance(spec, SemanticRetrieverSpec):
        index = load_semantic_index(
            corpus,
            model=spec.model,
            folder_path=retriever_filespace_path(artifact_dir, stored_retriever),
        )
        return SemanticRetriever.from_index(
            corpus,
            min_chars=min_chars,
            index=index,
        )

    raise TypeError(
        "Only built-in retriever specs can be restored from artifacts; "
        f"got {type(spec).__name__}"
    )


def _build_stored_retriever(
    index: int,
    spec: RetrieverSpec,
) -> StoredRetrieverSpec:
    if isinstance(spec, PrefixRetrieverSpec):
        return StoredRetrieverSpec(
            artifact_type="prefix",
            spec=spec,
            config={},
            path=None,
        )

    if isinstance(spec, NgramRetrieverSpec):
        return StoredRetrieverSpec(
            artifact_type="ngram",
            spec=spec,
            config={"n": spec.n, "max_df": spec.max_df},
            path=f"{_RETRIEVERS_DIR_NAME}/{index:02d}-{spec.name}",
        )

    if isinstance(spec, SemanticRetrieverSpec):
        return StoredRetrieverSpec(
            artifact_type="semantic",
            spec=spec,
            config={"model": spec.model},
            path=f"{_RETRIEVERS_DIR_NAME}/{index:02d}-{spec.name}",
        )

    raise TypeError(
        "Only built-in retriever specs can be persisted; " f"got {type(spec).__name__}"
    )


def _serialise_manifest(manifest: SaytArtifactManifest) -> dict[str, object]:
    return {
        "artifact_type": SAYT_ARTIFACT_TYPE,
        "artifact_version": SAYT_ARTIFACT_VERSION,
        "min_chars": manifest.min_chars,
        "max_suggestions": manifest.max_suggestions,
        "corpus_file": manifest.corpus_file,
        "corpus_size": manifest.corpus_size,
        "retrievers": [
            _serialise_stored_retriever(stored_retriever)
            for stored_retriever in manifest.retrievers
        ],
    }


def _serialise_stored_retriever(
    stored_retriever: StoredRetrieverSpec,
) -> dict[str, object]:
    return {
        "type": stored_retriever.artifact_type,
        "weight": stored_retriever.spec.weight,
        "path": stored_retriever.path,
        "config": stored_retriever.config,
    }


def _deserialise_stored_retriever(payload: dict[str, object]) -> StoredRetrieverSpec:
    retriever_type = str(payload["type"])
    weight = _coerce_float(payload["weight"], field_name="weight")
    path = payload.get("path")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"Malformed retriever config for type: {retriever_type}")
    spec: RetrieverSpec
    if retriever_type == "prefix":
        spec = PrefixRetrieverSpec(weight=weight)
    elif retriever_type == "ngram":
        spec = NgramRetrieverSpec(
            weight=weight,
            n=_coerce_int(config["n"], field_name="n"),
            max_df=_coerce_float(config["max_df"], field_name="max_df"),
        )
    elif retriever_type == "semantic":
        spec = SemanticRetrieverSpec(
            weight=weight,
            model=str(config["model"]),
        )
    else:
        raise ValueError(f"Unsupported stored retriever type: {retriever_type}")

    return StoredRetrieverSpec(
        artifact_type=retriever_type,
        spec=spec,
        config=dict(config),
        path=str(path) if isinstance(path, str) else None,
    )


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"Malformed integer value for retriever field: {field_name}")
    return int(value)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"Malformed float value for retriever field: {field_name}")
    return float(value)
