"""Artifact and storage helpers for SAYT builder and loader paths."""

import csv
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd

from .sayt_core import CleanCorpus, PersistedCorpusRow, SaytConfig
from .sayt_indexes import (
    build_ngram_index,
    build_semantic_index,
    load_ngram_index,
    load_semantic_index,
)
from .sayt_retriever_specs import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    Retriever,
    RetrieverArtifactHandler,
    RetrieverSpec,
    SemanticRetrieverSpec,
)
from .sayt_retrievers import NgramRetriever, SemanticRetriever

SAYT_ARTIFACT_TYPE = "sayt"
SAYT_ARTIFACT_VERSION = 1
MANIFEST_FILE_NAME = "manifest.json"
CORPUS_FILE_NAME = "corpus.csv"
_ARTIFACT_CORPUS_FIELDS = ["row_id", "search_text", "display_text"]
_RETRIEVERS_DIR_NAME = "retrievers"

_RETRIEVER_ARTIFACT_HANDLERS: dict[str, RetrieverArtifactHandler] = {}
SpecT = TypeVar("SpecT", bound=RetrieverSpec)


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

    config: SaytConfig
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
    config: SaytConfig,
    retriever_specs: tuple[RetrieverSpec, ...],
) -> SaytArtifactManifest:
    """Build the structured manifest payload for a SAYT artifact."""
    return SaytArtifactManifest(
        config=config.model_copy(deep=True),
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
            config=SaytConfig.model_validate(payload["config"]),
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


def register_retriever_artifact_handler(
    handler: RetrieverArtifactHandler,
    *,
    replace: bool = False,
) -> None:
    """Register a handler for artifact persistence of retriever specs."""
    artifact_type = handler.artifact_type
    if artifact_type in _RETRIEVER_ARTIFACT_HANDLERS and not replace:
        raise ValueError(
            f"Retriever artifact handler already registered for type: {artifact_type}"
        )
    _RETRIEVER_ARTIFACT_HANDLERS[artifact_type] = handler


def unregister_retriever_artifact_handler(artifact_type: str) -> None:
    """Remove a previously registered retriever artifact handler."""
    _RETRIEVER_ARTIFACT_HANDLERS.pop(artifact_type, None)


def build_retriever_artifact(
    *,
    corpus: CleanCorpus,
    stored_retriever: StoredRetrieverSpec,
    artifact_dir: str | Path,
) -> None:
    """Persist retriever-specific artifact state using its registered handler."""
    handler = _get_retriever_artifact_handler(stored_retriever.artifact_type)
    path = (
        retriever_filespace_path(artifact_dir, stored_retriever)
        if stored_retriever.path is not None
        else None
    )
    handler.build_artifact(
        spec=stored_retriever.spec,
        corpus=corpus,
        path=path,
    )


def load_retriever_from_artifact(
    *,
    corpus: CleanCorpus,
    config: SaytConfig,
    stored_retriever: StoredRetrieverSpec,
    artifact_dir: str | Path,
) -> Retriever:
    """Restore a runtime retriever using its registered artifact handler."""
    handler = _get_retriever_artifact_handler(stored_retriever.artifact_type)
    path = (
        retriever_filespace_path(artifact_dir, stored_retriever)
        if stored_retriever.path is not None
        else None
    )
    return handler.load_retriever(
        spec=stored_retriever.spec,
        corpus=corpus,
        config=config,
        path=path,
    )


def _build_stored_retriever(
    index: int,
    spec: RetrieverSpec,
) -> StoredRetrieverSpec:
    handler = _get_retriever_artifact_handler_for_spec(spec)
    return StoredRetrieverSpec(
        artifact_type=handler.artifact_type,
        spec=spec,
        config=handler.serialise_spec(spec),
        path=handler.default_path(index=index, spec=spec),
    )


def _serialise_manifest(manifest: SaytArtifactManifest) -> dict[str, object]:
    return {
        "artifact_type": SAYT_ARTIFACT_TYPE,
        "artifact_version": SAYT_ARTIFACT_VERSION,
        "config": manifest.config.model_dump(mode="json"),
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
    handler = _get_retriever_artifact_handler(retriever_type)
    spec = handler.deserialise_spec(weight=weight, config=config)
    return StoredRetrieverSpec(
        artifact_type=retriever_type,
        spec=spec,
        config=dict(config),
        path=str(path) if isinstance(path, str) else None,
    )


def _get_retriever_artifact_handler(artifact_type: str) -> RetrieverArtifactHandler:
    try:
        return _RETRIEVER_ARTIFACT_HANDLERS[artifact_type]
    except KeyError as exc:
        raise ValueError(
            f"No retriever artifact handler registered for type: {artifact_type}"
        ) from exc


def _get_retriever_artifact_handler_for_spec(
    spec: RetrieverSpec,
) -> RetrieverArtifactHandler:
    for handler in reversed(tuple(_RETRIEVER_ARTIFACT_HANDLERS.values())):
        if handler.can_handle(spec):
            return handler
    raise TypeError(
        f"No retriever artifact handler registered for spec type: {type(spec).__name__}"
    )


class _PrefixRetrieverArtifactHandler:  # pylint: disable=missing-function-docstring,useless-return
    """Artifact handler for the built-in prefix retriever spec."""

    artifact_type = "prefix"

    def can_handle(self, spec: RetrieverSpec) -> bool:
        return isinstance(spec, PrefixRetrieverSpec)

    def serialise_spec(self, spec: RetrieverSpec) -> dict[str, object]:
        _ = spec
        return {}

    def deserialise_spec(
        self,
        *,
        weight: float,
        config: Mapping[str, object],
    ) -> RetrieverSpec:
        _ = config
        return PrefixRetrieverSpec(weight=weight)

    def default_path(self, *, index: int, spec: RetrieverSpec) -> str | None:
        _ = (index, spec)
        return None

    def build_artifact(
        self,
        *,
        spec: RetrieverSpec,
        corpus: CleanCorpus,
        path: Path | None,
    ) -> None:
        _ = (spec, corpus, path)

    def load_retriever(
        self,
        *,
        spec: RetrieverSpec,
        corpus: CleanCorpus,
        config: SaytConfig,
        path: Path | None,
    ) -> Retriever:
        _ = path
        return spec.build(corpus, min_chars=config.min_chars)


class _NgramRetrieverArtifactHandler:  # pylint: disable=missing-function-docstring
    """Artifact handler for the built-in n-gram retriever spec."""

    artifact_type = "ngram"

    def can_handle(self, spec: RetrieverSpec) -> bool:
        return isinstance(spec, NgramRetrieverSpec)

    def serialise_spec(self, spec: RetrieverSpec) -> dict[str, object]:
        typed_spec = _require_spec_type(spec, NgramRetrieverSpec)
        return {"n": typed_spec.n, "max_df": typed_spec.max_df}

    def deserialise_spec(
        self,
        *,
        weight: float,
        config: Mapping[str, object],
    ) -> RetrieverSpec:
        return NgramRetrieverSpec(
            weight=weight,
            n=_coerce_int(config["n"], field_name="n"),
            max_df=_coerce_float(config["max_df"], field_name="max_df"),
        )

    def default_path(self, *, index: int, spec: RetrieverSpec) -> str | None:
        return f"{_RETRIEVERS_DIR_NAME}/{index:02d}-{spec.name}"

    def build_artifact(
        self,
        *,
        spec: RetrieverSpec,
        corpus: CleanCorpus,
        path: Path | None,
    ) -> None:
        typed_spec = _require_spec_type(spec, NgramRetrieverSpec)
        build_ngram_index(
            corpus,
            n=typed_spec.n,
            max_df=typed_spec.max_df,
            output_dir=_require_path(path, typed_spec.name),
            overwrite=True,
        )

    def load_retriever(
        self,
        *,
        spec: RetrieverSpec,
        corpus: CleanCorpus,
        config: SaytConfig,
        path: Path | None,
    ) -> Retriever:
        typed_spec = _require_spec_type(spec, NgramRetrieverSpec)
        index = load_ngram_index(
            corpus,
            n=typed_spec.n,
            max_df=typed_spec.max_df,
            folder_path=_require_path(path, typed_spec.name),
        )
        return NgramRetriever.from_index(
            corpus,
            min_chars=config.min_chars,
            index=index,
        )


class _SemanticRetrieverArtifactHandler:  # pylint: disable=missing-function-docstring
    """Artifact handler for the built-in semantic retriever spec."""

    artifact_type = "semantic"

    def can_handle(self, spec: RetrieverSpec) -> bool:
        return isinstance(spec, SemanticRetrieverSpec)

    def serialise_spec(self, spec: RetrieverSpec) -> dict[str, object]:
        typed_spec = _require_spec_type(spec, SemanticRetrieverSpec)
        return {"model": typed_spec.model}

    def deserialise_spec(
        self,
        *,
        weight: float,
        config: Mapping[str, object],
    ) -> RetrieverSpec:
        return SemanticRetrieverSpec(
            weight=weight,
            model=str(config["model"]),
        )

    def default_path(self, *, index: int, spec: RetrieverSpec) -> str | None:
        return f"{_RETRIEVERS_DIR_NAME}/{index:02d}-{spec.name}"

    def build_artifact(
        self,
        *,
        spec: RetrieverSpec,
        corpus: CleanCorpus,
        path: Path | None,
    ) -> None:
        typed_spec = _require_spec_type(spec, SemanticRetrieverSpec)
        build_semantic_index(
            corpus,
            model=typed_spec.model,
            output_dir=_require_path(path, typed_spec.name),
            overwrite=True,
        )

    def load_retriever(
        self,
        *,
        spec: RetrieverSpec,
        corpus: CleanCorpus,
        config: SaytConfig,
        path: Path | None,
    ) -> Retriever:
        typed_spec = _require_spec_type(spec, SemanticRetrieverSpec)
        index = load_semantic_index(
            corpus,
            model=typed_spec.model,
            folder_path=_require_path(path, typed_spec.name),
        )
        return SemanticRetriever.from_index(
            corpus,
            min_chars=config.min_chars,
            index=index,
        )


def _require_path(path: Path | None, retriever_name: str) -> Path:
    if path is None:
        raise ValueError(
            f"Retriever '{retriever_name}' requires a persisted filespace path"
        )
    return path


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ValueError(f"Malformed integer value for retriever field: {field_name}")
    return int(value)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"Malformed float value for retriever field: {field_name}")
    return float(value)


def _require_spec_type(spec: RetrieverSpec, spec_type: type[SpecT]) -> SpecT:
    if not isinstance(spec, spec_type):
        raise TypeError(
            f"Expected spec of type {spec_type.__name__}, got {type(spec).__name__}"
        )
    return spec


def _register_builtin_retriever_artifact_handlers() -> None:
    """Seed the artifact handler registry with the built-in retriever types."""
    for handler in (
        _PrefixRetrieverArtifactHandler(),
        _NgramRetrieverArtifactHandler(),
        _SemanticRetrieverArtifactHandler(),
    ):
        register_retriever_artifact_handler(handler)


_register_builtin_retriever_artifact_handlers()
