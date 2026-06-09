"""Tests for SAYT storage helper validation and artifact edge cases."""

# pylint: disable=protected-access,too-few-public-methods,missing-function-docstring

import json

import pytest

from industrial_classification_utils.sayt import (
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
    sayt_storage,
)
from industrial_classification_utils.sayt.sayt_core import CleanCorpus


class _DuplicateHandler:
    artifact_type = "test-duplicate"

    def can_handle(self, spec):
        _ = spec
        return False

    def serialise_spec(self, spec):
        _ = spec
        return {}

    def deserialise_spec(self, *, weight, config):
        _ = (weight, config)
        return PrefixRetrieverSpec()

    def default_path(self, *, index, spec):
        _ = (index, spec)

    def build_artifact(self, *, spec, corpus, path):
        _ = (spec, corpus, path)

    def load_retriever(self, *, spec, corpus, min_chars, path):
        _ = path
        return spec.build(corpus, min_chars=min_chars)


def test_prepare_artifact_dir_handles_existing_paths(tmp_path):
    """Reject accidental reuse, then replace existing directories or files."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    stale_file = artifact_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Artifact directory already exists"):
        sayt_storage.prepare_artifact_dir(artifact_dir)

    result = sayt_storage.prepare_artifact_dir(artifact_dir, overwrite=True)

    assert result == artifact_dir
    assert artifact_dir.is_dir()
    assert not stale_file.exists()

    artifact_file = tmp_path / "artifact-file"
    artifact_file.write_text("stale", encoding="utf-8")

    replaced = sayt_storage.prepare_artifact_dir(artifact_file, overwrite=True)

    assert replaced == artifact_file
    assert artifact_file.is_dir()


def test_read_artifact_inputs_validate_missing_and_malformed_state(tmp_path):
    """Raise clear errors for missing files and malformed manifest payloads."""
    with pytest.raises(FileNotFoundError, match="Artifact corpus file not found"):
        sayt_storage.read_artifact_corpus(artifact_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="Artifact manifest not found"):
        sayt_storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_type": "other", "artifact_version": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported artifact type"):
        sayt_storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path.write_text(
        json.dumps({"artifact_type": "sayt", "artifact_version": 999}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported artifact version"):
        sayt_storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "sayt",
                "artifact_version": 2,
                "min_chars": 3,
                "corpus_file": "corpus.csv",
                "corpus_size": 1,
                "retrievers": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="Malformed artifact manifest: missing max_suggestions"
    ):
        sayt_storage.read_artifact_manifest(artifact_dir=tmp_path)


def test_storage_helper_validation_errors():
    """Guard helper APIs against invalid types, paths, and missing handlers."""
    stored_retriever = sayt_storage.StoredRetrieverSpec(
        artifact_type="prefix",
        spec=PrefixRetrieverSpec(),
        config={},
        path=None,
    )

    with pytest.raises(ValueError, match="does not have a stored filespace"):
        sayt_storage.retriever_filespace_path("artifact", stored_retriever)

    with pytest.raises(ValueError, match="Malformed retriever config for type: prefix"):
        sayt_storage._deserialise_stored_retriever(
            {"type": "prefix", "weight": 1.0, "config": []}
        )

    with pytest.raises(
        ValueError, match="No retriever artifact handler registered for type: missing"
    ):
        sayt_storage._get_retriever_artifact_handler("missing")

    class _UnknownSpec:
        name = "unknown"
        weight = 1.0

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)

    with pytest.raises(
        TypeError,
        match="No retriever artifact handler registered for spec type: _UnknownSpec",
    ):
        sayt_storage._get_retriever_artifact_handler_for_spec(_UnknownSpec())

    with pytest.raises(
        ValueError, match="Retriever 'semantic' requires a persisted filespace path"
    ):
        sayt_storage._require_path(None, "semantic")

    with pytest.raises(
        ValueError, match="Malformed integer value for retriever field: n"
    ):
        sayt_storage._coerce_int(True, field_name="n")

    with pytest.raises(
        ValueError, match="Malformed float value for retriever field: weight"
    ):
        sayt_storage._coerce_float(True, field_name="weight")

    with pytest.raises(
        TypeError,
        match="Expected spec of type SemanticRetrieverSpec, got PrefixRetrieverSpec",
    ):
        sayt_storage._require_spec_type(PrefixRetrieverSpec(), SemanticRetrieverSpec)


def test_register_retriever_artifact_handler_rejects_duplicate_registration():
    """Require replace=True before reusing an artifact type registration."""
    handler = _DuplicateHandler()
    sayt_storage.register_retriever_artifact_handler(handler)
    try:
        with pytest.raises(
            ValueError,
            match="Retriever artifact handler already registered for type: test-duplicate",
        ):
            sayt_storage.register_retriever_artifact_handler(handler)
    finally:
        sayt_storage.unregister_retriever_artifact_handler(handler.artifact_type)


def test_semantic_artifact_handler_round_trips_and_loads(
    monkeypatch, tmp_path, small_corpus
):
    """Round-trip semantic spec state and delegate dense index load/build calls."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)
    handler = sayt_storage._SemanticRetrieverArtifactHandler()
    spec = SemanticRetrieverSpec(model="all-MiniLM-L6-v2", weight=2.5)
    path = tmp_path / "retrievers" / "02-semantic"

    def _fake_build_semantic_index(corpus_arg, *, model, output_dir, overwrite):
        captured["build"] = {
            "corpus": corpus_arg,
            "model": model,
            "output_dir": output_dir,
            "overwrite": overwrite,
        }

    def _fake_load_semantic_index(corpus_arg, *, model, folder_path):
        captured["load"] = {
            "corpus": corpus_arg,
            "model": model,
            "folder_path": folder_path,
        }
        return "loaded-index"

    class _StubSemanticRetriever:
        @classmethod
        def from_index(cls, corpus_arg, *, min_chars, index):
            captured["from_index"] = {
                "corpus": corpus_arg,
                "min_chars": min_chars,
                "index": index,
            }
            return {"index": index, "min_chars": min_chars}

    monkeypatch.setattr(
        sayt_storage, "build_semantic_index", _fake_build_semantic_index
    )
    monkeypatch.setattr(sayt_storage, "load_semantic_index", _fake_load_semantic_index)
    monkeypatch.setattr(sayt_storage, "SemanticRetriever", _StubSemanticRetriever)

    rebuilt = handler.deserialise_spec(weight=2.5, config={"model": "all-MiniLM-L6-v2"})

    assert handler.serialise_spec(spec) == {"model": "all-MiniLM-L6-v2"}
    assert isinstance(rebuilt, SemanticRetrieverSpec)
    assert rebuilt.weight == pytest.approx(2.5)
    assert rebuilt.model == "all-MiniLM-L6-v2"
    assert handler.default_path(index=2, spec=spec) == "retrievers/02-semantic"

    handler.build_artifact(spec=spec, corpus=corpus, path=path)
    retriever = handler.load_retriever(
        spec=spec,
        corpus=corpus,
        min_chars=3,
        path=path,
    )

    assert retriever == {"index": "loaded-index", "min_chars": 3}
    assert captured == {
        "build": {
            "corpus": corpus,
            "model": "all-MiniLM-L6-v2",
            "output_dir": path,
            "overwrite": True,
        },
        "load": {
            "corpus": corpus,
            "model": "all-MiniLM-L6-v2",
            "folder_path": path,
        },
        "from_index": {
            "corpus": corpus,
            "min_chars": 3,
            "index": "loaded-index",
        },
    }
