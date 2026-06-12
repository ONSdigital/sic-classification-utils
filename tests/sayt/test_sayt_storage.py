"""Tests for SAYT storage helper validation and artifact edge cases."""

# pylint: disable=protected-access,too-few-public-methods,missing-function-docstring

import json

import pytest

from industrial_classification_utils.sayt import (
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
    storage,
)
from industrial_classification_utils.sayt.core import CleanCorpus


def test_prepare_artifact_dir_handles_existing_paths(tmp_path):
    """Reject accidental reuse, then replace existing directories or files."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    stale_file = artifact_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Artifact directory already exists"):
        storage.prepare_artifact_dir(artifact_dir)

    result = storage.prepare_artifact_dir(artifact_dir, overwrite=True)

    assert result == artifact_dir
    assert artifact_dir.is_dir()
    assert not stale_file.exists()

    artifact_file = tmp_path / "artifact-file"
    artifact_file.write_text("stale", encoding="utf-8")

    replaced = storage.prepare_artifact_dir(artifact_file, overwrite=True)

    assert replaced == artifact_file
    assert artifact_file.is_dir()


def test_read_artifact_inputs_validate_missing_and_malformed_state(tmp_path):
    """Raise clear errors for missing files and malformed manifest payloads."""
    with pytest.raises(FileNotFoundError, match="Artifact corpus file not found"):
        storage.read_artifact_corpus(artifact_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="Artifact manifest not found"):
        storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_type": "other", "artifact_version": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported artifact type"):
        storage.read_artifact_manifest(artifact_dir=tmp_path)

    manifest_path.write_text(
        json.dumps({"artifact_type": "sayt", "artifact_version": 999}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported artifact version"):
        storage.read_artifact_manifest(artifact_dir=tmp_path)

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
        storage.read_artifact_manifest(artifact_dir=tmp_path)


def test_storage_helper_validation_errors():
    """Guard helper APIs against invalid types, paths, and unsupported specs."""
    stored_retriever = storage.StoredRetrieverSpec(
        artifact_type="prefix",
        spec=PrefixRetrieverSpec(),
        config={},
        path=None,
    )

    with pytest.raises(ValueError, match="does not have a stored filespace"):
        storage.retriever_filespace_path("artifact", stored_retriever)

    with pytest.raises(ValueError, match="Malformed retriever config for type: prefix"):
        storage._deserialise_stored_retriever(
            {"type": "prefix", "weight": 1.0, "config": []}
        )

    with pytest.raises(ValueError, match="Unsupported stored retriever type: missing"):
        storage._deserialise_stored_retriever(
            {"type": "missing", "weight": 1.0, "config": {}}
        )

    class _UnknownSpec:
        name = "unknown"
        weight = 1.0

        def build(self, corpus, *, min_chars):
            _ = (corpus, min_chars)

    with pytest.raises(
        TypeError,
        match="Only built-in retriever specs can be persisted; got _UnknownSpec",
    ):
        storage._build_stored_retriever(0, _UnknownSpec())

    with pytest.raises(
        ValueError, match="Malformed integer value for retriever field: n"
    ):
        storage._coerce_int(True, field_name="n")

    with pytest.raises(
        ValueError, match="Malformed float value for retriever field: weight"
    ):
        storage._coerce_float(True, field_name="weight")


def test_semantic_retriever_artifact_round_trips_and_loads(
    monkeypatch, tmp_path, small_corpus
):
    """Round-trip semantic artifact state and delegate dense index load/build calls."""
    captured = {}
    corpus = CleanCorpus.model_validate(small_corpus)
    spec = SemanticRetrieverSpec(model="all-MiniLM-L6-v2", weight=2.5)
    stored_retriever = storage._build_stored_retriever(2, spec)
    path = tmp_path / stored_retriever.path

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

    monkeypatch.setattr(storage, "build_semantic_index", _fake_build_semantic_index)
    monkeypatch.setattr(storage, "load_semantic_index", _fake_load_semantic_index)
    monkeypatch.setattr(storage, "SemanticRetriever", _StubSemanticRetriever)

    rebuilt = storage._deserialise_stored_retriever(
        {
            "type": stored_retriever.artifact_type,
            "weight": spec.weight,
            "path": stored_retriever.path,
            "config": stored_retriever.config,
        }
    )

    assert stored_retriever.artifact_type == "semantic"
    assert stored_retriever.config == {"model": "all-MiniLM-L6-v2"}
    assert stored_retriever.path == "retrievers/02-semantic"
    assert isinstance(rebuilt.spec, SemanticRetrieverSpec)
    assert rebuilt.spec.weight == pytest.approx(2.5)
    assert rebuilt.config == {"model": "all-MiniLM-L6-v2"}

    storage.build_retriever_artifact(
        corpus=corpus,
        stored_retriever=stored_retriever,
        artifact_dir=tmp_path,
    )
    retriever = storage.load_retriever_from_artifact(
        corpus=corpus,
        min_chars=3,
        stored_retriever=stored_retriever,
        artifact_dir=tmp_path,
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
