"""Tests for SAYT artifact building and loading."""

# pylint: disable=too-few-public-methods,missing-function-docstring,too-many-arguments,duplicate-code

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
)
from industrial_classification_utils.sayt.core import CleanCorpus
from industrial_classification_utils.sayt.suggester import SAYTSuggester


class _CustomRetrieverSpec:
    def __init__(self, *, trigger: str, weight: float = 1.0):
        self.trigger = trigger
        self.weight = weight
        self.name = "custom-trigger"

    def build(self, corpus, *, min_chars):
        _ = (corpus, min_chars)


@dataclass(frozen=True)
class _FailingArtifactRetrieverSpec:
    name: str = "failing"
    weight: float = 1.0

    def build(
        self,
        corpus,
        *,
        min_chars,
        filespace_path=None,
        overwrite=True,
    ):
        _ = (corpus, min_chars, overwrite)
        if filespace_path is not None:
            Path(filespace_path).mkdir(parents=True, exist_ok=True)
            Path(filespace_path, "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    def load_from_artifact(self, corpus, *, min_chars, filespace_path):
        _ = (corpus, min_chars, filespace_path)
        raise NotImplementedError


def test_builder_from_csv_loads_columns_and_persists_artifact(tmp_path, small_corpus):
    """Build an artifact from CSV input using the configured search/display columns."""
    csv_path = tmp_path / "responses.csv"
    pd.DataFrame(
        {
            "search": [row[0] for row in small_corpus],
            "display": [row[1] for row in small_corpus],
        }
    ).to_csv(csv_path, index=False)

    artifact_dir = SAYTBuilder.from_csv(
        csv_path,
        search_text_col="search",
        display_text_col="display",
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
    ).build_artifact(tmp_path / "artifact")

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["min_chars"] == 3
    assert manifest["max_suggestions"] == 5
    assert manifest["corpus_size"] == len(small_corpus)


def test_builder_writes_manifest_and_corpus(tmp_path, small_corpus):
    """Persist manifest metadata and cleaned corpus rows for an artifact."""
    artifact_dir = tmp_path / "artifact"

    result = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
    ).build_artifact(artifact_dir)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    with open(artifact_dir / "corpus.csv", encoding="utf-8") as corpus_file:
        rows = list(csv.DictReader(corpus_file))

    assert result == artifact_dir
    assert manifest == {
        "artifact_type": "sayt",
        "artifact_version": 2,
        "min_chars": 3,
        "max_suggestions": 5,
        "corpus_file": "corpus.csv",
        "corpus_size": len(small_corpus),
        "retrievers": [
            {"type": "prefix", "weight": 1.0, "path": None, "config": {}},
        ],
    }
    assert rows == [
        {
            "row_id": row_id,
            "search_text": search_text,
            "display_text": display_text,
        }
        for row_id, search_text, display_text in CleanCorpus.model_validate(
            small_corpus
        ).rows
    ]


def test_builder_writes_ngram_filespace(monkeypatch, tmp_path, small_corpus):
    """Persist the configured dense retriever filespace inside the artifact."""
    captured = {}
    artifact_dir = tmp_path / "artifact"

    class _StubPersistentVectorStore:
        def __init__(  # noqa: PLR0913
            self,
            *,
            file_name,
            data_type,
            vectoriser,
            batch_size,
            output_dir,
            overwrite,
            hooks,
        ):
            captured["file_name"] = file_name
            captured["data_type"] = data_type
            captured["vectoriser_type"] = type(vectoriser).__name__
            captured["batch_size"] = batch_size
            captured["output_dir"] = output_dir
            captured["overwrite"] = overwrite
            captured["hooks"] = hooks
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir, "metadata.json").write_text("{}", encoding="utf-8")
            Path(output_dir, "vectors.parquet").write_text("dummy", encoding="utf-8")
            self.num_vectors = 1

    monkeypatch.setattr(
        "industrial_classification_utils.sayt.indexes.VectorStore",
        _StubPersistentVectorStore,
    )

    SAYTBuilder(
        small_corpus,
        retrievers=[NgramRetrieverSpec(max_df=1.0)],
        min_chars=3,
    ).build_artifact(artifact_dir)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    filespace_path = artifact_dir / manifest["retrievers"][0]["path"]

    assert Path(captured["output_dir"]).name == filespace_path.name
    assert (filespace_path / "metadata.json").exists()
    assert (filespace_path / "vectors.parquet").exists()


def test_from_artifact_loads_persisted_ngram_filespace(
    monkeypatch, tmp_path, small_corpus
):
    """Load persisted dense retrievers from their artifact filespaces."""
    captured = {}
    artifact_dir = tmp_path / "artifact"
    target_row_id, _, target_display = CleanCorpus.model_validate(small_corpus).rows[-1]

    class _StubPersistentVectorStore:
        def __init__(  # noqa: PLR0913
            self,
            *,
            file_name,
            data_type,
            vectoriser,
            batch_size,
            output_dir,
            overwrite,
            hooks,
        ):
            _ = (file_name, data_type, vectoriser, batch_size, overwrite, hooks)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir, "metadata.json").write_text("{}", encoding="utf-8")
            Path(output_dir, "vectors.parquet").write_text("dummy", encoding="utf-8")
            self.num_vectors = 1

        @classmethod
        def from_filespace(cls, *, folder_path, vectoriser, hooks):
            captured["folder_path"] = folder_path
            captured["vectoriser_type"] = type(vectoriser).__name__
            captured["hooks"] = hooks
            return _StubLoadedVectorStore()

    class _StubSearchResults:
        def to_dict(self, orient="records"):
            assert orient == "records"
            return [{"doc_label": target_row_id, "score": 1.0}]

    class _StubLoadedVectorStore:
        num_vectors = 1

        def search(self, query, n_results=10):
            _ = query
            captured["n_results"] = n_results
            return _StubSearchResults()

    monkeypatch.setattr(
        "industrial_classification_utils.sayt.indexes.VectorStore",
        _StubPersistentVectorStore,
    )

    builder = SAYTBuilder(
        small_corpus,
        retrievers=[NgramRetrieverSpec(max_df=1.0)],
        min_chars=3,
    )
    builder.build_artifact(artifact_dir)

    suggester = SAYTSuggester.from_artifact(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert suggester.suggest("groom") == [target_display]
    assert captured == {
        "folder_path": str(artifact_dir / manifest["retrievers"][0]["path"]),
        "vectoriser_type": "_CharNgramVectoriser",
        "hooks": None,
        "n_results": 1,
    }


def test_builder_rejects_custom_runtime_only_retriever_specs(tmp_path, small_corpus):
    """Persisted artifacts currently support only the built-in retriever specs."""
    builder = SAYTBuilder(
        small_corpus,
        retrievers=[_CustomRetrieverSpec(trigger="groom", weight=1.5)],
        min_chars=3,
        max_suggestions=4,
    )

    with pytest.raises(
        TypeError,
        match="Only artifact-aware retriever specs can be persisted; got _CustomRetrieverSpec",
    ):
        builder.build_artifact(tmp_path / "artifact")


def test_builder_cleans_up_staged_artifact_when_later_retriever_fails(
    tmp_path, small_corpus
):
    """A failed staged build should not leave a partial artifact behind."""
    artifact_dir = tmp_path / "artifact"

    builder = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec(), _FailingArtifactRetrieverSpec()],
        min_chars=3,
    )

    with pytest.raises(RuntimeError, match="boom"):
        builder.build_artifact(artifact_dir)

    assert not artifact_dir.exists()
    assert not list(tmp_path.glob(".artifact.tmp-*"))


def test_builder_preserves_existing_artifact_when_staged_overwrite_fails(
    tmp_path, small_corpus
):
    """A failed overwrite should leave the previous complete artifact intact."""
    artifact_dir = tmp_path / "artifact"
    original_builder = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec()],
        min_chars=3,
        max_suggestions=5,
    )
    original_builder.build_artifact(artifact_dir)

    original_manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    original_corpus = (artifact_dir / "corpus.csv").read_text(encoding="utf-8")

    failing_builder = SAYTBuilder(
        small_corpus,
        retrievers=[PrefixRetrieverSpec(), _FailingArtifactRetrieverSpec()],
        min_chars=3,
        max_suggestions=7,
    )

    with pytest.raises(RuntimeError, match="boom"):
        failing_builder.build_artifact(artifact_dir, overwrite=True)

    assert json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8")) == (
        original_manifest
    )
    assert (artifact_dir / "corpus.csv").read_text(encoding="utf-8") == original_corpus
    assert not list(tmp_path.glob(".artifact.tmp-*"))
    assert not list(tmp_path.glob(".artifact.bak-*"))
