"""Tests for SAYT artifact building and loading."""

# pylint: disable=too-few-public-methods,missing-function-docstring,too-many-arguments,duplicate-code

import csv
import json
from pathlib import Path

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    RetrieverArtifactHandler,
    SAYTBuilder,
    register_retriever_artifact_handler,
    unregister_retriever_artifact_handler,
)
from industrial_classification_utils.sayt.sayt import SAYTSuggester
from industrial_classification_utils.sayt.sayt_core import CleanCorpus, Suggestion


class _CustomRetriever:
    def __init__(self, row, *, trigger: str, min_chars: int):
        self._row = row
        self._trigger = trigger
        self._min_chars = min_chars

    def suggest_with_scores(self, q_norm, num_suggestions):
        _ = num_suggestions
        if len(q_norm) < self._min_chars or self._trigger not in q_norm:
            return []
        return [
            Suggestion(
                display_text=self._row[2],
                score=1.0,
                search_text=self._row[1],
                row_id=self._row[0],
            )
        ]


class _CustomRetrieverSpec:
    def __init__(self, *, trigger: str, weight: float = 1.0):
        self.trigger = trigger
        self.weight = weight
        self.name = "custom-trigger"

    def build(self, corpus, *, min_chars):
        return _CustomRetriever(
            corpus.rows[-1], trigger=self.trigger, min_chars=min_chars
        )


class _CustomRetrieverArtifactHandlerImpl:
    artifact_type = "custom-trigger"

    def can_handle(self, spec):
        return isinstance(spec, _CustomRetrieverSpec)

    def serialise_spec(self, spec):
        return {"trigger": spec.trigger}

    def deserialise_spec(self, *, weight, config):
        return _CustomRetrieverSpec(trigger=str(config["trigger"]), weight=weight)

    def default_path(self, *, index, spec):
        _ = (index, spec)

    def build_artifact(self, *, spec, corpus, path):
        _ = (spec, corpus, path)

    def load_retriever(self, *, spec, corpus, min_chars, path):
        _ = path
        return spec.build(corpus, min_chars=min_chars)


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
        "industrial_classification_utils.sayt.sayt_indexes.VectorStore",
        _StubPersistentVectorStore,
    )

    SAYTBuilder(
        small_corpus,
        retrievers=[NgramRetrieverSpec(max_df=1.0)],
        min_chars=3,
    ).build_artifact(artifact_dir)

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    filespace_path = artifact_dir / manifest["retrievers"][0]["path"]

    assert captured["output_dir"] == str(filespace_path)
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
        "industrial_classification_utils.sayt.sayt_indexes.VectorStore",
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


def test_custom_retriever_artifact_handler_round_trips(tmp_path, small_corpus):
    """Allow custom retriever specs to participate in artifact build and load."""
    artifact_dir = tmp_path / "artifact"
    handler: RetrieverArtifactHandler = _CustomRetrieverArtifactHandlerImpl()
    register_retriever_artifact_handler(handler)
    try:
        spec = _CustomRetrieverSpec(trigger="groom", weight=1.5)
        builder = SAYTBuilder(
            small_corpus,
            retrievers=[spec],
            min_chars=3,
            max_suggestions=4,
        )

        builder.build_artifact(artifact_dir)

        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        suggester = SAYTSuggester.from_artifact(artifact_dir)

        assert manifest["retrievers"] == [
            {
                "type": "custom-trigger",
                "weight": 1.5,
                "path": None,
                "config": {"trigger": "groom"},
            }
        ]
        assert suggester.suggest("groom") == ["Dog grooming"]
    finally:
        unregister_retriever_artifact_handler("custom-trigger")
