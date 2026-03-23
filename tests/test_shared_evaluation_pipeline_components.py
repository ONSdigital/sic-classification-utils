"""Unit tests for shared evaluation pipeline helper utilities.

These tests focus on the helper functions in
`industrial_classification_utils.utils.shared_evaluation_pipeline_components`.
They avoid external I/O (e.g. real GCS access) by using `tmp_path` and mocks.
"""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

import io
import json
import re
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from industrial_classification_utils.utils import (
    shared_evaluation_pipeline_components as shared,
)


@pytest.mark.utils
def test_write_and_read_json_local_roundtrip(tmp_path: Path):
    obj = {"a": 1, "b": {"c": "d"}}
    out_path = tmp_path / "nested" / "metadata.json"

    shared._write_json(obj, str(out_path))
    assert out_path.exists()

    loaded = shared._read_json(str(out_path))
    assert loaded == obj


@pytest.mark.utils
def test_read_json_gcs_path_uses_gcsfs():
    payload = {"hello": "world", "n": 2}
    buf = io.StringIO(json.dumps(payload))

    fs = MagicMock()
    fs.open.return_value = buf

    with patch.object(shared.gcsfs, "GCSFileSystem", return_value=fs) as gcsfs_cls:
        loaded = shared._read_json("gs://some-bucket/some.json")

    gcsfs_cls.assert_called_once()
    fs.open.assert_called_once()
    assert loaded == payload


@pytest.mark.utils
def test_write_json_gcs_path_uses_gcsfs():
    payload = {"x": 1, "y": ["a", "b"]}

    class _NonClosingStringIO(io.StringIO):
        def close(self):
            """Keep buffer readable after context manager exit."""

    buf = _NonClosingStringIO()
    fs = MagicMock()
    fs.open.return_value = buf

    with patch.object(shared.gcsfs, "GCSFileSystem", return_value=fs) as gcsfs_cls:
        shared._write_json(payload, "gs://some-bucket/out.json")

    gcsfs_cls.assert_called_once()
    fs.open.assert_called_once()
    buf.seek(0)
    assert json.loads(buf.read()) == payload


@pytest.mark.utils
def test_delete_folder_contents_local_removes_folder(tmp_path: Path):
    folder = tmp_path / "to_delete"
    folder.mkdir()
    (folder / "file.txt").write_text("x", encoding="utf8")

    shared._delete_folder_contents(str(folder))
    assert not folder.exists()


@pytest.mark.utils
def test_delete_folder_contents_gcs_path_calls_rm():
    fs = MagicMock()
    with patch.object(shared.gcsfs, "GCSFileSystem", return_value=fs) as gcsfs_cls:
        shared._delete_folder_contents("gs://bucket/path")

    gcsfs_cls.assert_called_once()
    fs.rm.assert_called_once_with("gs://bucket/path", recursive=True)


@pytest.mark.utils
def test_persist_results_intermediate_writes_json_and_parquet(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    metadata = {"batch_size": 25, "batch_size_async": 10}
    out_dir = tmp_path / "out"

    with patch.object(pd.DataFrame, "to_parquet", autospec=True) as to_parquet:
        shared.persist_results(
            df=df,
            metadata=metadata,
            output_folder=str(out_dir),
            output_shortname="STG9",
            is_final=False,
            completed_batches=3,
        )

    ckpt_path = out_dir / "intermediate_outputs" / "STG9_checkpoint_info.json"
    meta_path = out_dir / "intermediate_outputs" / "STG9_metadata.json"
    assert ckpt_path.exists()
    assert meta_path.exists()

    ckpt = json.loads(ckpt_path.read_text(encoding="utf8"))
    assert ckpt["completed_batches"] == 3
    assert ckpt["batch_size"] == 25
    assert ckpt["batch_size_async"] == 10

    persisted_meta = json.loads(meta_path.read_text(encoding="utf8"))
    assert persisted_meta["batch_size"] == 25

    to_parquet.assert_called_once()
    assert str(out_dir / "intermediate_outputs" / "STG9.parquet") in str(
        to_parquet.call_args[0][1]
    )


@pytest.mark.utils
def test_persist_results_final_writes_outputs_and_deletes_intermediate(tmp_path: Path):
    df = pd.DataFrame({"a": [1]})
    metadata = {"batch_size": 100, "batch_size_async": 10}
    out_dir = tmp_path / "out"
    intermediate = out_dir / "intermediate_outputs"
    intermediate.mkdir(parents=True)
    (intermediate / "dummy.txt").write_text("x", encoding="utf8")

    with patch.object(
        pd.DataFrame, "to_parquet", autospec=True
    ) as to_parquet, patch.object(pd.DataFrame, "to_csv", autospec=True) as to_csv:
        shared.persist_results(
            df=df,
            metadata=metadata,
            output_folder=str(out_dir),
            output_shortname="STG9",
            is_final=True,
        )

    assert (out_dir / "STG9_metadata.json").exists()
    assert not intermediate.exists()
    to_parquet.assert_called_once()
    to_csv.assert_called_once()


@pytest.mark.utils
def test_try_to_restart_loads_persisted_assets(tmp_path: Path):
    out_dir = tmp_path / "out"
    inter = out_dir / "intermediate_outputs"
    inter.mkdir(parents=True)

    (inter / "STG1_checkpoint_info.json").write_text(
        json.dumps({"completed_batches": 7}),
        encoding="utf8",
    )
    (inter / "STG1_metadata.json").write_text(
        json.dumps({"k": "v"}),
        encoding="utf8",
    )

    df = pd.DataFrame({"x": [1, 2]})
    with patch("pandas.read_parquet", return_value=df) as read_parquet:
        loaded_df, loaded_meta, completed_batches = shared._try_to_restart(
            Namespace(output_folder=str(out_dir), output_shortname="STG1")
        )

    read_parquet.assert_called_once()
    assert loaded_df.equals(df)
    assert loaded_meta == {"k": "v"}
    assert completed_batches == 7


@pytest.mark.utils
def test_set_up_initial_state_restart_success_short_circuits(tmp_path: Path):
    df = pd.DataFrame({"a": [1]})
    meta = {"batch_size": 100}

    args = Namespace(
        restart=True,
        output_folder=str(tmp_path / "out"),
        output_shortname="STG1",
        input_file=str(tmp_path / "in.csv"),
        metadata_json=str(tmp_path / "meta.json"),
        batch_size=None,
        second_run=False,
    )

    with patch.object(
        shared, "_try_to_restart", return_value=(df, meta, 4)
    ) as tr, patch("pandas.read_csv") as read_csv:
        out_df, out_meta, start = shared.set_up_initial_state(args)

    tr.assert_called_once()
    read_csv.assert_not_called()
    assert out_df.equals(df)
    assert out_meta == meta
    assert start == 4


@pytest.mark.utils
def test_set_up_initial_state_restart_missing_checkpoint_falls_back(tmp_path: Path):
    in_csv = tmp_path / "in.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(in_csv, index=False)

    args = Namespace(
        restart=True,
        output_folder=str(tmp_path / "out"),
        output_shortname="STG1",
        input_file=str(in_csv),
        metadata_json=str(tmp_path / "missing.json"),
        batch_size=5,
        second_run=False,
    )

    with patch.object(shared, "_try_to_restart", side_effect=FileNotFoundError):
        df, metadata, start = shared.set_up_initial_state(args)

    assert start == 0
    assert df.shape == (2, 1)
    assert metadata["batch_size"] == 5
    assert metadata["original_dataset_name"] == str(in_csv)
    assert f"{args.output_shortname}_input_file" in metadata
    assert isinstance(metadata[f"{args.output_shortname}_start_timestamp"], float)
    assert re.fullmatch(
        r"\d{4}/\d{2}/\d{2}_\d{2}:\d{2}:\d{2}",
        metadata[f"{args.output_shortname}_start_time_readable"],
    )


@pytest.mark.utils
def test_update_metadata_preserves_existing_batch_size_async(tmp_path: Path):
    args = Namespace(
        output_shortname="STG2",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    in_metadata = {"batch_size": 7, "batch_size_async": 3}

    out = shared._update_metadata_with_args_and_defaults(args, in_metadata)
    assert out["batch_size"] == 7
    assert out["batch_size_async"] == 3


@pytest.mark.utils
def test_update_metadata_sets_original_dataset_name_only_for_stg1(tmp_path: Path):
    args_stg1 = Namespace(
        output_shortname="STG1",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    out = shared._update_metadata_with_args_and_defaults(args_stg1, {})
    assert out["original_dataset_name"] == str(tmp_path / "in.csv")

    args_second_run = Namespace(
        output_shortname="STG1",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=True,
    )
    out2 = shared._update_metadata_with_args_and_defaults(args_second_run, {})
    assert "original_dataset_name" not in out2

    args_not_stg1 = Namespace(
        output_shortname="STG3",
        input_file=str(tmp_path / "in.csv"),
        batch_size=None,
        second_run=False,
    )
    out3 = shared._update_metadata_with_args_and_defaults(args_not_stg1, {})
    assert "original_dataset_name" not in out3


@pytest.mark.utils
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--input_file",
            "in.csv",
            "--output_folder",
            "out",
        ],
    )

    args = shared.parse_args(default_output_shortname="STGX")
    assert args.input_file == "in.csv"
    assert args.output_folder == "out"
    assert args.output_shortname == "STGX"
    assert args.batch_size is None
    assert args.restart is False
    assert args.second_run is False
