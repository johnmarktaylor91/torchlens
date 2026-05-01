"""Tests for Phase 10 export surfaces."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import pytest
import torch
from torch import nn

import torchlens as tl


class _Tracker:
    """Small object with tracker-like methods for export tests."""

    def __init__(self) -> None:
        """Initialize recorded calls."""

        self.metrics: list[tuple[str, int]] = []

    def log_metric(self, name: str, value: int) -> None:
        """Record an MLflow-like metric call.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Metric value.
        """

        self.metrics.append((name, value))

    def track(self, value: int, name: str) -> None:
        """Record an Aim-like track call.

        Parameters
        ----------
        value:
            Metric value.
        name:
            Metric name.
        """

        self.metrics.append((name, value))


class _FakeHubApi:
    """Small Hugging Face API double."""

    def __init__(self) -> None:
        """Initialize recorded calls."""

        self.created: list[dict[str, Any]] = []
        self.uploaded: list[dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        """Record repository creation.

        Parameters
        ----------
        **kwargs:
            Repository creation arguments.
        """

        self.created.append(kwargs)

    def upload_file(self, **kwargs: Any) -> str:
        """Record file upload.

        Parameters
        ----------
        **kwargs:
            Upload arguments.

        Returns
        -------
        str
            Fake upload URL.
        """

        self.uploaded.append(kwargs)
        return "https://huggingface.co/example/repo/blob/main/torchlens_artifact.pkl"


@pytest.fixture
def export_log() -> Any:
    """Build a small ModelLog for export tests.

    Returns
    -------
    Any
        Logged model.
    """

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    return tl.log_forward_pass(model, torch.randn(2, 3), capture=tl.options.CaptureOptions())


def test_trace_timeline_exports_are_parseable(export_log: Any, tmp_path: Path) -> None:
    """Trace/timeline exports should write viewer-conformant payloads."""

    chrome_path = tl.export.chrome_trace(export_log, tmp_path / "trace.json")
    chrome_payload = json.loads(chrome_path.read_text(encoding="utf-8"))
    assert "traceEvents" in chrome_payload
    assert any(event.get("ph") == "X" for event in chrome_payload["traceEvents"])

    speedscope_path = tl.export.speedscope(export_log, tmp_path / "profile.json")
    speedscope_payload = json.loads(speedscope_path.read_text(encoding="utf-8"))
    assert speedscope_payload["$schema"].endswith("file-format-schema.json")
    assert speedscope_payload["profiles"][0]["type"] == "evented"

    flamegraph_path = tl.export.flamegraph(export_log, tmp_path / "profile.folded")
    assert ";" in flamegraph_path.read_text(encoding="utf-8")

    memory_path = tl.export.memory_timeline(export_log, tmp_path / "memory.json")
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    assert memory_payload["scope"] == "tensor"
    assert "not an allocator trace" in memory_payload["disclaimer"]


def test_xarray_export_has_neuroidassembly_shape(export_log: Any) -> None:
    """xarray export should expose presentation and neuroid dimensions."""

    assembly = tl.export.xarray(export_log)

    assert assembly.dims == ("presentation", "neuroid")
    assert "layer" in assembly.coords
    assert assembly.attrs["assembly"] == "NeuroidAssembly"
    assert assembly.sizes["presentation"] == 2
    assert assembly.sizes["neuroid"] > 0


def test_tracker_exports_accept_existing_objects(export_log: Any, tmp_path: Path) -> None:
    """Tracker helpers should work with caller-owned writer/run objects."""

    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tmp_path / "tb")
    returned_writer = tl.export.tensorboard(export_log, writer, step=3, prefix="tl")
    writer.close()

    assert returned_writer is writer
    accumulator = EventAccumulator(str(tmp_path / "tb"))
    accumulator.Reload()
    assert "tl/num_layers" in accumulator.Tags()["scalars"]

    tracker = _Tracker()
    assert tl.export.mlflow(export_log, client=tracker, prefix="tl")["num_layers"] > 0
    assert tracker.metrics

    aim_tracker = _Tracker()
    assert tl.export.aim(export_log, run=aim_tracker, prefix="tl")["num_layers"] > 0
    assert aim_tracker.metrics

    pytest.importorskip("wandb")
    wandb_result = tl.export.wandb(export_log)
    assert "table" in wandb_result


def test_emit_nvtx_capture_option_does_not_change_capture() -> None:
    """emit_nvtx should be accepted and should not alter normal logging output."""

    log = tl.log_forward_pass(
        nn.Linear(2, 2),
        torch.randn(1, 2),
        capture=tl.options.CaptureOptions(emit_nvtx=True),
    )

    assert log.emit_nvtx is True
    assert len(log.layer_list) > 0


def test_tabular_exports_round_trip_and_old_methods_warn(export_log: Any, tmp_path: Path) -> None:
    """New tabular exports should round-trip and old ModelLog writers should warn."""

    expected = export_log.to_pandas()
    assert "func_config" in expected.columns
    assert "cond_branch_then_children" in expected.columns

    csv_path = tl.export.csv(export_log, tmp_path / "model.csv")
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == list(expected.columns)
    assert len(csv_df) == len(expected)

    json_path = tl.export.json(export_log, tmp_path / "model.json")
    json_df = pd.read_json(json_path, orient="records")
    assert list(json_df.columns) == list(expected.columns)
    assert len(json_df) == len(expected)

    parquet_path = tmp_path / "model.parquet"
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
            tl.export.parquet(export_log, parquet_path)
    else:
        tl.export.parquet(export_log, parquet_path)
        parquet_df = pd.read_parquet(parquet_path)
        assert list(parquet_df.columns) == list(expected.columns)
        assert len(parquet_df) == len(expected)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_log.to_csv(tmp_path / "legacy.csv")
        export_log.to_json(tmp_path / "legacy.json")
    assert any("torchlens.export.csv" in str(warning.message) for warning in caught)
    assert any("torchlens.export.json" in str(warning.message) for warning in caught)


def test_static_graph_adapters_and_hub_dry_run(export_log: Any, tmp_path: Path) -> None:
    """Static graph adapters and Hub publisher should write planned payloads."""

    explorer_path = tl.export.model_explorer(export_log, tmp_path / "explorer.json")
    explorer_payload = json.loads(explorer_path.read_text(encoding="utf-8"))
    assert explorer_payload["graphs"][0]["nodes"]

    netron_path = tl.export.netron(export_log, tmp_path / "netron.json")
    netron_payload = json.loads(netron_path.read_text(encoding="utf-8"))
    assert netron_payload["runnable"] is False
    assert "not a real ONNX" in netron_payload["disclaimer"]

    result = tl.bridge.huggingface.push_to_hub(
        export_log,
        "example/repo",
        dry_run=True,
    )
    assert result["repo_id"] == "example/repo"
    assert result["dry_run"] is True

    api = _FakeHubApi()
    uploaded = tl.bridge.huggingface.push_to_hub(export_log, "example/repo", api=api)
    assert uploaded["upload_result"].startswith("https://huggingface.co/")
    assert api.created
    assert api.uploaded


def test_depyf_bridge_fails_soft_when_extra_missing() -> None:
    """depyf bridge should explain the missing optional dependency."""

    if importlib.util.find_spec("depyf") is not None:
        pytest.skip("Installed depyf API varies; smoke coverage is in the extras matrix.")
    with pytest.raises(ImportError, match=r"torchlens\[depyf\]"):
        tl.bridge.depyf.dump(nn.Linear(1, 1), torch.randn(1, 1))
