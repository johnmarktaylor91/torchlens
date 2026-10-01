"""Behavioral tests for the tracker, stub-dependency, and node-typing export paths.

These cover the ``torchlens.export`` surfaces that need no heavyweight optional
dependency at all (duck-typed tracker objects), plus the optional-dependency
contracts exercised through minimal in-test stand-ins so the flattening and
refusal semantics run on every environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402

pytestmark = pytest.mark.smoke


def _small_trace() -> Any:
    """Return a deterministic two-layer trace."""

    torch.manual_seed(11)
    return tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(2, 3))


class _RecordingWriter:
    """TensorBoard-shaped writer that records every summary call."""

    def __init__(self) -> None:
        self.scalars: dict[str, tuple[float, int]] = {}
        self.texts: dict[str, tuple[str, int]] = {}
        self.flushed = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars[name] = (value, step)

    def add_text(self, name: str, value: str, step: int) -> None:
        self.texts[name] = (value, step)

    def flush(self) -> None:
        self.flushed = True


def test_tensorboard_writes_layer_count_memory_and_model_name() -> None:
    """tensorboard() emits the documented scalar/text summaries to any writer."""

    log = _small_trace()
    writer = _RecordingWriter()

    returned = tl.export.tensorboard(log, writer, step=7, prefix="tlx")

    assert returned is writer
    assert writer.scalars["tlx/num_layers"] == (len(log.layer_list), 7)
    memory_value, memory_step = writer.scalars["tlx/total_activation_memory"]
    assert memory_step == 7
    assert memory_value == int(log.total_activation_memory or 0)
    text_value, text_step = writer.texts["tlx/model_class_name"]
    assert text_step == 7
    assert text_value == str(log.model_class_name)
    assert writer.flushed


def test_tracker_exports_reject_objects_without_required_method() -> None:
    """A tracker object missing its required method fails with a clear TypeError."""

    log = _small_trace()

    with pytest.raises(TypeError, match="add_scalar"):
        tl.export.tensorboard(log, object(), step=0)
    with pytest.raises(TypeError, match="log_metric"):
        tl.export.mlflow(log, client=object())
    with pytest.raises(TypeError, match="track"):
        tl.export.aim(log, run=object())


def test_mlflow_logs_prefixed_metrics_and_returns_them() -> None:
    """mlflow() logs each summary metric under the prefix and returns the mapping."""

    log = _small_trace()

    class _Client:
        def __init__(self) -> None:
            self.logged: dict[str, float] = {}

        def log_metric(self, key: str, value: float) -> None:
            self.logged[key] = value

    client = _Client()
    metrics = tl.export.mlflow(log, client=client, prefix="tlm")

    # capture_honesty is RETURNED, never logged (log_metric takes numerics).
    assert set(metrics) == {
        "num_layers",
        "num_saved_ops",
        "total_activation_memory",
        "capture_honesty",
    }
    assert metrics["capture_honesty"]["capture_status"] == "complete"
    assert metrics["num_layers"] == len(log.layer_list)
    numeric_metrics = {key: value for key, value in metrics.items() if key != "capture_honesty"}
    assert client.logged == {f"tlm.{key}": value for key, value in numeric_metrics.items()}
    # Without a client the metrics are still prepared and returned.
    assert tl.export.mlflow(log, client=None) == metrics


def test_aim_tracks_prefixed_metrics_on_run_object() -> None:
    """aim() tracks each summary metric on the provided run."""

    log = _small_trace()

    class _Run:
        def __init__(self) -> None:
            self.tracked: dict[str, float] = {}

        def track(self, value: float, name: str) -> None:
            self.tracked[name] = value

    run = _Run()
    metrics = tl.export.aim(log, run=run, prefix="tla")

    numeric_metrics = {key: value for key, value in metrics.items() if key != "capture_honesty"}
    assert run.tracked == {f"tla.{key}": value for key, value in numeric_metrics.items()}
    assert metrics["capture_honesty"]["capture_status"] == "complete"
    assert tl.export.aim(log, run=None) == metrics


def test_wandb_builds_scalar_safe_table_and_logs_to_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wandb() builds a Table from primitive-only cells and logs it to the run."""

    log = _small_trace()
    built: list[Any] = []

    class _Table:
        def __init__(self, dataframe: Any) -> None:
            self.dataframe = dataframe
            built.append(self)

    class _Run:
        def __init__(self) -> None:
            self.logged: dict[str, Any] = {}

        def log(self, payload: dict[str, Any]) -> None:
            self.logged.update(payload)

    stub = SimpleNamespace(Table=_Table, run=None)
    monkeypatch.setitem(sys.modules, "wandb", stub)

    run = _Run()
    result = tl.export.wandb(log, run=run, name="my_table")

    assert result["table"] is built[0]
    assert result["artifact"] is None
    assert run.logged["my_table"] is built[0]
    dataframe = built[0].dataframe
    assert len(dataframe) == len(log.layer_list)
    primitive = (str, int, float, bool)
    for column in dataframe.columns:
        for cell in dataframe[column]:
            assert cell is None or isinstance(cell, primitive)


def test_wandb_names_extra_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wandb() raises an ImportError naming the install extra when absent."""

    log = _small_trace()
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ImportError, match="torchlens\\[wandb\\]"):
        tl.export.wandb(log)


def _stub_xarray(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Install a minimal recording xarray stand-in and return its capture list."""

    captured: list[Any] = []

    class _DataArray:
        def __init__(
            self,
            data: Any,
            dims: tuple[str, ...],
            coords: dict[str, Any],
            name: str,
            attrs: dict[str, Any],
        ) -> None:
            self.data = data
            self.dims = dims
            self.coords = coords
            self.name = name
            self.attrs = attrs
            captured.append(self)

    monkeypatch.setitem(sys.modules, "xarray", SimpleNamespace(DataArray=_DataArray))
    return captured


def test_xarray_flattens_scalar_vector_and_matrix_outs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xarray() flattens 0-d/1-d outs to one presentation row and batches by row."""

    captured = _stub_xarray(monkeypatch)
    log = SimpleNamespace(
        layer_list=[
            SimpleNamespace(out=torch.tensor(2.5), layer_label="scalar_1_1"),
            SimpleNamespace(out=torch.tensor([1.0, 2.0, 3.0]), layer_label="vector_1_1"),
            SimpleNamespace(out="not a tensor", layer_label="skipped_1_1"),
        ]
    )

    array = tl.export.xarray(log)

    assert array is captured[0]
    assert array.dims == ("presentation", "neuroid")
    assert array.data.shape == (1, 4)
    assert list(array.data[0]) == [2.5, 1.0, 2.0, 3.0]
    assert array.coords["layer"] == ("neuroid", ["scalar_1_1"] * 1 + ["vector_1_1"] * 3)
    assert array.coords["neuroid_index"] == ("neuroid", [0, 0, 1, 2])
    assert array.attrs["assembly"] == "NeuroidAssembly"

    batched = SimpleNamespace(
        layer_list=[
            SimpleNamespace(out=torch.ones(2, 3), layer_label="a_1_1"),
            SimpleNamespace(out=torch.zeros(2, 2, 2), layer_label="b_1_1"),
        ]
    )
    array_2 = tl.export.xarray(batched)
    assert array_2.data.shape == (2, 7)
    assert array_2.coords["presentation"] == [0, 1]


def test_xarray_rejects_mismatched_presentation_counts_and_empty_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xarray() names the offending layer on row-count mismatch and rejects no-out logs."""

    _stub_xarray(monkeypatch)
    mismatched = SimpleNamespace(
        layer_list=[
            SimpleNamespace(out=torch.ones(2, 3), layer_label="first_1_1"),
            SimpleNamespace(out=torch.ones(3, 3), layer_label="second_1_1"),
        ]
    )
    with pytest.raises(ValueError, match="second_1_1"):
        tl.export.xarray(mismatched)

    with pytest.raises(ValueError, match="No saved tensor outs"):
        tl.export.xarray(SimpleNamespace(layer_list=[]))


def test_xarray_missing_dependency_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xarray() raises ImportError when xarray is not importable."""

    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError, match="xarray"):
        tl.export.xarray(_small_trace())


def test_parquet_names_tabular_extra_when_pyarrow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """parquet() raises ImportError naming the tabular extra when pyarrow is absent."""

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match="tabular"):
        tl.export.parquet(_small_trace(), Path("unused.parquet"))


def test_json_export_sanitizes_nonprimitive_and_missing_cells(tmp_path: Path) -> None:
    """json() coerces object cells to primitives: repr for objects, null for NA."""

    frame = pd.DataFrame(
        {
            "label": ["a", "b"],
            "payload": [torch.tensor([1.0]), pd.NaT],
        }
    )
    log = SimpleNamespace(to_pandas=lambda: frame)

    destination = tl.export.json(log, tmp_path / "cells.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    # Duck-typed subjects with no capture facts disclose "unknown", not clean.
    assert payload["capture_honesty"]["capture_status"] == "unknown"
    records = payload["rows"]
    assert records[0]["payload"] == repr(torch.tensor([1.0]))
    assert records[1]["payload"] is None


class _BufferAndBoolModel(nn.Module):
    """Model whose trace contains buffer and terminal-bool nodes."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn(x)
        if y.mean() > 0:
            return torch.relu(y)
        return torch.tanh(y)


def test_svg_marks_buffer_and_terminal_bool_node_classes(tmp_path: Path) -> None:
    """svg(editable=True) assigns the buffer and bool semantic node classes."""

    torch.manual_seed(3)
    model = _BufferAndBoolModel().eval()
    log = tl.trace(model, torch.randn(4, 3))

    destination = tl.export.svg(log, tmp_path / "typed.svg", editable=True)
    text = destination.read_text(encoding="utf-8")

    # Match rendered node class attributes, not the always-present stylesheet.
    assert 'class="tl-node tl-node-buffer"' in text
    assert 'class="tl-node tl-node-bool"' in text


def test_chrome_trace_diff_places_members_on_separate_timelines(tmp_path: Path) -> None:
    """chrome_trace_diff() writes one metadata process per member and per-member events."""

    torch.manual_seed(20)
    x = torch.randn(2, 3)
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    capture = tl.options.CaptureOptions(intervention_ready=True)
    first = tl.trace(model, x, capture=capture)
    second = tl.trace(model, x, capture=capture)
    bundle = tl.bundle({"first": first, "second": second})

    destination = tl.export.chrome_trace_diff(bundle, tmp_path / "diff.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["metadata"]["schema"] == "torchlens.chrome_trace_diff.v1"
    assert payload["metadata"]["members"] == ["first", "second"]
    metadata_events = [e for e in payload["traceEvents"] if e["ph"] == "M"]
    assert {e["args"]["name"] for e in metadata_events} == {"first", "second"}
    span_events = [e for e in payload["traceEvents"] if e["ph"] == "X"]
    assert span_events
    pids = {e["pid"] for e in span_events}
    assert pids == {1, 2}
    for event in span_events:
        assert event["dur"] == 1000
        assert event["ts"] % 1000 == 0
        assert "delta" in event["args"]
