"""Record-level tabular export conformance tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402
from torchlens._constants import (  # noqa: E402
    BUFFER_LOG_FIELD_ORDER,
    GRAD_FN_LOG_FIELD_ORDER,
    GRAD_FN_PASS_LOG_FIELD_ORDER,
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)
from torchlens._data_classes.buffer import _TO_PANDAS_EXCLUDED_BUFFER_FIELDS  # noqa: E402


class _RecordExportModel(nn.Module):
    """Small model that exercises all record-level export surfaces."""

    def __init__(self) -> None:
        """Initialize modules, params, and a used buffer."""

        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, 1)
        self.register_buffer("offset", torch.tensor([0.25, -0.5, 0.75]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a buffer use, two module calls, and a scalar output."""

        hidden = torch.relu(self.encoder(x + self.offset))
        return self.head(hidden).sum()


@pytest.fixture
def record_export_trace() -> tl.Trace:
    """Return a trace with forward and backward records populated."""

    torch.manual_seed(0)
    model = _RecordExportModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
    trace.log_backward(trace[trace.output_layers[0]].out, retain_graph=True)
    return trace


def _assert_file_exporters_removed(record: Any) -> None:
    """Assert deprecated record-level file exporters are absent."""

    for method_name in ("to_csv", "to_parquet", "to_json"):
        assert not hasattr(record, method_name)


def _assert_round_trip(record: Any, columns: list[str], tmp_path: Path) -> None:
    """Assert CSV and JSON round-trips preserve stable record values."""

    expected = record.to_pandas()
    assert len(expected) >= 1
    assert set(columns).issubset(expected.columns)
    _assert_file_exporters_removed(record)

    csv_path = tmp_path / f"{type(record).__name__}.csv"
    tl.export.csv(record, csv_path)
    # tl.export.csv prepends the capture-honesty disclosure as a comment line.
    csv_df = pd.read_csv(csv_path, comment="#")

    json_path = tmp_path / f"{type(record).__name__}.json"
    tl.export.json(record, json_path, orient="records")
    # tl.export.json writes the torchlens.table_export.v1 honesty ENVELOPE
    # ({schema, capture_honesty, orient, rows}), not a bare records array --
    # consumers read the rows key.
    envelope = json.loads(json_path.read_text())
    assert envelope["schema"] == "torchlens.table_export.v1"
    assert envelope["orient"] == "records"
    json_df = pd.DataFrame(envelope["rows"])

    for actual in (csv_df, json_df):
        assert list(actual.columns) == list(expected.columns)
        expected_subset = expected.loc[:, columns].astype(object).astype(str).reset_index(drop=True)
        actual_subset = actual.loc[:, columns].astype(object).astype(str).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_subset, actual_subset, check_dtype=False)


def test_forward_record_exports_round_trip(record_export_trace: tl.Trace, tmp_path: Path) -> None:
    """Forward records expose the full tabular export quartet."""

    op = record_export_trace.ops[0]
    layer = record_export_trace.layers[0]
    module = record_export_trace.modules["encoder"]
    module_call = record_export_trace.module_calls["encoder:1"]
    param = record_export_trace.params["encoder.weight"]
    buffer = record_export_trace.buffers["offset"]

    assert list(op.to_pandas().columns) == LAYER_PASS_LOG_FIELD_ORDER
    assert list(layer.to_pandas().columns) == LAYER_LOG_FIELD_ORDER
    assert list(param.to_pandas().columns) == PARAM_LOG_FIELD_ORDER
    assert list(buffer.to_pandas().columns) == [
        field for field in BUFFER_LOG_FIELD_ORDER if field not in _TO_PANDAS_EXCLUDED_BUFFER_FIELDS
    ]
    assert "initial_value" in buffer.to_pandas().columns
    assert list(module_call.to_pandas().columns) == MODULE_PASS_LOG_FIELD_ORDER

    _assert_round_trip(op, ["label", "layer_label", "type"], tmp_path)
    _assert_round_trip(layer, ["layer_label", "layer_type", "num_passes"], tmp_path)
    _assert_round_trip(module, ["layer_label", "layer_type", "num_ops"], tmp_path)
    _assert_round_trip(module_call, ["address", "call_index", "label"], tmp_path)
    _assert_round_trip(param, ["address", "name", "num_params"], tmp_path)
    _assert_round_trip(buffer, ["address", "name", "call_index"], tmp_path)


def test_backward_record_exports_round_trip(record_export_trace: tl.Trace, tmp_path: Path) -> None:
    """Backward records expose the full tabular export quartet."""

    grad_fn = next(
        record for record in record_export_trace.autograd.grad_fns if record.num_calls == 1
    )
    grad_fn_call = grad_fn.calls[0]

    assert list(grad_fn.to_pandas().columns) == GRAD_FN_LOG_FIELD_ORDER
    assert list(grad_fn_call.to_pandas().columns) == GRAD_FN_PASS_LOG_FIELD_ORDER

    _assert_round_trip(grad_fn, ["label", "class_name", "num_calls"], tmp_path)
    _assert_round_trip(grad_fn_call, ["call_index", "label"], tmp_path)


@pytest.fixture
def record_export_recording() -> tuple[tl.Recording, nn.Module, torch.Tensor]:
    """Return a sparse recording of the export model with its model and input."""

    torch.manual_seed(0)
    model = _RecordExportModel()
    x = torch.randn(2, 3)
    return tl.record(model, x, save=tl.func("relu")), model, x


def test_recording_to_agent_json_matches_cooked_trace(
    record_export_recording: tuple[tl.Recording, nn.Module, torch.Tensor],
) -> None:
    """Recording.to_agent_json cooks through to_trace and delegates (RC-271)."""

    recording, _model, _x = record_export_recording
    card = recording.to_agent_json(max_ops=4)
    cooked = recording.to_trace().to_agent_json(max_ops=4)
    assert isinstance(card, dict)
    assert card["schema"] == cooked["schema"]
    assert set(card) == set(cooked)
    json.dumps(card)


def test_recording_xarray_export_cooks_first(
    record_export_recording: tuple[tl.Recording, nn.Module, torch.Tensor],
) -> None:
    """tl.export.xarray accepts a Recording by cooking it (RC-271)."""

    xr = pytest.importorskip("xarray")
    recording, _model, _x = record_export_recording
    dataset = tl.export.xarray(recording)
    assert isinstance(dataset, xr.Dataset)
    expected = tl.export.xarray(recording.to_trace())
    assert list(dataset.data_vars) == list(expected.data_vars)


def test_recording_save_round_trips_through_load(
    record_export_recording: tuple[tl.Recording, nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    """Recording.save writes the cooked Trace artifact that tl.load reads back (RC-271)."""

    recording, _model, _x = record_export_recording
    target = tmp_path / "recording.tlspec"
    recording.save(target)
    assert target.exists()
    with pytest.raises(Exception, match="exists|overwrite"):
        recording.save(target)
    recording.save(target, overwrite=True)
    loaded = tl.load(target)
    assert isinstance(loaded, tl.Trace)
    assert loaded.num_ops == recording.to_trace().num_ops


def test_recording_io_save_cooks_first(
    record_export_recording: tuple[tl.Recording, nn.Module, torch.Tensor], tmp_path: Path
) -> None:
    """torchlens._io.bundle.save (tl.save) accepts a Recording by cooking it (RC-271)."""

    from torchlens._io.bundle import save as io_save

    recording, _model, _x = record_export_recording
    target = tmp_path / "io.tlspec"
    io_save(recording, str(target))
    assert target.exists()
    assert tl.load(target).num_ops == recording.to_trace().num_ops
