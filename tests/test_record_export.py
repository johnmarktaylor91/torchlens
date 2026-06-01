"""Record-level tabular export conformance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.constants import (
    BUFFER_LOG_FIELD_ORDER,
    GRAD_FN_LOG_FIELD_ORDER,
    GRAD_FN_PASS_LOG_FIELD_ORDER,
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)


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
    trace = tl.trace(model, x, gradients_to_save="all")
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
    csv_df = pd.read_csv(csv_path)

    json_path = tmp_path / f"{type(record).__name__}.json"
    tl.export.json(record, json_path, orient="records")
    json_df = pd.read_json(json_path, orient="records")

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
    assert list(buffer.to_pandas().columns) == BUFFER_LOG_FIELD_ORDER
    assert list(module_call.to_pandas().columns) == MODULE_PASS_LOG_FIELD_ORDER

    _assert_round_trip(op, ["label", "layer_label", "type"], tmp_path)
    _assert_round_trip(layer, ["layer_label", "layer_type", "num_passes"], tmp_path)
    _assert_round_trip(module, ["layer_label", "layer_type", "num_ops"], tmp_path)
    _assert_round_trip(module_call, ["address", "call_index", "call_label"], tmp_path)
    _assert_round_trip(param, ["address", "name", "num_params"], tmp_path)
    _assert_round_trip(buffer, ["address", "name", "call_index"], tmp_path)


def test_backward_record_exports_round_trip(record_export_trace: tl.Trace, tmp_path: Path) -> None:
    """Backward records expose the full tabular export quartet."""

    grad_fn = next(record for record in record_export_trace.grad_fns if record.num_calls == 1)
    grad_fn_call = grad_fn.calls[0]

    assert list(grad_fn.to_pandas().columns) == GRAD_FN_LOG_FIELD_ORDER
    assert list(grad_fn_call.to_pandas().columns) == GRAD_FN_PASS_LOG_FIELD_ORDER

    _assert_round_trip(grad_fn, ["label", "class_name", "num_calls"], tmp_path)
    _assert_round_trip(grad_fn_call, ["call_index", "call_label"], tmp_path)
