"""Raw-label lookup and expression helper tests."""

from __future__ import annotations

import torch
from pandas.api.types import is_bool_dtype, is_float_dtype

import torchlens as tl
from torchlens.constants import RAW_LABEL_SUFFIX


class RawLabelModel(torch.nn.Module):
    """Small model with a source-visible single-call operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple traced forward pass."""

        y = torch.relu(x + 1)
        return y * 2


def _raw_label_trace() -> tl.Trace:
    """Return a trace with stable raw-label lookup targets."""

    return tl.trace(RawLabelModel(), torch.randn(2, 3))


def test_raw_label_round_trip_via_trace_getitem() -> None:
    """A public raw label resolves to the exact Op through Trace lookup."""

    trace = _raw_label_trace()
    op = next(candidate for candidate in trace.ops if candidate.func_name == "relu")

    assert trace[op.raw_label] is op
    assert trace[op._label_raw] is op
    assert op._label_raw == f"{op.raw_label}{RAW_LABEL_SUFFIX}"


def test_raw_label_round_trip_via_ops_accessor() -> None:
    """Trace.ops accepts public and compatibility raw-label forms."""

    trace = _raw_label_trace()
    op = next(candidate for candidate in trace.ops if candidate.func_name == "relu")

    assert trace.ops[op.raw_label] is op
    assert trace.ops[op._label_raw] is op


def test_ops_accessor_fetches_by_raw_index() -> None:
    """Trace.ops exposes a raw-index lookup helper without changing ordinal indexing."""

    trace = _raw_label_trace()
    op = next(candidate for candidate in trace.ops if candidate.func_name == "relu")

    assert trace.ops.by_raw_index(op.raw_index) is op
    assert trace.ops[op.ordinal_index] is op


def test_four_form_fetch_on_simple_trace() -> None:
    """Ordinal, display label, qualified label, raw label, and raw index find one Op."""

    trace = _raw_label_trace()
    op = next(candidate for candidate in trace.ops if candidate.func_name == "relu")

    assert trace.ops[op.ordinal_index] is op
    assert trace.ops[op.layer_label] is op
    assert trace.ops[op.label] is op
    assert trace.ops[op.raw_label] is op
    assert trace.ops.by_raw_index(op.raw_index) is op


def test_arg_expressions_returns_call_arguments_from_source() -> None:
    """arg_expressions lazily returns per-argument source expressions."""

    trace = _raw_label_trace()
    op = next(candidate for candidate in trace.ops if candidate.func_name == "relu")

    assert op.arg_expressions == ["x + 1"]


def test_to_pandas_converts_new_bool_and_distance_columns() -> None:
    """to_pandas exports bool-like and distance fields with stable dtypes."""

    trace = _raw_label_trace()
    op = next(candidate for candidate in trace.ops if candidate.func_name == "relu")
    frame = op.to_pandas()

    assert is_bool_dtype(frame["is_internally_initialized"])
    assert is_float_dtype(frame["min_distance_from_input"])
    assert is_float_dtype(frame["max_distance_from_input"])
    assert is_float_dtype(frame["min_distance_to_output"])
    assert is_float_dtype(frame["max_distance_to_output"])
