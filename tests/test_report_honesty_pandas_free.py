"""A09 agent stage-0 item 4: the analytics WIRE PATH runs without pandas.

Four of the six ready-made debug analytics returned DataFrames, so any
agent/MCP consumer of ``pip install torchlens[mcp]`` served tools that
ImportError'd unless the pandas extra happened to be present. The fix is a
pandas-free rows core per analytic (``hot_path_rows``, ``dead_neurons_rows``,
``compare_rows``, ``gradient_flow_audit_rows``); the DataFrame becomes the
optional view. These tests prove the property the memo names: the cores are
importable and runnable with pandas UNAVAILABLE, carry the same
capture-honesty facts the DataFrame attrs carry, and the views refuse with
the teaching ``[tabular]`` message instead of a bare ImportError.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import debug as dbg

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def grad_trace() -> Iterator[tl.Trace]:
    """One small capture with saved gradients for every core."""

    torch.manual_seed(0)
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
        x,
        capture=tl.options.CaptureOptions(save_grads=True),
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture()
def _no_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import pandas`` raise ImportError inside the test."""

    monkeypatch.setitem(sys.modules, "pandas", None)


@pytest.mark.usefixtures("_no_pandas")
def test_hot_path_rows_runs_without_pandas(grad_trace: tl.Trace) -> None:
    """hot_path_rows returns ranked row dicts + honesty attrs, no pandas."""

    rows, attrs = dbg.hot_path_rows(grad_trace)
    assert rows, "expected at least one costed source line"
    assert all(isinstance(row, dict) for row in rows)
    costs = [float(row["total_cost"]) for row in rows]
    assert costs == sorted(costs, reverse=True)
    assert attrs["metric"] == "flops"
    assert attrs["torchlens_capture_honesty"]["capture_status"] == "complete"


@pytest.mark.usefixtures("_no_pandas")
def test_dead_neurons_rows_runs_without_pandas(grad_trace: tl.Trace) -> None:
    """dead_neurons_rows returns per-op row dicts + honesty attrs, no pandas."""

    rows, attrs = dbg.dead_neurons_rows(grad_trace)
    assert isinstance(rows, list)
    assert "insufficient sample" in attrs["note"]
    assert "torchlens_capture_honesty" in attrs


@pytest.mark.usefixtures("_no_pandas")
def test_compare_rows_runs_without_pandas(grad_trace: tl.Trace) -> None:
    """compare_rows returns matched row dicts + per-trace honesty, no pandas."""

    rows, attrs = dbg.compare_rows(grad_trace, grad_trace)
    assert rows, "expected one row per pass-qualified op"
    assert attrs["torchlens_capture_honesty"]["trace_a"]["capture_status"] == "complete"
    assert attrs["torchlens_capture_honesty"]["trace_b"]["capture_status"] == "complete"


@pytest.mark.usefixtures("_no_pandas")
def test_gradient_flow_audit_rows_runs_without_pandas(grad_trace: tl.Trace) -> None:
    """gradient_flow_audit_rows returns ranked row dicts, no pandas."""

    rows, attrs = dbg.gradient_flow_audit_rows(grad_trace)
    assert rows, "expected one row per saved-grad op"
    severities = [int(row["severity"]) for row in rows]
    assert severities == sorted(severities, reverse=True)
    assert attrs["bwd"] == 1
    assert {"vanishing", "exploding", "dead", "unavailable"}.issubset(attrs)
    assert attrs["torchlens_capture_honesty"]["capture_status"] == "complete"


@pytest.mark.usefixtures("_no_pandas")
def test_dataframe_views_teach_the_tabular_extra(grad_trace: tl.Trace) -> None:
    """Without pandas the DataFrame views refuse with the [tabular] remedy."""

    for view in (dbg.hot_path, dbg.dead_neurons, dbg.gradient_flow_audit):
        with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
            view(grad_trace)
    with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
        dbg.compare(grad_trace, grad_trace)


def test_view_attrs_match_core_attrs(grad_trace: tl.Trace) -> None:
    """With pandas present, each DataFrame view carries the core's attrs."""

    pytest.importorskip("pandas")
    frame = dbg.gradient_flow_audit(grad_trace)
    rows, attrs = dbg.gradient_flow_audit_rows(grad_trace)
    assert len(frame) == len(rows)
    assert list(frame["op"]) == [row["op"] for row in rows]
    for key, value in attrs.items():
        assert frame.attrs[key] == value
