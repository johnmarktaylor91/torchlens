"""U42.2: every report-shaped ``tl.debug`` helper returns the one ``DebugReport``.

One test per helper checks the common surface (type, lead facts, ``to_dict``,
``to_pandas``, the printed form). The six table helpers additionally pin that
``.to_pandas()`` equals the frame they returned before the migration, so the
break is the return type only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.report._report_protocol import REPORT_MAX_FINDINGS, BoundedReport

pd = pytest.importorskip("pandas")
assert_frame_equal = pd.testing.assert_frame_equal


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture(scope="module")
def model() -> nn.Module:
    torch.manual_seed(0)
    return _TinyMLP().eval()


@pytest.fixture(scope="module")
def x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 4)


@pytest.fixture(scope="module")
def trace(model: nn.Module, x: torch.Tensor) -> Iterator[Any]:
    captured = tl.trace(model, x)
    try:
        yield captured
    finally:
        captured.cleanup()


@pytest.fixture(scope="module")
def grad_trace(model: nn.Module, x: torch.Tensor) -> Iterator[Any]:
    captured = tl.trace(model, x, capture=CaptureOptions(save_grads=True))
    captured.log_backward(captured[captured.output_layers[0]].out.sum())
    try:
        yield captured
    finally:
        captured.cleanup()


def _helper_calls() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Each report-shaped helper, called on the module fixtures."""

    debug = tl.debug
    return {
        "audit_trace": lambda f: debug.audit_trace(f["trace"]),
        "check_determinism": lambda f: debug.check_determinism(f["model"], f["x"]),
        "dtype_range_audit": lambda f: debug.dtype_range_audit(f["trace"]),
        "flops_vs_dispatch": lambda f: debug.flops_vs_dispatch(f["model"], f["x"]),
        "graph_breaks": lambda f: debug.graph_breaks(f["model"], f["x"]),
        "infer_input_shape": lambda f: debug.infer_input_shape(nn.Linear(4, 2)),
        "lineage": lambda f: debug.lineage(f["trace"], f["trace"].output_layers[0]),
        "find_nan": lambda f: debug.find_nan(f["model"], f["x"]),
        "bisect_nan": lambda f: debug.bisect_nan(f["trace"]),
        "nan_report": lambda f: debug.nan_report(f["model"], f["x"]),
        "bisect_nan_backward": lambda f: debug.bisect_nan_backward(f["grad_trace"]),
        "bisect_precision": lambda f: debug.bisect_precision(f["model"], f["x"]),
        "param_health": lambda f: debug.param_health(f["model"]),
        "first_bad_thing": lambda f: debug.find_nan(f["model"], f["x"]).first_bad_thing,
        "hot_path": lambda f: debug.hot_path(f["trace"]),
        "dead_neurons": lambda f: debug.dead_neurons(f["trace"]),
        "compare": lambda f: debug.compare(f["trace"], f["trace"]),
        "recompute_candidates": lambda f: debug.recompute_candidates(f["trace"]),
        "gradient_flow_audit": lambda f: debug.gradient_flow_audit(f["grad_trace"]),
        "compare_params": lambda f: debug.compare_params(f["model"], f["model"]),
    }


@pytest.fixture(scope="module")
def fixtures(model: nn.Module, x: torch.Tensor, trace: Any, grad_trace: Any) -> dict[str, Any]:
    return {"model": model, "x": x, "trace": trace, "grad_trace": grad_trace}


@pytest.mark.parametrize("helper", sorted(_helper_calls()))
def test_helper_returns_the_debug_report(helper: str, fixtures: dict[str, Any]) -> None:
    report = _helper_calls()[helper](fixtures)

    assert isinstance(report, tl.debug.DebugReport)
    assert isinstance(report, BoundedReport)
    assert report.report_type and report.subject and report.status
    payload = report.to_dict()
    if type(report).to_dict is tl.debug.DebugReport.to_dict:
        assert payload["report_type"] == report.report_type
        assert payload["status"] == report.status
    json.dumps(payload)
    assert isinstance(report.to_pandas(), pd.DataFrame)

    text = str(report)
    if type(report)._prints_repr:
        assert text == repr(report)
    else:
        lines = text.splitlines()
        assert lines[0].startswith(f"{report.report_type}({report.subject}) status=")
        assert lines[-1] == "  More: .to_pandas()  .to_dict()"
        assert len([line for line in lines if line.startswith("  - ")]) <= REPORT_MAX_FINDINGS
    card = report.card()
    assert card.splitlines()[0].startswith(report.report_type)


def test_param_audit_prints_exactly_as_before(model: nn.Module) -> None:
    """ParamHealthReport kept its house-voice summary: str() is its unchanged repr."""

    audit = tl.debug.param_health(model)
    text = str(audit)
    assert text == audit.__repr__()
    assert text.splitlines()[0].startswith("ParamHealthReport: ")


def test_table_report_never_is_a_dataframe(trace: Any) -> None:
    report = tl.debug.hot_path(trace)

    assert not isinstance(report, pd.DataFrame)
    assert not hasattr(report, "columns")
    assert not hasattr(report, "attrs")
    frame = report.to_pandas()
    frame["extra"] = 1
    assert "extra" not in report.to_pandas().columns


# --- the six table helpers: .to_pandas() is the pre-migration frame ----------


def _frame(rows: list[dict[str, Any]], columns: list[str], attrs: dict[str, Any]) -> Any:
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs.update(attrs)
    return frame


def _assert_same(report: Any, expected: Any) -> None:
    actual = report.to_pandas()
    assert_frame_equal(actual, expected)
    for key, value in expected.attrs.items():
        assert actual.attrs[key] == value


def test_hot_path_frame_is_unchanged(trace: Any) -> None:
    rows, attrs = tl.debug.hot_path_rows(trace, "flops")
    columns = ["source_file:line", "op_count", "total_cost", "pct_total"]
    _assert_same(tl.debug.hot_path(trace), _frame(rows, columns, attrs))


def test_dead_neurons_frame_is_unchanged(trace: Any) -> None:
    rows, attrs = tl.debug.dead_neurons_rows(trace, dim=1, threshold=0.0)
    columns = ["op", "total_units", "dead_count", "dead_frac", "sample_dead_idx", "reason"]
    _assert_same(tl.debug.dead_neurons(trace), _frame(rows, columns, attrs))


def test_compare_frame_is_unchanged(trace: Any) -> None:
    rows, attrs = tl.debug.compare_rows(trace, trace)
    columns = [
        "op",
        "status",
        "shape_match",
        "dtype_match",
        "max_abs",
        "mean_abs",
        "allclose",
        "reason",
    ]
    _assert_same(tl.debug.compare(trace, trace), _frame(rows, columns, attrs))


def test_gradient_flow_audit_frame_is_unchanged(grad_trace: Any) -> None:
    from torchlens.debug._gradients import _GRAD_AUDIT_COLUMNS

    rows, attrs = tl.debug.gradient_flow_audit_rows(grad_trace)
    expected = _frame(rows, list(_GRAD_AUDIT_COLUMNS), attrs)
    _assert_same(tl.debug.gradient_flow_audit(grad_trace), expected)


def test_recompute_candidates_frame_is_unchanged(trace: Any) -> None:
    from torchlens.debug._common import _compute_ops, _op_label

    rows = []
    for op in _compute_ops(trace):
        memory, flops = op.activation_memory, op.flops_forward
        if memory is None or flops is None or float(flops) <= 0:
            continue
        rows.append(
            {
                "op": _op_label(op),
                "activation_memory": int(memory),
                "flops_forward": float(flops),
                "mem_per_flop": int(memory) / float(flops),
                "suggested": False,
            }
        )
    rows.sort(key=lambda row: float(row["mem_per_flop"]), reverse=True)
    columns = ["op", "activation_memory", "flops_forward", "mem_per_flop", "suggested"]
    expected = _frame(rows, columns, {"total_freeable": sum(r["activation_memory"] for r in rows)})
    assert rows, "the fixture must yield candidates"
    _assert_same(tl.debug.recompute_candidates(trace), expected)


def test_compare_params_frame_is_unchanged() -> None:
    torch.manual_seed(0)
    left = nn.Linear(4, 2)
    right = nn.Linear(4, 2)
    right.load_state_dict(left.state_dict())
    with torch.no_grad():
        right.bias.add_(1.0)
    row = {
        "kind": "parameter",
        "status": "present-in-both",
        "shape_match": True,
        "dtype_match": True,
        "reason": "",
    }
    rows = [
        {"name": "bias", **row, "max_abs": 1.0, "mean_abs": 1.0, "allclose": False},
        {"name": "weight", **row, "max_abs": 0.0, "mean_abs": 0.0, "allclose": True},
    ]
    columns = [
        "name",
        "kind",
        "status",
        "shape_match",
        "dtype_match",
        "max_abs",
        "mean_abs",
        "allclose",
        "reason",
    ]
    expected = _frame(rows, columns, {"matched": 1, "value_diverged": 1})
    _assert_same(tl.debug.compare_params(left, right), expected)
