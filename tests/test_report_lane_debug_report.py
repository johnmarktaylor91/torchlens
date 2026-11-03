"""Every report-shaped diagnostic helper returns the one ``tl.report.Report``.

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
from torchlens.debug._audit import HealthReport
from torchlens.debug._graph import _compare_rows, dead_neurons_rows as _dead_neurons_rows
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
    captured.backward(captured[captured.output_layers[0]].out.sum())
    try:
        yield captured
    finally:
        captured.cleanup()


def _helper_calls() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Each report-shaped helper, called on the module fixtures."""

    debug = tl.debug
    return {
        "audit_trace": lambda f: f["trace"].audit(),
        "check_determinism": lambda f: tl.debug.check_determinism(f["model"], f["x"]),
        "dtype_range_audit": lambda f: tl.debug.dtype_range(f["trace"]),
        "flops_vs_dispatch": lambda f: tl.report.flops_vs_dispatch(f["model"], f["x"]),
        "graph_breaks": lambda f: debug.graph_breaks(f["model"], f["x"]),
        "guess_input_shape": lambda f: debug.guess_input_shape(nn.Linear(4, 2)),
        "lineage": lambda f: debug.lineage(f["trace"], f["trace"].output_layers[0]),
        "find_nan": lambda f: tl.debug.find_nan(f["model"], f["x"]),
        "bisect_nan": lambda f: debug.bisect_nan(f["trace"]),
        "nan_report": lambda f: tl.debug.nonfinite_origin(f["model"], f["x"]),
        "bisect_nan_backward": lambda f: debug.bisect_nan_backward(f["grad_trace"]),
        "bisect_precision": lambda f: debug.bisect_precision(f["model"], f["x"]),
        "param_health": lambda f: tl.debug.param_health(f["model"]),
        "first_bad_thing": lambda f: tl.debug.find_nan(f["model"], f["x"]).first_bad_thing,
        "hot_path": lambda f: debug.hot_path(f["trace"]),
        "dead_neurons": lambda f: debug.dead_neurons(f["trace"]),
        "compare": lambda f: tl.debug.compare(f["trace"], f["trace"]),
        "recompute_candidates": lambda f: debug.recompute_candidates(f["trace"]),
        "gradient_flow_audit": lambda f: tl.debug.grad_health(f["grad_trace"]),
        "compare_params": lambda f: tl.debug.compare_params(f["model"], f["model"]),
    }


@pytest.fixture(scope="module")
def fixtures(model: nn.Module, x: torch.Tensor, trace: Any, grad_trace: Any) -> dict[str, Any]:
    return {"model": model, "x": x, "trace": trace, "grad_trace": grad_trace}


# Kept by the shrink-r4 coverage proof: sole branch coverage of _debug_report.py, debug/_audit.py and more.
# Rows kept by the shrink-r4 per-id coverage proof (the rest were covered by kept tests).
@pytest.mark.parametrize(
    "helper",
    sorted(
        set(_helper_calls())
        - {
            "bisect_nan",
            "compare",
            "compare_params",
            "dead_neurons",
            "gradient_flow_audit",
            "hot_path",
        }
    ),
)
def test_helper_returns_the_debug_report(helper: str, fixtures: dict[str, Any]) -> None:
    report = _helper_calls()[helper](fixtures)

    assert isinstance(report, tl.report.Report)
    assert isinstance(report, BoundedReport)
    assert report.report_type and report.subject and report.status
    payload = report.to_dict()
    if type(report).to_dict is tl.report.Report.to_dict:
        assert payload["report_type"] == report.report_type
        assert payload["status"] == report.status
    json.dumps(payload)
    assert isinstance(report.to_pandas(), pd.DataFrame)

    text = str(report)
    if isinstance(report, HealthReport):
        assert text.splitlines()[0] == repr(report)
        assert text.splitlines()[-1].startswith("  more ")
    elif type(report)._prints_repr:
        assert text == repr(report)
    else:
        lines = text.splitlines()
        assert lines[0].startswith(f"{report.report_type}({report.subject}) status=")
        assert lines[-1] == "  More: .to_pandas()  .to_dict()"
        assert len([line for line in lines if line.startswith("  - ")]) <= REPORT_MAX_FINDINGS
    card = report.card()
    assert card.splitlines()[0].startswith(report.report_type)


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


# Kept at the C-063 stack merge: debug/_graph.py changed after the coverage proof.
def test_dead_neurons_frame_is_unchanged(trace: Any) -> None:
    rows, attrs = _dead_neurons_rows(trace, dim=None, max_activation=0.0)
    columns = [
        "op",
        "total_units",
        "dead_count",
        "dead_frac",
        "sample_dead_idx",
        "reason",
        "source_line",
    ]
    _assert_same(tl.debug.dead_neurons(trace), _frame(rows, columns, attrs))


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_gradients.py.


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
def test_compare_frame_is_unchanged(trace: Any) -> None:
    rows, attrs = _compare_rows(trace, trace)
    columns = [
        "op",
        "status",
        "shape_match",
        "dtype_match",
        "max_abs",
        "mean_abs",
        "allclose",
        "reason",
        "source_line",
    ]
    _assert_same(tl.debug.compare(trace, trace), _frame(rows, columns, attrs))


def test_gradient_flow_audit_frame_is_unchanged(grad_trace: Any) -> None:
    from torchlens.debug._gradients import _GRAD_AUDIT_COLUMNS, gradient_flow_audit_rows

    rows, attrs = gradient_flow_audit_rows(grad_trace)
    expected = _frame(rows, list(_GRAD_AUDIT_COLUMNS), attrs)
    _assert_same(tl.debug.grad_health(grad_trace), expected)


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.


def test_recompute_candidates_frame_is_unchanged(trace: Any) -> None:
    from torchlens.debug._common import _compute_ops, _op_label, _source_line

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
                "source_line": _source_line(op),
            }
        )
    rows.sort(key=lambda row: float(row["mem_per_flop"]), reverse=True)
    columns = [
        "op",
        "activation_memory",
        "flops_forward",
        "mem_per_flop",
        "suggested",
        "source_line",
    ]
    expected = _frame(rows, columns, {"total_freeable": sum(r["activation_memory"] for r in rows)})
    assert rows, "the fixture must yield candidates"
    _assert_same(tl.debug.recompute_candidates(trace), expected)
