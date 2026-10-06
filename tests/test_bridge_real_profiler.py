"""Profiler join against a REAL ``torch.profiler`` Chrome trace.

Each op event is assigned to exactly one layer: the k-th ``aten::conv2d``
event to the k-th conv layer, so per-layer rows sum to the profiler's own
event total with one event per row (the old name-substring join gave every
conv layer every conv event and doubled the total).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = [pytest.mark.optional]


class _TwoConv(nn.Module):
    """Two convs of the same op type, no in-place ops."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.mean(dim=(2, 3)))


class _NoBiasAttn(nn.Module):
    """No-bias linears on a 3D input (each calls ``aten::matmul`` internally) plus two matmuls."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(8, 8, bias=False)
        self.k = nn.Linear(8, 8, bias=False)
        self.v = nn.Linear(8, 6, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.q(x), self.k(x).transpose(1, 2))
        return torch.matmul(scores, self.v(x))


def _profile(
    model: nn.Module, x: torch.Tensor, path: Path, forwards: int, *, record_shapes: bool = False
) -> dict:
    """Profile ``forwards`` forward passes and return the loaded Chrome trace."""

    from torch.profiler import ProfilerActivity, profile

    with (
        torch.no_grad(),
        profile(activities=[ProfilerActivity.CPU], record_shapes=record_shapes) as prof,
    ):
        for _ in range(forwards):
            model(x)
    prof.export_chrome_trace(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _aten_events(trace: dict, name: str) -> list[dict]:
    return [
        e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("name") == f"aten::{name}"
    ]


@pytest.mark.parametrize("forwards", [1, 2])
def test_repeated_op_rows_partition_the_profiler_total(tmp_path: Path, forwards: int) -> None:
    """conv2d and relu rows each hold `forwards` events and sum to the aten total."""

    torch.manual_seed(0)
    model = _TwoConv().eval()
    x = torch.randn(2, 3, 16, 16)
    log = tl.trace(model, x)
    path = tmp_path / "trace.json"
    trace = _profile(model, x, path, forwards)

    joined = tl.bridge.profiler.join(log, path)
    for func_name in ("conv2d", "relu"):
        events = _aten_events(trace, func_name)
        rows = [row for row in joined["ops"] if row["func_name"] == func_name]
        assert len(rows) == 2
        assert len(events) == 2 * forwards
        assert all(row["kineto_event_count"] == forwards for row in rows)
        total = sum(float(e["dur"]) for e in events)
        assert sum(row["kineto_duration_us"] for row in rows) == pytest.approx(total)
        assigned = [id(e) for row in rows for e in row["kineto_events"]]
        assert len(set(assigned)) == len(assigned)
    assert "conv2d" not in joined["mismatched_op_types"]
    assert "aten::conv2d" not in joined["unmatched_event_counts"]


def _input_dims(event: dict) -> list[list[int]]:
    """Return an event's first two recorded input shapes."""

    return [list(dims) for dims in (event.get("args") or {}).get("Input Dims", [])][:2]


@pytest.mark.parametrize("forwards", [1, 2])
def test_matmul_inside_linear_is_not_a_matmul_layers_event(tmp_path: Path, forwards: int) -> None:
    """Each matmul row holds its own top-level matmul, never a linear's internal one.

    A no-bias linear on a 3D input runs ``aten::matmul`` inside ``aten::linear``:
    with two forwards the old same-name-only nesting filter saw 10 matmul events
    over 2 matmul layers, a clean "5 forwards", and assigned internal calls to
    the matmul rows. Placement is checked by recorded input shapes.
    """

    torch.manual_seed(0)
    model = _NoBiasAttn().eval()
    x = torch.randn(2, 5, 8)
    log = tl.trace(model, x)
    path = tmp_path / "trace.json"
    trace = _profile(model, x, path, forwards, record_shapes=True)
    all_matmul = _aten_events(trace, "matmul")
    assert len(all_matmul) > 2 * forwards, "no internal matmul: the test proves nothing"

    joined = tl.bridge.profiler.join(log, path)
    expected = {
        "linear": [[[2, 5, 8], [8, 8]], [[2, 5, 8], [8, 8]], [[2, 5, 8], [6, 8]]],
        "matmul": [[[2, 5, 8], [2, 8, 5]], [[2, 5, 5], [2, 5, 6]]],
    }
    for func_name, dims in expected.items():
        rows = [row for row in joined["ops"] if row["func_name"] == func_name]
        placed = [[_input_dims(event) for event in row["kineto_events"]] for row in rows]
        assert placed == [[row_dims] * forwards for row_dims in dims], func_name
        assert func_name not in joined["mismatched_op_types"]
    assert joined["forwards"] == forwards
    internal = len(all_matmul) - 2 * forwards
    assert joined["unmatched_event_counts"]["aten::matmul"] == internal
