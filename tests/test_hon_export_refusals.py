"""Typed "expected a Trace" refusals for the graph and tracker exporters.

U09.2 + U10.5: ``tl.export.netron`` and ``tl.export.model_explorer`` share one
refusal (``export_expected_trace``) at their top. U33.2: the one-shot
``tl.export.tensorboard`` exporter used to write plausible zeros for a
non-Trace input; it now refuses with the same code. Each test also runs the
remedy the refusal prints and checks it works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ArgumentTypeError


class _Tiny(nn.Module):
    """Two linear layers around a ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class _RecordingWriter:
    """Minimal SummaryWriter stand-in that records every call."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, Any]] = []

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        self.rows.append((tag, value))

    def add_text(self, tag: str, value: Any, step: int) -> None:
        self.rows.append((tag, value))


def _assert_expected_trace_refusal(exc: ArgumentTypeError, exporter: str) -> None:
    """The shared refusal carries the code, the exporter name, and a remedy."""

    assert exc.fields["code"] == "export_expected_trace"
    assert exc.fields["received_type"] == "dict"
    assert exporter in str(exc)
    assert "tl.trace(model, x)" in exc.fields["remedy"]


def test_netron_refuses_a_non_trace_and_the_remedy_runs(tmp_path: Path) -> None:
    """U09.2: a non-Trace refuses typed before any projection work."""

    destination = tmp_path / "graph.json"
    with pytest.raises(ArgumentTypeError) as info:
        tl.export.netron({"not": "a trace"}, destination)
    _assert_expected_trace_refusal(info.value, "tl.export.netron")
    assert not destination.exists()

    model, x = _Tiny().eval(), torch.randn(2, 4)
    written = tl.export.netron(tl.trace(model, x), destination)
    assert written == destination and destination.stat().st_size > 0


def test_model_explorer_refuses_a_non_trace_and_the_remedy_runs(tmp_path: Path) -> None:
    """U10.5: the Model Explorer exporter shares the netron refusal."""

    destination = tmp_path / "graph.json"
    with pytest.raises(ArgumentTypeError) as info:
        tl.export.model_explorer({"not": "a trace"}, destination)
    _assert_expected_trace_refusal(info.value, "tl.export.model_explorer")
    assert not destination.exists()

    model, x = _Tiny().eval(), torch.randn(2, 4)
    written = tl.export.model_explorer(tl.trace(model, x), destination)
    assert written == destination and destination.stat().st_size > 0


def test_tensorboard_refuses_a_non_trace_instead_of_writing_zeros() -> None:
    """U33.2: no fabricated zero scalars reach the writer for a non-Trace."""

    writer = _RecordingWriter()
    with pytest.raises(ArgumentTypeError) as info:
        tl.export.tensorboard({"not": "a trace"}, writer, step=0)
    _assert_expected_trace_refusal(info.value, "tl.export.tensorboard")
    assert writer.rows == [], "a refused export must write nothing"

    trace = tl.trace(_Tiny().eval(), torch.randn(2, 4))
    tl.export.tensorboard(trace, writer, step=0)
    scalars = dict(writer.rows)
    assert scalars["torchlens/num_layers"] == len(trace.layer_list) > 0
