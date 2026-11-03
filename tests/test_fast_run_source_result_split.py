"""``trace.run(inputs=..., fast=True)`` on a live trace returns a result Trace and leaves the source untouched."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _Two(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def _capture() -> tuple[nn.Module, tl.Trace, torch.Tensor]:
    torch.manual_seed(0)
    model = _Two()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x)
    return model, trace, x


def test_fast_run_returns_a_new_trace_and_leaves_the_source_untouched() -> None:
    model, trace, x = _capture()
    label = trace.saved_layers[-1].ops[-1].label
    source_out = trace[label].out.clone()
    source_state = trace.state
    source_last_run = trace.last_run

    result = trace.run(inputs=x + 1.0, fast=True)

    assert result.trace is not trace
    assert torch.equal(trace[label].out, source_out)
    assert trace.state is source_state
    assert trace.last_run == source_last_run
    assert not torch.equal(result.trace[label].out, source_out)
    assert result.trace.last_run["engine"] == "guarded_fast"
    assert model is not None


def test_repeated_fast_runs_refresh_the_result_trace_only() -> None:
    _, trace, x = _capture()
    label = trace.saved_layers[-1].ops[-1].label
    source_out = trace[label].out.clone()

    first = trace.run(inputs=x + 1.0, fast=True)
    first_out = first.trace[label].out.clone()
    second = trace.run(inputs=x + 2.0, fast=True)

    assert not torch.equal(first_out, second.trace[label].out)
    assert torch.equal(trace[label].out, source_out)
    assert torch.equal(second.trace[label].out, trace.run(inputs=x + 2.0).trace[label].out)
