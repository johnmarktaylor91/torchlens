"""forks.13: the forward peak and its reason move together through a rerun.

``log.run`` refreshes a trace in place when the rerun's labels match. The
refresh must carry the rerun's ``forward_peak_memory_reason`` with its
``forward_peak_memory``: a CPU-zero first capture (``None`` plus a reason)
followed by a positive rerun otherwise keeps the stale reason beside the new
number, and the load-side pair check refuses the saved artifact. The host
measurement is pinned so both orders are deterministic. The captures opt in
with ``measure_host_peak_memory=True`` (measure.2), which the rerun inherits.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture import peak_memory, trace as capture_trace
from torchlens._capture.peak_memory import CPU_ZERO_PEAK_REASON
from torchlens.options import CaptureOptions, ReplayOptions


class _LinearRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + 1


def _measure(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """Pin the host forward peak that every later CPU capture measures."""

    monkeypatch.setattr(peak_memory, "host_forward_peak", lambda *args, **kwargs: value)


def _rerun_log(
    monkeypatch: pytest.MonkeyPatch, first: int, rerun: int
) -> tuple[tl.Trace, nn.Module]:
    torch.manual_seed(0)
    model = _LinearRelu().eval()
    _measure(monkeypatch, first)
    log = tl.trace(
        model,
        torch.randn(2, 3),
        capture=CaptureOptions(intervention_ready=True, measure_host_peak_memory=True),
    )
    assert (log.forward_peak_memory is None) == (first == 0)
    _measure(monkeypatch, rerun)
    log.attach_hooks(tl.func("relu"), tl.zero_ablate())
    log.run(model, torch.randn(2, 3))
    return log, model


def test_zero_then_positive_rerun_drops_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log, _ = _rerun_log(monkeypatch, first=0, rerun=4096)

    assert int(log.forward_peak_memory) == 4096
    assert log.forward_peak_memory_reason is None
    log.save(tmp_path / "bundle")
    loaded = tl.load(tmp_path / "bundle")
    assert int(loaded.forward_peak_memory) == 4096
    assert loaded.forward_peak_memory_reason is None


def test_positive_then_zero_rerun_carries_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log, _ = _rerun_log(monkeypatch, first=4096, rerun=0)

    assert log.forward_peak_memory is None
    assert log.forward_peak_memory_reason == CPU_ZERO_PEAK_REASON
    log.save(tmp_path / "bundle")
    loaded = tl.load(tmp_path / "bundle")
    assert loaded.forward_peak_memory is None
    assert loaded.forward_peak_memory_reason == CPU_ZERO_PEAK_REASON


def test_append_rerun_after_a_cpu_zero_round_trips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log, model = _rerun_log(monkeypatch, first=0, rerun=4096)
    log.run(model, torch.randn(1, 3), replay=ReplayOptions(append=True))

    assert log.forward_peak_memory_reason is None
    log.save(tmp_path / "append_bundle")
    loaded = tl.load(tmp_path / "append_bundle")
    assert loaded.is_appended is True
    assert int(loaded.forward_peak_memory) == int(log.forward_peak_memory)
    assert loaded.forward_peak_memory_reason is None


def test_cuda_bracket_clears_a_stale_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    allocated = iter([100, 200])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: next(allocated))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 0)
    trace = SimpleNamespace(
        forward_peak_memory=None, forward_peak_memory_reason=CPU_ZERO_PEAK_REASON
    )

    with capture_trace._forward_peak_memory_bracket(trace, SimpleNamespace(type="cuda")):
        pass

    assert int(trace.forward_peak_memory) == 200
    assert trace.forward_peak_memory_reason is None
    assert trace.forward_memory_backend == "cuda"
