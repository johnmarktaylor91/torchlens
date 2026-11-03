"""forks.13: a CPU forward peak delta of 0 reads ``None`` with a recorded reason.

JMT 2026-10-12: "a, but zero becomes None with a reason". CPU forward peak
only: positive values and their basis, CUDA zero semantics, MPS and the
backward peak are unchanged. The host RSS reader is pinned so the zero (and
the positive) delta is deterministic. Every capture here opts in with
``measure_host_peak_memory=True`` (measure.2: without it a CPU peak is never
measured); the process-wide reset is pinned too, so nothing is written.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._capture._forward_helpers as capture_forward_helpers
from torchlens._capture import peak_memory
from torchlens._capture.peak_memory import (
    CPU_ZERO_PEAK_REASON,
    CPU_ZERO_PEAK_WITH_PYTHON_REASON,
    forward_peak_value,
)
from torchlens._io import ArtifactError
from torchlens._observe import format_pass_peak
from torchlens._observe._peaks import pass_peak_facts
from torchlens.options import CaptureOptions

_FLAG = "measure_python_peak_memory=True"


def _rss(monkeypatch: pytest.MonkeyPatch, *readings: int) -> None:
    """Serve successive host RSS readings to the forward bracket."""

    values = iter(readings)
    last = [readings[-1]]

    def fake_rss() -> int:
        last[0] = next(values, last[0])
        return last[0]

    monkeypatch.setattr(capture_forward_helpers, "process_rss_bytes", fake_rss)
    # The CPU peak also reads the resident high-water: serve the largest reading.
    monkeypatch.setattr(peak_memory, "peak_rss_bytes", lambda: max(readings))
    monkeypatch.setattr(capture_forward_helpers, "reset_peak_rss", lambda: True)


def _capture(**kwargs: Any) -> Any:
    torch.manual_seed(0)
    kwargs.setdefault("capture", CaptureOptions(measure_host_peak_memory=True))
    return tl.trace(nn.Linear(4, 2), torch.randn(2, 4), **kwargs)


def test_every_printed_surface_shows_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _rss(monkeypatch, 1_000_000)
    trace = _capture()

    facts = pass_peak_facts(trace)
    assert facts.value_bytes is None
    assert facts.meaning == CPU_ZERO_PEAK_REASON
    assert format_pass_peak(trace) == (
        f"Live forward-memory peak: unavailable ({CPU_ZERO_PEAK_REASON})"
    )
    summary = trace.summary()
    assert f"forward peak unavailable ({CPU_ZERO_PEAK_REASON})" in summary.render("unicode")
    assert _FLAG in str(summary)
    for text in (str(trace), repr(trace), str(trace.profile()), trace.explain()):
        assert isinstance(text, str)


def test_python_peak_opt_in_reason_when_both_read_zero() -> None:
    assert forward_peak_value("cpu", 0, python_peak_measured=True) == (
        None,
        CPU_ZERO_PEAK_WITH_PYTHON_REASON,
    )
    assert _FLAG in CPU_ZERO_PEAK_WITH_PYTHON_REASON


def test_cuda_mps_and_backward_semantics_are_unchanged() -> None:
    assert forward_peak_value("cuda", 0, python_peak_measured=False) == (0, None)
    assert forward_peak_value("mps", 0, python_peak_measured=False) == (0, None)
    assert forward_peak_value("cpu", 512, python_peak_measured=False) == (512, None)

    cuda = SimpleNamespace(_provenance_forward_memory_backend="cuda", _cost_forward_peak_memory=0)
    assert pass_peak_facts(cuda).value_bytes == 0
    assert "high-water" in pass_peak_facts(cuda).meaning
    mps = SimpleNamespace(_provenance_forward_memory_backend="mps", _cost_forward_peak_memory=0)
    assert pass_peak_facts(mps).value_bytes == 0
    backward = SimpleNamespace(
        _provenance_backward_memory_backend="cpu", _cost_backward_peak_memory=0
    )
    assert pass_peak_facts(backward, "backward").value_bytes == 0


@pytest.mark.parametrize(
    ("peak", "reason", "why"),
    [(None, 5, "not_a_string"), (tl.types.Bytes(4096), "forged", "reason_beside_a_value")],
)
def test_malformed_reason_refuses_at_load(tmp_path: Path, peak: Any, reason: Any, why: str) -> None:
    trace = _capture()
    trace._cost_forward_peak_memory = peak
    trace._cost_forward_peak_memory_reason = reason
    path = tmp_path / f"forged_{why}.tlspec"
    trace.save(path)

    with pytest.raises(ArtifactError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "artifact_forward_peak_memory_invalid"
    assert excinfo.value.fields["reason"] == why
