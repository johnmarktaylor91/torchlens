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
from torchlens._io import TorchLensIOError
from torchlens.capture import peak_memory, trace as capture_trace
from torchlens.capture.peak_memory import (
    CPU_ZERO_PEAK_REASON,
    CPU_ZERO_PEAK_WITH_PYTHON_REASON,
    forward_peak_value,
)
from torchlens.observe import format_pass_peak
from torchlens.observe._peaks import pass_peak_facts
from torchlens.options import CaptureOptions

_FLAG = "measure_python_peak_memory=True"


def _rss(monkeypatch: pytest.MonkeyPatch, *readings: int) -> None:
    """Serve successive host RSS readings to the forward bracket."""

    values = iter(readings)
    last = [readings[-1]]

    def fake_rss() -> int:
        last[0] = next(values, last[0])
        return last[0]

    monkeypatch.setattr(capture_trace, "process_rss_bytes", fake_rss)
    # The CPU peak also reads the resident high-water: serve the largest reading.
    monkeypatch.setattr(peak_memory, "peak_rss_bytes", lambda: max(readings))
    monkeypatch.setattr(capture_trace, "reset_peak_rss", lambda: True)


def _capture(**kwargs: Any) -> Any:
    torch.manual_seed(0)
    kwargs.setdefault("capture", CaptureOptions(measure_host_peak_memory=True))
    return tl.trace(nn.Linear(4, 2), torch.randn(2, 4), **kwargs)


def test_cpu_zero_delta_reads_none_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _rss(monkeypatch, 1_000_000)
    trace = _capture()

    assert trace.forward_memory_backend == "cpu"
    assert trace.forward_peak_memory is None
    assert trace.forward_peak_memory_reason == CPU_ZERO_PEAK_REASON
    assert _FLAG in trace.forward_peak_memory_reason


def test_cpu_positive_delta_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _rss(monkeypatch, 1_000_000, 1_004_096)
    trace = _capture()

    assert isinstance(trace.forward_peak_memory, tl.Bytes)
    assert int(trace.forward_peak_memory) == 4096
    assert trace.forward_peak_memory_reason is None
    assert pass_peak_facts(trace).meaning == "process RSS growth (NOT a tensor peak)"


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
    for text in (str(trace), repr(trace), str(trace.profile()), tl.report.explain(trace)):
        assert isinstance(text, str)


def test_python_peak_opt_in_reason_when_both_read_zero() -> None:
    assert forward_peak_value("cpu", 0, python_peak_measured=True) == (
        None,
        CPU_ZERO_PEAK_WITH_PYTHON_REASON,
    )
    assert _FLAG in CPU_ZERO_PEAK_WITH_PYTHON_REASON


def test_opt_in_positive_python_peak_is_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _rss(monkeypatch, 1_000_000)
    trace = _capture(
        capture=CaptureOptions(measure_python_peak_memory=True, measure_host_peak_memory=True)
    )

    assert isinstance(trace.forward_peak_memory, tl.Bytes)
    assert int(trace.forward_peak_memory) > 0
    assert trace.forward_peak_memory_reason is None


def test_cuda_mps_and_backward_semantics_are_unchanged() -> None:
    assert forward_peak_value("cuda", 0, python_peak_measured=False) == (0, None)
    assert forward_peak_value("mps", 0, python_peak_measured=False) == (0, None)
    assert forward_peak_value("cpu", 512, python_peak_measured=False) == (512, None)

    cuda = SimpleNamespace(forward_memory_backend="cuda", forward_peak_memory=0)
    assert pass_peak_facts(cuda).value_bytes == 0
    assert "high-water" in pass_peak_facts(cuda).meaning
    mps = SimpleNamespace(forward_memory_backend="mps", forward_peak_memory=0)
    assert pass_peak_facts(mps).value_bytes == 0
    backward = SimpleNamespace(backward_memory_backend="cpu", backward_peak_memory=0)
    assert pass_peak_facts(backward, "backward").value_bytes == 0


def test_none_and_reason_round_trip_through_save_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _rss(monkeypatch, 1_000_000)
    trace = _capture()
    path = tmp_path / "zero_peak.tlspec"
    tl.save(trace, path)

    loaded = tl.load(path)
    assert loaded.forward_peak_memory is None
    assert loaded.forward_peak_memory_reason == CPU_ZERO_PEAK_REASON
    assert format_pass_peak(loaded).endswith(f"({CPU_ZERO_PEAK_REASON})")


def test_positive_value_round_trips_with_no_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _rss(monkeypatch, 1_000_000, 1_004_096)
    trace = _capture()
    path = tmp_path / "positive_peak.tlspec"
    tl.save(trace, path)

    loaded = tl.load(path)
    assert int(loaded.forward_peak_memory) == 4096
    assert loaded.forward_peak_memory_reason is None


def test_legacy_artifact_loads_with_no_reason() -> None:
    """A real pre-change artifact (tlspec v7, no reason field) loads, reason None."""

    import warnings

    fixture = Path(__file__).parent / "fixtures" / "tlspec_v7" / "tiny_v7.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = tl.load(str(fixture))
    assert loaded.forward_peak_memory_reason is None
    assert isinstance(loaded.forward_peak_memory, tl.Bytes)


def test_state_without_the_field_restores_with_no_reason() -> None:
    """A pickled state that predates the field restores with reason None."""

    trace = _capture()
    state = trace.__getstate__()
    state.pop("forward_peak_memory_reason", None)
    restored = type(trace).__new__(type(trace))
    restored.__setstate__(state)
    assert restored.forward_peak_memory_reason is None
    assert restored.forward_peak_memory is None or isinstance(
        restored.forward_peak_memory, tl.Bytes
    )


@pytest.mark.parametrize(
    ("peak", "reason", "why"),
    [(None, 5, "not_a_string"), (tl.Bytes(4096), "forged", "reason_beside_a_value")],
)
def test_malformed_reason_refuses_at_load(tmp_path: Path, peak: Any, reason: Any, why: str) -> None:
    trace = _capture()
    trace.forward_peak_memory = peak
    trace.forward_peak_memory_reason = reason
    path = tmp_path / f"forged_{why}.tlspec"
    tl.save(trace, path)

    with pytest.raises(TorchLensIOError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "artifact_forward_peak_memory_invalid"
    assert excinfo.value.fields["reason"] == why
