"""measure.2: CPU peaks are opt-in, and a default capture never writes the host mark.

JMT's walk, measure.2 (b), 28 Oct 2026: "Make it opt-in; report CPU peaks as
unavailable otherwise." Measuring a CPU peak resets the process-wide resident
high-water mark (``VmHWM``), which also resets the caller's own ``ru_maxrss``
and cannot be undone. ``CaptureOptions(measure_host_peak_memory=True)`` opts in;
by default nothing is written and the CPU forward and backward peaks read
``None`` with :data:`CPU_PEAK_NOT_REQUESTED_REASON`.

The untouched-mark test is armed by its own positive control: the same reading
drops under the opt-in, so the detector sees a reset when one happens.
"""

from __future__ import annotations

import os
import resource
from pathlib import Path
from typing import Any

import pytest
import torch
from _fresh_plant import PLANT_BYTES, touch_fresh_plant
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens.capture.peak_memory import CPU_PEAK_NOT_REQUESTED_REASON
from torchlens.observe._peaks import pass_peak_facts
from torchlens.options import CaptureOptions

MARGIN = 2**20
HOST = CaptureOptions(measure_host_peak_memory=True)

procfs = pytest.mark.skipif(
    not (Path("/proc/self/status").exists() and os.access("/proc/self/clear_refs", os.W_OK)),
    reason="the resident high-water reset is Linux procfs only (skipped by design)",
)


def _vm_hwm() -> int:
    """The raw kernel ``VmHWM`` in bytes (no lag correction: the mark itself)."""

    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise AssertionError("no VmHWM line")


def _ru_maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class _Planting(nn.Module):
    """Writes and frees a fresh 64 MiB mapping inside its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        touch_fresh_plant().close()
        return self.fc(x)


def _capture_with_backward(capture: CaptureOptions | None = None, plant: bool = False) -> Any:
    """A CPU capture plus one logged backward, optionally planting in both passes."""

    torch.manual_seed(0)
    model = (_Planting() if plant else nn.Linear(4, 4)).train()
    trace = tl.trace(model, torch.randn(2, 4), capture=capture)
    out = trace[trace.output_layers[0]].out
    if plant:
        out.register_hook(lambda grad: touch_fresh_plant().close())
    trace.backward(out.sum())
    return trace


def _raise_mark_above_rss() -> None:
    """Leave the process high-water mark a full plant above the current RSS."""

    touch_fresh_plant().close()


def test_option_defaults_off_and_is_mirrored_on_the_trace() -> None:
    assert CaptureOptions().measure_host_peak_memory is False
    trace = _capture_with_backward(HOST)
    try:
        assert trace.measure_host_peak_memory is True
    finally:
        trace.cleanup()


@pytest.mark.smoke
@procfs
def test_default_capture_leaves_the_process_high_water_mark_untouched() -> None:
    _raise_mark_above_rss()
    hwm_before, maxrss_before = _vm_hwm(), _ru_maxrss()
    trace = _capture_with_backward()
    try:
        # Without a reset the mark only ever rises; a reset drops it to the
        # current RSS, a full plant below the mark raised above.
        assert _vm_hwm() >= hwm_before
        assert _ru_maxrss() >= maxrss_before
        assert trace.forward_memory_backend == "cpu"
        assert trace.forward_peak_memory is None
        assert trace.forward_peak_memory_reason == CPU_PEAK_NOT_REQUESTED_REASON
        assert trace.backward_memory_backend == "cpu"
        assert trace.backward_peak_memory is None
        assert trace.backward_peak_memory_reason == CPU_PEAK_NOT_REQUESTED_REASON
        assert "measure_host_peak_memory=True" in CPU_PEAK_NOT_REQUESTED_REASON
        pair = trace.forward_peak_memory_pair
        assert pair is not None
        assert (pair["resident"], pair["resident_basis"]) == (None, "not_requested")
        for which in ("forward", "backward"):
            facts = pass_peak_facts(trace, which)
            assert facts.value_bytes is None
            assert facts.meaning == CPU_PEAK_NOT_REQUESTED_REASON
    finally:
        trace.cleanup()


@procfs
def test_the_untouched_mark_detector_sees_the_opt_in_reset() -> None:
    """Positive control: under the opt-in the same reading drops by about a plant."""

    _raise_mark_above_rss()
    hwm_before = _vm_hwm()
    trace = _capture_with_backward(HOST)
    try:
        assert _vm_hwm() < hwm_before - PLANT_BYTES // 2
    finally:
        trace.cleanup()


def test_default_capture_gets_no_number_for_a_planted_and_freed_buffer() -> None:
    """ARMED NEGATIVE: a 64 MiB plant in both passes still reads unavailable."""

    trace = _capture_with_backward(plant=True)
    try:
        assert trace.forward_peak_memory is None
        assert trace.forward_peak_memory_reason == CPU_PEAK_NOT_REQUESTED_REASON
        assert trace.backward_peak_memory is None
        assert trace.backward_peak_memory_reason == CPU_PEAK_NOT_REQUESTED_REASON
    finally:
        trace.cleanup()


@procfs
def test_opt_in_measures_the_planted_buffer_as_before() -> None:
    """The arm of the negative: the opt-in reports the plant in both passes."""

    trace = _capture_with_backward(HOST, plant=True)
    try:
        assert int(trace.forward_peak_memory) >= PLANT_BYTES - MARGIN
        assert trace.forward_peak_memory_reason is None
        assert int(trace.backward_peak_memory) >= PLANT_BYTES - MARGIN
        assert trace.backward_peak_memory_reason is None
        assert trace.forward_peak_memory_pair["resident_basis"] == "per_capture"
    finally:
        trace.cleanup()


def test_backward_reason_round_trips_and_the_knob_does_not(tmp_path: Path) -> None:
    trace = _capture_with_backward(HOST)
    trace.backward_peak_memory = None
    trace.backward_peak_memory_reason = CPU_PEAK_NOT_REQUESTED_REASON
    path = tmp_path / "backward_none.tlspec"
    tl.save(trace, path)
    loaded = tl.load(path)
    assert loaded.backward_peak_memory is None
    assert loaded.backward_peak_memory_reason == CPU_PEAK_NOT_REQUESTED_REASON
    assert loaded.measure_host_peak_memory is False  # a DROP session knob


def test_state_without_the_backward_reason_restores_none() -> None:
    """A state that predates the field (v9-shaped) default-fills the reason."""

    trace = _capture_with_backward(HOST)
    state = trace.__getstate__()
    state.pop("backward_peak_memory_reason", None)
    state.pop("measure_host_peak_memory", None)
    restored = type(trace).__new__(type(trace))
    restored.__setstate__(state)
    assert restored.backward_peak_memory_reason is None
    assert restored.measure_host_peak_memory is False


@pytest.mark.parametrize(
    ("peak", "reason", "why"),
    [(None, 5, "not_a_string"), (tl.Bytes(4096), "forged", "reason_beside_a_value")],
)
def test_malformed_backward_reason_refuses_at_load(
    tmp_path: Path, peak: Any, reason: Any, why: str
) -> None:
    trace = _capture_with_backward(HOST)
    trace.backward_peak_memory = peak
    trace.backward_peak_memory_reason = reason
    path = tmp_path / f"forged_{why}.tlspec"
    tl.save(trace, path)

    with pytest.raises(TorchLensIOError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "artifact_backward_peak_memory_invalid"
    assert excinfo.value.fields["reason"] == why
