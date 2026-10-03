"""Observe item 15: track_device_memory CPU-side core (fake-provider legs).

Never-reset counters, signed deltas, high-water ADVANCE semantics, the
None+status grammar replacing every fabricated zero, and the attempted-call
OOM row. GPU cells (real allocator, OOM attribution, pass-observation parity
on device) run on the shared real-GPU lane (C-OBS); nothing here publishes a
rate or claim.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.observe._device_memory as device_memory
from torchlens.observe._device_memory import (
    DeviceMemoryReading,
    DeviceMemorySample,
    device_memory_samples,
    flat_field_projection,
)


class _ScriptedProvider:
    """Fake provider replaying scripted counter readings per device."""

    def __init__(self, script: dict[str, list[DeviceMemoryReading]]) -> None:
        self.script = {key: list(readings) for key, readings in script.items()}

    def device_keys(self) -> tuple[str, ...]:
        """Return the scripted device keys."""

        return tuple(self.script)

    def read(self, device_key: str) -> DeviceMemoryReading:
        """Pop the next scripted reading (repeating the last one)."""

        readings = self.script[device_key]
        return readings.pop(0) if len(readings) > 1 else readings[0]


def _capture_with_provider(monkeypatch, provider, model: nn.Module, x: torch.Tensor):
    """Capture with track_device_memory=True and the injected provider."""

    monkeypatch.setattr(device_memory, "_provider_for", lambda trace: provider)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(track_device_memory=True))


def _reading(allocated: int, peak: int, reserved: int = 4096, peak_reserved: int = 8192):
    """Shorthand scripted reading."""

    return DeviceMemoryReading(
        allocated=allocated, reserved=reserved, peak_allocated=peak, peak_reserved=peak_reserved
    )


def test_default_off_and_swept_fields_are_none_not_zero() -> None:
    """Fabricated zeros are gone: unmeasured flat fields read None."""

    assert tl.options.CaptureOptions().track_device_memory is False
    captured = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    try:
        for op in captured.layer_list:
            assert op.bytes_delta_at_call is None, op.layer_label
            assert op.bytes_peak_at_call is None, op.layer_label
        assert device_memory_samples(captured) == {}
    finally:
        captured.cleanup()


def test_signed_deltas_and_peak_advance_semantics(monkeypatch) -> None:
    """Negative deltas are valid; peak_advance 0 is 'did not exceed high water'."""

    # One device that grows and advances its peak; one that shrinks under an
    # old (never-advancing) high-water mark.
    readings_growing = [_reading(1000 + 64 * step, 1000 + 64 * step) for step in range(64)]
    readings_shrinking = [_reading(5000 - 16 * step, 999_999) for step in range(64)]
    provider = _ScriptedProvider({"fake:0": readings_growing, "fake:1": readings_shrinking})
    captured = _capture_with_provider(
        monkeypatch, provider, nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4)
    )
    try:
        samples = device_memory_samples(captured)
        assert samples, "sampling was armed and a provider was available"
        measured = [
            sample for rows in samples.values() for sample in rows if sample.status == "measured"
        ]
        growing = [s for s in measured if s.device == "fake:0"]
        shrinking = [s for s in measured if s.device == "fake:1"]
        assert all(s.allocated_delta > 0 and s.peak_advance > 0 for s in growing)
        assert all(s.allocated_delta < 0 for s in shrinking)
        # The old high-water never moves: advance is exactly 0, which is NOT
        # "no allocation" (the shrinking device kept de/re-allocating).
        assert all(s.peak_advance == 0 for s in shrinking)
    finally:
        captured.cleanup()


def test_flat_fields_are_projections_of_the_typed_samples(monkeypatch) -> None:
    """bytes_delta/peak_at_call become compatibility projections when sampled."""

    readings = [_reading(1000 + 32 * step, 1000 + 32 * step) for step in range(64)]
    provider = _ScriptedProvider({"fake:0": readings})
    captured = _capture_with_provider(
        monkeypatch, provider, nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4)
    )
    try:
        relu = captured["relu_1_2"]
        assert relu.bytes_delta_at_call is not None
        assert int(relu.bytes_delta_at_call) > 0
        assert int(relu.bytes_peak_at_call) > 0
    finally:
        captured.cleanup()


def test_projection_peak_is_none_without_an_advance() -> None:
    """A high-water that did not move is not this call's peak."""

    no_advance = (
        DeviceMemorySample(
            device="fake:0",
            status="measured",
            allocated_before=100,
            allocated_after=90,
            allocated_delta=-10,
            peak_before=500,
            peak_after=500,
            peak_advance=0,
        ),
    )
    delta, peak = flat_field_projection(no_advance)
    assert delta == -10
    assert peak is None
    assert flat_field_projection(None) == (None, None)
    assert flat_field_projection((DeviceMemorySample(device="fake:0", status="not_measured"),)) == (
        None,
        None,
    )


def test_unsupported_devices_produce_typed_absence(monkeypatch) -> None:
    """No samplable device -> no samples, never zeros (CPU is 'unsupported')."""

    class _EmptyProvider:
        def device_keys(self) -> tuple[str, ...]:
            return ()

        def read(self, device_key: str) -> DeviceMemoryReading:
            raise AssertionError("never called")

    captured = _capture_with_provider(
        monkeypatch, _EmptyProvider(), nn.Linear(4, 4), torch.randn(2, 4)
    )
    try:
        assert device_memory_samples(captured) == {}
        for op in captured.layer_list:
            assert op.bytes_delta_at_call is None
    finally:
        captured.cleanup()


@pytest.mark.smoke
def test_provider_never_resets_counters(monkeypatch) -> None:
    """The no-reset law (R36-2): sampling reads and only reads."""

    calls: list[str] = []

    class _AuditingProvider:
        def device_keys(self) -> tuple[str, ...]:
            calls.append("device_keys")
            return ("fake:0",)

        def read(self, device_key: str) -> DeviceMemoryReading:
            calls.append("read")
            return _reading(1000, 1000)

    captured = _capture_with_provider(
        monkeypatch, _AuditingProvider(), nn.Linear(4, 4), torch.randn(2, 4)
    )
    try:
        assert set(calls) == {"device_keys", "read"}
    finally:
        captured.cleanup()


def test_pass_peak_observation_independent_of_sampling(monkeypatch) -> None:
    """Sampling never touches the pass-level peak machinery.

    The BYTE-IDENTICAL on/off comparison is a CUDA-lane cell (C-OBS): the CPU
    figure is a non-reproducible RSS delta by its own contract, so equality
    here would pin noise. The CPU leg pins the invariants sampling CAN break:
    the recorded backend is unchanged and the fake provider observed only
    reads (the no-reset audit below).
    """

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
    x = torch.randn(2, 8)
    plain = tl.trace(model, x)
    provider = _ScriptedProvider({"fake:0": [_reading(1000, 1000)]})
    sampled = _capture_with_provider(monkeypatch, provider, model, x)
    try:
        assert plain.forward_memory_backend == sampled.forward_memory_backend
        assert type(plain.forward_peak_memory) is type(sampled.forward_peak_memory)
    finally:
        plain.cleanup()
        sampled.cleanup()


@pytest.mark.smoke
def test_oom_settles_an_attempted_call_row(monkeypatch) -> None:
    """A CUDA OOM inside a bracketed op lands an exception-side attempted row.

    The OOM raises BEFORE the op commits, so the last COMMITTED op is the
    wrong suspect: the attempted call is named, the committed predecessor is
    asserted NOT to be.
    """

    provider = _ScriptedProvider(
        {"fake:0": [_reading(1000 + step, 1000 + step) for step in range(16)]}
    )
    monkeypatch.setattr(device_memory, "_provider_for", lambda trace: provider)

    class _OomOnSigmoid(torch.Tensor):
        """Subclass whose sigmoid dispatch raises OOM INSIDE the wrapped call."""

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if getattr(func, "__name__", "") == "sigmoid":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (fixture)")
            return super().__torch_function__(func, types, args, kwargs or {})

    class _OomModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = torch.tanh(x)  # a committed op precedes the OOM
            return torch.sigmoid(y.as_subclass(_OomOnSigmoid))

    with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
        tl.trace(
            _OomModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(track_device_memory=True),
        )
    partial = tl.partial.from_failed_capture(excinfo.value)
    samples = device_memory_samples(partial.trace)
    attempted = [label for label in samples if label.startswith("attempted:")]
    assert attempted, f"no attempted-call row settled: {sorted(samples)}"
    attempted_row = samples[attempted[0]][0]
    assert attempted_row.status == "raised_cuda_oom"
    # No invented requested-byte count rides the row.
    assert not hasattr(attempted_row, "requested_bytes")
    # The committed predecessor is NOT named as the OOM suspect.
    for label, rows in samples.items():
        if label.startswith("tanh"):
            assert all(row.status != "raised_cuda_oom" for row in rows)
