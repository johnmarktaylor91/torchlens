"""Device-memory sampling provider (observe item 15, wave-2 core).

Opt-in ``CaptureOptions(track_device_memory=True)`` samples device allocator
counters around each op call: allocated/reserved before/after with SIGNED
deltas plus the high-water ADVANCE -- never resetting the counters (repo law
R36-2: resets clobber the caller's process-wide peak; ``peak_advance`` is the
better quantity anyway -- monotone, composes with the pass bracket, and
mutation-free). The v1 provider is the PyTorch CUDA caching allocator,
factored behind ONE interface so record-tier sampling and other allocators
land later without a schema break.

Column semantics the consumers must keep straight: negative deltas are valid;
a zero ``peak_advance`` means only "did not exceed the process's earlier high
water" -- it is NOT an op-local transient peak and NOT "no allocation";
concurrent foreign allocator activity is a named confound. CPU/MPS and other
unsupported devices produce TYPED ABSENCE (status ``unsupported``), never a
fabricated zero. Nothing here publishes a rate or claim: the GPU cells run on
the shared real-GPU lane (C-OBS) before anything publishes. Every spelling is
DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "CudaAllocatorProvider",
    "DeviceMemoryProvider",
    "DeviceMemoryReading",
    "DeviceMemorySample",
    "device_memory_samples",
]

#: Closed status vocabulary: four different facts a single column would blur.
SAMPLE_STATUSES = ("measured", "not_measured", "unsupported", "raised_cuda_oom")


@dataclass(frozen=True)
class DeviceMemoryReading:
    """One point-in-time read of a device's allocator counters (bytes)."""

    allocated: int
    reserved: int
    peak_allocated: int
    peak_reserved: int


class DeviceMemoryProvider(Protocol):
    """The ONE allocator-read interface (reads only; resets are repo-illegal)."""

    def device_keys(self) -> tuple[str, ...]:
        """Return the device keys this provider can sample (may be empty)."""
        ...

    def read(self, device_key: str) -> DeviceMemoryReading:
        """Read the four counters for one device without mutating any state."""
        ...


class CudaAllocatorProvider:
    """PyTorch CUDA caching-allocator provider (initialized devices only).

    Reading counters for devices this process never initialized would
    allocate their CUDA contexts, so only live devices are sampled.
    """

    def device_keys(self) -> tuple[str, ...]:
        """Return ``cuda:N`` keys for initialized CUDA devices."""

        import torch

        from ..utils.tensor_utils import _is_cuda_initialized

        if not (_is_cuda_initialized() and torch.cuda.is_available()):
            return ()
        return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))

    def read(self, device_key: str) -> DeviceMemoryReading:
        """Read allocated/reserved/current-peak counters for one device."""

        import torch

        index = int(device_key.split(":", 1)[1])
        return DeviceMemoryReading(
            allocated=int(torch.cuda.memory_allocated(index)),
            reserved=int(torch.cuda.memory_reserved(index)),
            peak_allocated=int(torch.cuda.max_memory_allocated(index)),
            peak_reserved=int(torch.cuda.max_memory_reserved(index)),
        )


@dataclass(frozen=True)
class DeviceMemorySample:
    """One per-device sample bracketing one op call.

    Parameters
    ----------
    device:
        Provider device key (``"cuda:0"``).
    status:
        ``measured`` / ``not_measured`` (before/after read incomplete) /
        ``unsupported`` (device has no provider) / ``raised_cuda_oom`` (the
        bracketed call raised OOM; ``after`` facts are best-effort and no
        requested-byte count is invented).
    allocated_before:
        Allocated bytes before the call, when read.
    allocated_after:
        Allocated bytes after the call, when read.
    allocated_delta:
        Signed difference (negative deltas are valid).
    reserved_before:
        Reserved bytes before the call, when read.
    reserved_after:
        Reserved bytes after the call, when read.
    reserved_delta:
        Signed reserved difference.
    peak_before:
        Allocated high-water before the call.
    peak_after:
        Allocated high-water after the call.
    peak_advance:
        High-water increase across the call. ``0`` means only "did not
        exceed the process's earlier high water".
    reserved_peak_advance:
        Reserved high-water increase across the call.
    """

    device: str
    status: str
    allocated_before: int | None = None
    allocated_after: int | None = None
    allocated_delta: int | None = None
    reserved_before: int | None = None
    reserved_after: int | None = None
    reserved_delta: int | None = None
    peak_before: int | None = None
    peak_after: int | None = None
    peak_advance: int | None = None
    reserved_peak_advance: int | None = None


def _provider_for(trace: Any) -> DeviceMemoryProvider:
    """Return the active provider (test seam: ``trace._device_memory_provider``)."""

    override = trace.__dict__.get("_device_memory_provider")
    if override is not None:
        return override
    return CudaAllocatorProvider()


def read_before(trace: Any) -> dict[str, DeviceMemoryReading] | None:
    """Take the pre-call readings for one op bracket (never raises).

    Returns
    -------
    dict[str, DeviceMemoryReading] | None
        Per-device readings, or ``None`` when sampling is off or no device
        is samplable.
    """

    if not getattr(trace, "track_device_memory", False):
        return None
    try:
        provider = _provider_for(trace)
        keys = provider.device_keys()
        if not keys:
            return None
        return {key: provider.read(key) for key in keys}
    except Exception:  # noqa: BLE001 - sampling must never break a capture.
        return None


def settle_sample(
    trace: Any,
    raw_label: str,
    before: dict[str, DeviceMemoryReading] | None,
    *,
    oom: bool = False,
) -> None:
    """Settle one op bracket's samples onto the trace store (never raises).

    Parameters
    ----------
    trace:
        Active trace.
    raw_label:
        Raw op label (or the provisional attempted-call identity for OOM
        rows); consumers join to public labels through the item-1 resolver.
    before:
        Pre-call readings from :func:`read_before`, or ``None``.
    oom:
        Whether the bracketed call raised a CUDA OOM: the sample settles an
        EXCEPTION-SIDE attempted-call row (the OOM raises BEFORE the op
        commits, so the last committed op is the wrong suspect).
    """

    if not getattr(trace, "track_device_memory", False):
        return
    try:
        provider = _provider_for(trace)
        keys = provider.device_keys()
        samples: list[DeviceMemorySample] = []
        status = "raised_cuda_oom" if oom else "measured"
        for key in keys:
            before_reading = (before or {}).get(key)
            try:
                after_reading = provider.read(key)
            except Exception:  # noqa: BLE001 - a dead device reads as absence.
                after_reading = None
            if before_reading is None or after_reading is None:
                samples.append(
                    DeviceMemorySample(
                        device=key,
                        status="raised_cuda_oom" if oom else "not_measured",
                        allocated_after=(
                            after_reading.allocated if after_reading is not None else None
                        ),
                        peak_after=(
                            after_reading.peak_allocated if after_reading is not None else None
                        ),
                    )
                )
                continue
            samples.append(
                DeviceMemorySample(
                    device=key,
                    status=status,
                    allocated_before=before_reading.allocated,
                    allocated_after=after_reading.allocated,
                    allocated_delta=after_reading.allocated - before_reading.allocated,
                    reserved_before=before_reading.reserved,
                    reserved_after=after_reading.reserved,
                    reserved_delta=after_reading.reserved - before_reading.reserved,
                    peak_before=before_reading.peak_allocated,
                    peak_after=after_reading.peak_allocated,
                    peak_advance=after_reading.peak_allocated - before_reading.peak_allocated,
                    reserved_peak_advance=(
                        after_reading.peak_reserved - before_reading.peak_reserved
                    ),
                )
            )
        if samples:
            store = trace.__dict__.setdefault("_device_memory_samples", {})
            store[str(raw_label)] = tuple(samples)
    except Exception:  # noqa: BLE001 - sampling must never break a capture.
        return


def settle_op_bracket(
    trace: Any,
    func_name: str,
    func_call_id: int,
    before: dict[str, DeviceMemoryReading] | None,
    *,
    oom: bool = False,
) -> None:
    """Settle one wrapped-call bracket (the wrapper hot path's one-liner door).

    Clean exits settle under the provisional ``call:<id>`` identity (re-keyed
    to the committed raw label at the commit site); a CUDA OOM settles the
    exception-side ``attempted:`` row -- the OOM raised before the op
    committed, so the last committed op is the wrong suspect.
    """

    label = f"attempted:{func_name}:{func_call_id}" if oom else f"call:{func_call_id}"
    settle_sample(trace, label, before, oom=oom)


def device_memory_samples(trace: Any) -> dict[str, tuple[DeviceMemorySample, ...]]:
    """Return the per-call device-memory samples recorded on one live trace.

    Returns
    -------
    dict[str, tuple[DeviceMemorySample, ...]]
        Raw-label-keyed samples (the OOM attempted-call row rides the
        provisional identity of the call that raised). Empty when the option
        was off or nothing was samplable; loaded traces carry none (the
        store is session-time, field_intent declared live-only).
    """

    store = trace.__dict__.get("_device_memory_samples")
    return dict(store) if isinstance(store, dict) else {}


def flat_field_projection(
    samples: tuple[DeviceMemorySample, ...] | None,
) -> tuple[int | None, int | None]:
    """Project one call's samples onto the legacy flat op fields.

    ``bytes_delta_at_call`` is the single-device allocated delta;
    ``bytes_peak_at_call`` is the absolute high-water after the call ONLY
    when the call observed an advance, else ``None`` (a high-water that did
    not move is not this call's peak). Multi-device calls project the first
    measured device; the typed per-device samples stay authoritative.

    Parameters
    ----------
    samples:
        One call's samples, or ``None``.

    Returns
    -------
    tuple[int | None, int | None]
        ``(bytes_delta_at_call, bytes_peak_at_call)``.
    """

    if not samples:
        return None, None
    for sample in samples:
        if sample.status == "measured":
            peak = (
                sample.peak_after
                if sample.peak_advance is not None and sample.peak_advance > 0
                else None
            )
            return sample.allocated_delta, peak
    return None, None
