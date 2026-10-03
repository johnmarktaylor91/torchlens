"""R36 capture-lifecycle memory hygiene: the CPU-provable halves.

The runtime CUDA halves (fence integrity under contention, allocator
interference, ledger split on a multi-GPU box) are on the GPU-box plan; these
tests pin the host-side contracts that do not need a device.
"""

import pytest
import torch

import torchlens  # noqa: F401 -- full package init before submodule imports


def test_cpu_async_pending_events_drain_and_clear() -> None:
    """The cpu_async fence registry drains fully and tolerates odd devices (R36-1)."""

    from torchlens.utils.tensor_utils import (
        _CPU_ASYNC_PENDING_EVENTS,
        synchronize_pending_cpu_async_copies,
    )

    assert _CPU_ASYNC_PENDING_EVENTS == []
    synchronize_pending_cpu_async_copies()  # idempotent when empty

    # A non-CUDA accelerator entry falls back to the device synchronize and
    # never crashes the drain even when the backend module has no synchronize.
    _CPU_ASYNC_PENDING_EVENTS.append(torch.device("meta"))
    synchronize_pending_cpu_async_copies()
    assert _CPU_ASYNC_PENDING_EVENTS == []


def test_failed_fence_preserves_the_unfenced_tail_and_retries() -> None:
    """A mid-drain fence failure must not lose the copies behind it (R36-1).

    The drain used to clear the pending list BEFORE fencing, so any
    non-TypeError synchronize failure permanently dropped every remaining
    entry and the retry returned at the empty-list guard -- a silent D2H
    loss. The failing entry and the tail must stay pending for retry.
    """

    from torchlens.utils import tensor_utils as tu

    class _FakeEvent:
        def __init__(self) -> None:
            self.failures_left = 1
            self.synchronized = False

        def synchronize(self) -> None:
            if self.failures_left:
                self.failures_left -= 1
                raise RuntimeError("device fell off the bus")
            self.synchronized = True

    class _HealthyEvent:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            self.synchronized = True

    failing = _FakeEvent()
    healthy = _HealthyEvent()
    assert tu._CPU_ASYNC_PENDING_EVENTS == []
    try:
        tu._CPU_ASYNC_PENDING_EVENTS.extend([failing, healthy])
        with pytest.raises(RuntimeError, match="fell off the bus"):
            tu.synchronize_pending_cpu_async_copies()
        # Both the failing entry and the never-reached tail stay pending.
        assert [failing, healthy] == tu._CPU_ASYNC_PENDING_EVENTS
        # The retry is a real drain, not a no-op: everything fences.
        tu.synchronize_pending_cpu_async_copies()
        assert tu._CPU_ASYNC_PENDING_EVENTS == []
        assert failing.synchronized
        assert healthy.synchronized
    finally:
        tu._CPU_ASYNC_PENDING_EVENTS.clear()


def test_capture_touched_cuda_predicate_gates_on_trace_fact() -> None:
    """The empty_cache gate keys on the capture's backend fact (R36-3)."""

    from types import SimpleNamespace

    from torchlens.utils.tensor_utils import capture_touched_cuda

    assert capture_touched_cuda(SimpleNamespace(forward_memory_backend="cpu")) is False
    assert capture_touched_cuda(SimpleNamespace(forward_memory_backend="mps")) is False
    assert capture_touched_cuda(SimpleNamespace(forward_memory_backend="cuda")) is True
    # Unknown/missing fails toward the historical flush, never toward skipping.
    assert capture_touched_cuda(SimpleNamespace(forward_memory_backend="unknown")) is True
    assert capture_touched_cuda(SimpleNamespace()) is True
    # Other KNOWN accelerator homes must not flush the CUDA allocator (the
    # old blanket `not in ("cpu", "mps")` over-broadly flushed for them).
    assert capture_touched_cuda(SimpleNamespace(forward_memory_backend="xpu")) is False
    assert capture_touched_cuda(SimpleNamespace(forward_memory_backend="hpu")) is False


def test_capture_touched_cuda_consults_recorded_op_devices() -> None:
    """A CPU-homed capture that moved tensors to CUDA in forward flushes (R36).

    The stamped backend fact comes from the MODEL device, so a CPU-homed
    model that moves tensors to CUDA inside ``forward`` read as "cpu" and the
    allocator cache was never returned (false negative). A recorded cuda op
    device must flip the verdict; a scan failure keeps the stamped verdict.
    """

    from types import SimpleNamespace

    from torchlens.utils.tensor_utils import capture_touched_cuda

    cpu_op = SimpleNamespace(device_ref="cpu")
    cuda_op = SimpleNamespace(device_ref="cuda:0")
    assert (
        capture_touched_cuda(SimpleNamespace(forward_memory_backend="cpu", ops=[cpu_op, cuda_op]))
        is True
    )
    assert (
        capture_touched_cuda(SimpleNamespace(forward_memory_backend="cpu", ops=[cpu_op])) is False
    )

    class _RaisingOps:
        def __iter__(self):
            raise RuntimeError("husked")

    # Consult failure keeps the stamped fact's verdict (adds flushes only).
    assert (
        capture_touched_cuda(SimpleNamespace(forward_memory_backend="cpu", ops=_RaisingOps()))
        is False
    )


def test_failed_drain_restores_unfenced_tail_and_retry_fences_it() -> None:
    """A synchronize failure mid-drain never loses the unfenced tail (R36-1).

    Regression (b6 S HIGH + O MED, round 5): the drain cleared the pending
    registry BEFORE fencing, so a failing event dropped every remaining fence
    and the retry drain no-opped at the empty-list guard -- un-fenced
    ``non_blocking=True`` D2H copies could then never be fenced and a host
    read could observe partial bytes with no diagnostic.
    """

    from torchlens.utils import tensor_utils as tu

    class _FlakyEvent:
        def __init__(self) -> None:
            self.fail = True
            self.synced = 0

        def synchronize(self) -> None:
            if self.fail:
                raise RuntimeError("device fell off the bus")
            self.synced += 1

    class _HealthyEvent:
        def __init__(self) -> None:
            self.synced = 0

        def synchronize(self) -> None:
            self.synced += 1

    flaky = _FlakyEvent()
    healthy = _HealthyEvent()
    assert tu._CPU_ASYNC_PENDING_EVENTS == []
    try:
        tu._CPU_ASYNC_PENDING_EVENTS.extend([flaky, healthy])
        with pytest.raises(RuntimeError, match="fell off the bus"):
            tu.synchronize_pending_cpu_async_copies()
        # The failing entry AND the never-reached tail are both still pending.
        assert [flaky, healthy] == tu._CPU_ASYNC_PENDING_EVENTS
        assert healthy.synced == 0
        # The retry drain is NOT a no-op: it fences the restored tail.
        flaky.fail = False
        tu.synchronize_pending_cpu_async_copies()
        assert tu._CPU_ASYNC_PENDING_EVENTS == []
        assert flaky.synced == 1
        assert healthy.synced == 1
    finally:
        tu._CPU_ASYNC_PENDING_EVENTS.clear()


def test_failed_drain_keeps_tail_ahead_of_copies_recorded_since() -> None:
    """Restored unfenced entries precede fences recorded after the failure (R36-1)."""

    from torchlens.utils import tensor_utils as tu

    class _Event:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        def synchronize(self) -> None:
            if self.fail:
                raise RuntimeError("boom")

    first = _Event(fail=True)
    assert tu._CPU_ASYNC_PENDING_EVENTS == []
    try:
        tu._CPU_ASYNC_PENDING_EVENTS.append(first)
        with pytest.raises(RuntimeError):
            tu.synchronize_pending_cpu_async_copies()
        later = _Event()
        tu._CPU_ASYNC_PENDING_EVENTS.append(later)
        assert [first, later] == tu._CPU_ASYNC_PENDING_EVENTS
    finally:
        tu._CPU_ASYNC_PENDING_EVENTS.clear()


def test_capture_touched_cuda_keys_on_recorded_op_devices() -> None:
    """The empty_cache gate keys on observed op devices, both directions (R36 b6 pair).

    False-negative half (fable): a CPU-homed model that moves tensors to CUDA
    inside ``forward`` stamps backend ``"cpu"`` but records CUDA-deviced ops --
    the gate must flush. Over-broad half (opus): a non-CUDA accelerator capture
    whose backend fact reads ``"unknown"`` but whose recorded ops are provably
    non-CUDA must NOT flush the caller's CUDA allocator.
    """

    from types import SimpleNamespace

    from torchlens.ir.refs import DeviceRef
    from torchlens.utils.tensor_utils import capture_touched_cuda

    def _trace(backend: str, device_names: list[str]) -> object:
        ops = [SimpleNamespace(device_ref=DeviceRef.from_value(name)) for name in device_names]

        class _FakeTrace:
            forward_memory_backend = backend

            def __iter__(self):
                return iter(ops)

        return _FakeTrace()

    # False-negative half: cpu-stamped capture with a CUDA op -> flush.
    assert capture_touched_cuda(_trace("cpu", ["cpu", "cuda:0", "cpu"])) is True
    # Over-broad half: unknown-stamped capture, all ops provably non-CUDA -> no flush.
    assert capture_touched_cuda(_trace("unknown", ["xpu:0", "xpu:0"])) is False
    # Completed scans on plain cpu captures stay False; cuda fact stays True.
    assert capture_touched_cuda(_trace("cpu", ["cpu"])) is False
    assert capture_touched_cuda(_trace("cuda", [])) is True
    # Zero scannable ops + unknown fact fails toward the historical flush.
    assert capture_touched_cuda(_trace("unknown", [])) is True

    # Ops WITHOUT device facts are no evidence: fall back to the stamped
    # backend rule (unknown -> flush), never a false "provably non-CUDA".
    class _NoRefTrace:
        forward_memory_backend = "unknown"

        def __iter__(self):
            return iter([SimpleNamespace()])

    assert capture_touched_cuda(_NoRefTrace()) is True


def test_pure_view_probes_consume_no_global_rng() -> None:
    """The lazy-import _pure_view probes never draw from the user's generator (B8-7)."""

    from torchlens.utils._callable_safety import _pure_view

    torch.manual_seed(1234)
    before = torch.get_rng_state()
    _pure_view("T")
    _pure_view("real")
    _pure_view("data")
    assert torch.equal(torch.get_rng_state(), before), (
        "_pure_view probes perturbed the global default generator"
    )


def test_forward_peak_bracket_never_resets_cuda_peak_counter() -> None:
    """No capture-lifecycle code path calls reset_peak_memory_stats (R36-2)."""

    import pathlib

    root = pathlib.Path(torchlens.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "reset_peak_memory_stats(" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == [], (
        "reset_peak_memory_stats clobbers the caller's process-wide peak "
        f"counter; snapshot instead (R36-2). Offenders: {offenders}"
    )


@pytest.mark.smoke
def test_mps_style_no_arg_synchronize_does_not_typeerror() -> None:
    """A backend synchronize that takes no device argument drains cleanly (R36).

    ``torch.mps.synchronize()`` takes no device argument, so the drain's
    ``sync(entry)`` raised TypeError -- and on the failure-scrub arms that
    TypeError MASKED the original capture exception with a drain traceback.
    """

    from torchlens.utils import tensor_utils as tu

    calls = {"noarg": 0}

    class _NoArgSyncModule:
        @staticmethod
        def synchronize() -> None:
            calls["noarg"] += 1

    saved_torch_attr = tu.torch_attr
    try:
        tu.torch_attr = lambda name: _NoArgSyncModule if name == "meta" else saved_torch_attr(name)
        tu._CPU_ASYNC_PENDING_EVENTS.append(torch.device("meta"))
        tu.synchronize_pending_cpu_async_copies()  # must not raise
    finally:
        tu.torch_attr = saved_torch_attr
        tu._CPU_ASYNC_PENDING_EVENTS.clear()
    assert calls["noarg"] == 1


@pytest.mark.smoke
def test_cpu_async_pending_events_are_bounded() -> None:
    """The fence-event registry is hard-bounded, never unbounded cross-capture (R36).

    A capture that never reaches a drain seam must not grow the pending list
    without limit; crossing the bound drains inline (a fence -- correctness
    neutral).
    """

    from torchlens.utils import tensor_utils as tu

    assert tu._CPU_ASYNC_PENDING_EVENTS == []
    try:
        for _ in range(tu._CPU_ASYNC_PENDING_EVENTS_MAX + 10):
            tu._record_cpu_async_copy_event(torch.device("meta"))
        assert len(tu._CPU_ASYNC_PENDING_EVENTS) <= tu._CPU_ASYNC_PENDING_EVENTS_MAX
    finally:
        tu._CPU_ASYNC_PENDING_EVENTS.clear()


def test_cleanup_does_not_flush_cuda_for_cpu_capture() -> None:
    """cleanup() must not flush the CUDA allocator for a CPU-only capture (R36-3).

    This was the THIRD empty_cache site; the backend teardown and postprocess
    step-13 sites were already gated on capture_touched_cuda. A CPU-only trace
    cleaned up inside a GPU training loop flushed the caller's allocator.
    """

    import torch.nn as nn

    import torchlens as tl
    from torchlens.data_classes import cleanup as cleanup_module

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4)), torch.randn(2, 4))
    assert getattr(trace, "forward_memory_backend", None) in ("cpu", "mps")

    calls = {"flush": 0}
    saved_available = cleanup_module._is_cuda_available
    saved_flush = torch.cuda.empty_cache
    try:
        cleanup_module._is_cuda_available = lambda: True
        torch.cuda.empty_cache = lambda: calls.__setitem__("flush", calls["flush"] + 1)
        trace.cleanup()
    finally:
        cleanup_module._is_cuda_available = saved_available
        torch.cuda.empty_cache = saved_flush
    assert calls["flush"] == 0
