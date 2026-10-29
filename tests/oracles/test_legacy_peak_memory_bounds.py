"""Outside-measurement bounds for ``Trace.forward_peak_memory`` and ``backward_peak_memory``.

Each forward (or backward hook) plants a buffer of known size, writes every
page, and frees it before returning: a private anonymous 64 MiB mapping
(``_fresh_plant.touch_fresh_plant``), so its resident growth is its full size
whatever earlier tests in the process left in the heap. The test samples the
process's resident set itself (``/proc/self/statm``, ``/proc/self/status``), independent of TorchLens.
A peak must include that buffer even though it is gone when the bracket ends.

The margin, registered before these tests first ran, is 1 MiB on every CPU
bound: it covers resident-set movement between TorchLens's bracket baseline and
the test's own sample. The resident high-water reset is Linux procfs only; the
CPU bounds skip by design elsewhere, and the CUDA bounds skip without a GPU.
Every CPU capture opts in with ``measure_host_peak_memory=True`` (measure.2: a
default capture measures no CPU peak and never resets the mark).
"""

from __future__ import annotations

import os
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
import torch
from _fresh_plant import touch_fresh_plant
from torch import nn

import torchlens as tl
from torchlens.capture import peak_memory, trace as capture_trace
from torchlens.capture.peak_memory import CPU_ZERO_PEAK_REASON
from torchlens.options import CaptureOptions

PLANT = 8 * 2**20  # float64 elements
PLANT_BYTES = PLANT * 8  # 67,108,864
PY_PLANT_BYTES = 32 * 2**20  # 33,554,432
MARGIN = 2**20
HOST = CaptureOptions(measure_host_peak_memory=True)
#: In-forward samples live outside the module: a default trace restores
#: module attributes after the forward (pure observer).
SAMPLES: dict[str, int] = {}

procfs = pytest.mark.skipif(
    not (Path("/proc/self/statm").exists() and os.access("/proc/self/clear_refs", os.W_OK)),
    reason="the per-bracket resident high-water needs Linux procfs (skipped by design)",
)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _rss() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _hwm() -> int:
    """The resident high-water mark on the exact-RSS basis of ``_rss()``.

    Where ``/proc`` RSS is the exact per-CPU sum, the kernel sets the
    ``VmHWM`` mark from its approximate total (``/proc/self/stat``), so a mark
    above the current RSS is raised by that counter lag (exact minus
    approximate) to put both bounds on the same basis.
    """

    fields: dict[str, int] = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            key = line.split(":", 1)[0]
            if key in ("VmHWM", "VmRSS"):
                fields[key] = int(line.split()[1]) * 1024
    if "VmHWM" not in fields:
        raise AssertionError("no VmHWM line")
    hwm, rss = fields["VmHWM"], fields.get("VmRSS", 0)
    if hwm > rss:
        with open("/proc/self/stat") as handle:
            text = handle.read()
        approximate = int(text[text.rindex(")") + 2 :].split()[21]) * os.sysconf("SC_PAGE_SIZE")
        hwm += max(0, rss - approximate)
    return hwm


class Transient(nn.Module):
    """Holds the 64 MiB host buffer during the forward, frees it before returning."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SAMPLES["entry"] = _rss()
        buffer = touch_fresh_plant()
        SAMPLES["alive"] = _rss()
        buffer.close()
        return self.fc(x)


class PyTransient(nn.Module):
    """Holds a 32 MiB Python bytearray during the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SAMPLES["entry"] = tracemalloc.get_traced_memory()[0]
        buffer = bytearray(PY_PLANT_BYTES)
        SAMPLES["alive"] = tracemalloc.get_traced_memory()[0]
        del buffer
        return self.fc(x)


@procfs
def test_cpu_forward_peak_includes_a_buffer_freed_inside_the_forward() -> None:
    torch.manual_seed(0)
    tl.trace(Transient().eval(), torch.randn(2, 4)).cleanup()  # warm the wrappers
    model = Transient().eval()
    SAMPLES.clear()
    before = _rss()
    trace = tl.trace(model, torch.randn(2, 4), capture=HOST)
    high_water = _hwm()
    try:
        assert trace.forward_memory_backend == "cpu"
        outside = SAMPLES["alive"] - SAMPLES["entry"]
        assert outside >= PLANT_BYTES - MARGIN
        assert trace.forward_peak_memory is not None
        reported = int(trace.forward_peak_memory)
        assert reported >= outside - MARGIN
        assert reported >= PLANT_BYTES - MARGIN
        assert reported <= high_water - before + MARGIN
        assert trace.forward_peak_memory_reason is None
    finally:
        trace.cleanup()


def test_cpu_forward_python_peak_includes_a_freed_bytearray() -> None:
    torch.manual_seed(0)
    model = PyTransient().eval()
    SAMPLES.clear()
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        capture=CaptureOptions(measure_python_peak_memory=True, measure_host_peak_memory=True),
    )
    try:
        outside = SAMPLES["alive"] - SAMPLES["entry"]
        assert outside >= PY_PLANT_BYTES
        assert trace.forward_peak_memory is not None
        assert int(trace.forward_peak_memory) >= outside - MARGIN
        assert int(trace.forward_peak_memory) >= PY_PLANT_BYTES - MARGIN
    finally:
        trace.cleanup()


def test_cpu_forward_zero_reads_none_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    # An outside measurement of zero growth: every resident reading is the same.
    monkeypatch.setattr(capture_trace, "process_rss_bytes", lambda: 1_000_000)
    monkeypatch.setattr(peak_memory, "peak_rss_bytes", lambda: 1_000_000)
    torch.manual_seed(0)
    trace = tl.trace(nn.Linear(4, 2).eval(), torch.randn(2, 4), capture=HOST)
    try:
        assert trace.forward_memory_backend == "cpu"
        assert trace.forward_peak_memory is None
        assert trace.forward_peak_memory_reason == CPU_ZERO_PEAK_REASON
    finally:
        trace.cleanup()


def _backward_hook_trace(device: str, samples: dict[str, list[int]]) -> Any:
    torch.manual_seed(0)
    model = nn.Linear(4, 4).to(device).train()
    host = CaptureOptions(measure_host_peak_memory=device == "cpu")
    trace = tl.trace(model, torch.randn(2, 4, device=device), capture=host)
    out = trace[trace.output_layers[0]].out

    def plant(grad: torch.Tensor) -> None:
        if device == "cpu":
            samples["entry"].append(_rss())
            region = touch_fresh_plant()
            samples["alive"].append(_rss())
            region.close()
        else:
            buffer = torch.ones(2 * PLANT, device=device)  # float32: 64 MiB
            del buffer

    out.register_hook(plant)
    return trace, out


@procfs
def test_cpu_backward_peak_is_the_largest_pass_peak() -> None:
    samples: dict[str, list[int]] = {"entry": [], "alive": []}
    trace, out = _backward_hook_trace("cpu", samples)
    try:
        windows = []
        for retain in (True, False):
            before = _rss()
            trace.backward(out.sum(), retain_graph=retain)
            windows.append(_hwm() - before)
            outside = samples["alive"][-1] - samples["entry"][-1]
            assert outside >= PLANT_BYTES - MARGIN
            assert int(trace.backward_peak_memory) >= outside - MARGIN
            assert int(trace.backward_peak_memory) >= PLANT_BYTES - MARGIN
        assert trace.backward_memory_backend == "cpu"
        # The largest single-pass peak, never the sum of the two passes.
        assert int(trace.backward_peak_memory) <= max(windows) + MARGIN
    finally:
        trace.cleanup()


@cuda
def test_cuda_forward_peak_includes_a_freed_device_buffer() -> None:
    class DeviceTransient(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            buffer = torch.ones(2 * PLANT, device=x.device)  # float32: 64 MiB
            del buffer
            return self.fc(x)

    torch.manual_seed(0)
    model = DeviceTransient().cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    trace = tl.trace(model, torch.randn(2, 4, device="cuda"))
    try:
        assert trace.forward_memory_backend == "cuda"
        reported = int(trace.forward_peak_memory)
        assert reported >= PLANT_BYTES
        assert reported <= torch.cuda.max_memory_allocated()
    finally:
        trace.cleanup()


@cuda
def test_cuda_backward_peak_bounds_a_device_buffer_in_a_hook() -> None:
    samples: dict[str, list[int]] = {"entry": [], "alive": []}
    trace, out = _backward_hook_trace("cuda", samples)
    try:
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.max_memory_allocated()
        trace.backward(out.sum(), retain_graph=True)
        trace.backward(out.sum())
        assert trace.backward_memory_backend == "cuda"
        reported = int(trace.backward_peak_memory)
        assert reported >= PLANT_BYTES
        assert reported <= torch.cuda.max_memory_allocated() - baseline
    finally:
        trace.cleanup()
