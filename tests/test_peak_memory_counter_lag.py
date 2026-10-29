"""The CPU peaks count a freed buffer when the kernel's high-water mark lags the exact RSS.

Linux keeps the RSS counters per CPU. Where ``/proc`` reports the exact sum
(``statm``, ``VmRSS``), the kernel still sets the ``VmHWM`` mark at unmap time
from the approximate total (``/proc/self/stat``), which leaves out each CPU's
unfolded residue. These tests build a known residue (44 to 63 pages in each of
the three RSS counters on every allowed CPU, left after that counter folds),
pin the measurement to one CPU, plant a fresh 64 MiB mapping and free it. The expected value is
the test's own exact RSS growth around the plant (``statm``), never a TorchLens
output. Before the lag correction the reported peak was the residue short
(several MiB on a 32-CPU host). Hosts where the residue cannot be built (an
approximate ``/proc`` RSS, or too few CPUs) skip by design. The captures opt in
with ``measure_host_peak_memory=True`` (measure.2: a default CPU capture
measures no peak).
"""

from __future__ import annotations

import mmap
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from _fresh_plant import PLANT_BYTES, touch_fresh_plant
from torch import nn

import torchlens as tl
from torchlens.capture import peak_memory
from torchlens.options import CaptureOptions

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
MARGIN = 2**20
#: Lag needed before the measurement: the margin plus room for the pinned
#: CPU's own counter moving during the plant (under 64 pages per counter).
MIN_LAG = MARGIN + 2**19
#: Pages left in each per-CPU counter after it folds (fault-around maps up to
#: 16 file pages per fault, so stopping at 44 stays below the 64-page batch).
RESIDUE_PAGES = 44
REGION_PAGES = 256
SAMPLES: dict[str, int] = {}
HOST = CaptureOptions(measure_host_peak_memory=True)

linux_procfs = pytest.mark.skipif(
    not (
        Path("/proc/self/stat").exists()
        and Path("/proc/self/statm").exists()
        and os.access("/proc/self/clear_refs", os.W_OK)
        and hasattr(os, "sched_setaffinity")
        and hasattr(mmap, "MAP_PRIVATE")
    ),
    reason="the per-CPU RSS counter lag is a Linux procfs effect (skipped by design)",
)


def _exact_pages() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1])


def _approximate_pages() -> int:
    with open("/proc/self/stat") as handle:
        text = handle.read()
    return int(text[text.rindex(")") + 2 :].split()[21])


def _lag_pages() -> int:
    return _exact_pages() - _approximate_pages()


def _rss() -> int:
    return _exact_pages() * PAGE


def _regions(held: list[Any]) -> list[tuple[mmap.mmap, bool]]:
    """One fresh region per kernel RSS counter (anonymous, shared memory, file), with writability."""

    anon = mmap.mmap(-1, REGION_PAGES * PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    shmem = mmap.mmap(-1, REGION_PAGES * PAGE, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
    with tempfile.TemporaryFile() as backing:  # the mapping keeps its own reference
        backing.write(bytes(REGION_PAGES * PAGE))
        backing.flush()
        file_pages = mmap.mmap(backing.fileno(), REGION_PAGES * PAGE, access=mmap.ACCESS_READ)
    held.extend([anon, shmem, file_pages])
    return [(anon, True), (shmem, True), (file_pages, False)]


def _touch(region: mmap.mmap, page: int, writable: bool) -> None:
    if writable:
        region[page * PAGE] = 1
    else:
        region[page * PAGE]  # a read fault maps the cached file page


def _leave_residue(region: mmap.mmap, writable: bool) -> None:
    """Fold this CPU's counter for ``region``'s kind, then leave 44 to 63 pages in it."""

    page = 0
    previous = _lag_pages()
    while page < REGION_PAGES // 2:  # touch until this counter folds (the lag drops)
        _touch(region, page, writable)
        page += 1
        lag = _lag_pages()
        if lag < previous:
            break
        previous = lag
    folded = _lag_pages()
    while page < REGION_PAGES and _lag_pages() - folded < RESIDUE_PAGES:
        _touch(region, page, writable)
        page += 1


class _Lagged:
    """Leave a counter residue on every allowed CPU, then pin to the first; restore on exit."""

    def __enter__(self) -> _Lagged:
        self.home = os.sched_getaffinity(0)
        self.held: list[Any] = []
        cpus = sorted(self.home)
        try:
            for cpu in [*cpus[1:], cpus[0]]:
                os.sched_setaffinity(0, {cpu})
                for region, writable in _regions(self.held):
                    _leave_residue(region, writable)
            self.lag = _lag_pages() * PAGE
        except BaseException:
            self.__exit__()
            raise
        if self.lag < MIN_LAG:
            self.__exit__()
            pytest.skip(f"per-CPU RSS counter lag reached only {self.lag} bytes on this host")
        return self

    def __exit__(self, *exc: Any) -> None:
        os.sched_setaffinity(0, self.home)
        for item in reversed(self.held):
            item.close()
        self.held.clear()


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


@linux_procfs
def test_cpu_forward_peak_counts_a_freed_buffer_under_the_counter_lag() -> None:
    torch.manual_seed(0)
    tl.trace(Transient().eval(), torch.randn(2, 4)).cleanup()  # warm the wrappers
    model = Transient().eval()
    SAMPLES.clear()
    with _Lagged() as lagged:
        trace = tl.trace(model, torch.randn(2, 4), capture=HOST)
    try:
        outside = SAMPLES["alive"] - SAMPLES["entry"]
        assert outside >= PLANT_BYTES - MARGIN
        assert lagged.lag >= MIN_LAG
        assert trace.forward_peak_memory is not None
        assert int(trace.forward_peak_memory) >= outside - MARGIN
    finally:
        trace.cleanup()


def _planted_trace(samples: dict[str, int]) -> tuple[Any, torch.Tensor]:
    """A trace whose output gradient hook plants and frees the 64 MiB buffer."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Linear(4, 4).train(), torch.randn(2, 4), capture=HOST)
    out = trace[trace.output_layers[0]].out

    def plant(grad: torch.Tensor) -> None:
        samples["entry"] = _rss()
        buffer = touch_fresh_plant()
        samples["alive"] = _rss()
        buffer.close()

    out.register_hook(plant)
    return trace, out


@linux_procfs
def test_cpu_backward_peak_counts_a_freed_buffer_under_the_counter_lag() -> None:
    # Warm the backward path on another trace: the measured trace's
    # ``backward_peak_memory`` is the largest pass, so it gets only the lagged pass.
    warm, warm_out = _planted_trace({})
    warm.backward(warm_out.sum())
    warm.cleanup()
    samples: dict[str, int] = {}
    trace, out = _planted_trace(samples)
    try:
        with _Lagged() as lagged:
            trace.backward(out.sum())
        outside = samples["alive"] - samples["entry"]
        assert outside >= PLANT_BYTES - MARGIN
        assert lagged.lag >= MIN_LAG
        assert int(trace.backward_peak_memory) >= outside - MARGIN
    finally:
        trace.cleanup()


def _fake_proc(
    monkeypatch: pytest.MonkeyPatch, hwm_kib: int, rss_kib: int, stat_pages: int
) -> None:
    import builtins
    import io

    real_open = builtins.open
    status = f"Name:\tpython\nVmHWM:\t{hwm_kib} kB\nVmRSS:\t{rss_kib} kB\n"
    stat = "1 (py thon) S" + " 0" * 20 + f" {stat_pages}" + " 0" * 20 + "\n"

    def fake_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path) == "/proc/self/status":
            return io.StringIO(status)
        if str(path) == "/proc/self/stat":
            return io.StringIO(stat)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_peak_rss_adds_the_lag_only_when_the_mark_is_above_the_current_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mark is the current RSS: no transient above it, nothing added.
    _fake_proc(monkeypatch, hwm_kib=4000, rss_kib=4000, stat_pages=100)
    assert peak_memory.peak_rss_bytes() == 4000 * 1024
    # The mark came from the kernel's approximate counter: add exact minus approximate.
    _fake_proc(monkeypatch, hwm_kib=9000, rss_kib=4000, stat_pages=100)
    assert peak_memory.peak_rss_bytes() == 9000 * 1024 + (4000 * 1024 - 100 * PAGE)
    # An approximate total above the exact one is never a negative lag.
    _fake_proc(monkeypatch, hwm_kib=9000, rss_kib=4000, stat_pages=4000 * 1024 // PAGE + 50)
    assert peak_memory.peak_rss_bytes() == 9000 * 1024


def test_kernel_mark_lag_is_exact_minus_approximate_never_negative() -> None:
    assert peak_memory._kernel_mark_lag_bytes(10 * 2**20, 9 * 2**20) == 2**20
    assert peak_memory._kernel_mark_lag_bytes(9 * 2**20, 10 * 2**20) == 0
    assert peak_memory._kernel_mark_lag_bytes(9 * 2**20, None) == 0
