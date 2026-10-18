"""The CPU peaks count a freed buffer when the kernel's high-water mark lags the exact RSS.

Linux keeps the RSS counters per CPU. Where ``/proc`` reports the exact sum
(``statm``, ``VmRSS``), the kernel still sets the ``VmHWM`` mark at unmap time
from the approximate total (``/proc/self/stat``), which leaves out each CPU's
unfolded residue. These tests build a known residue (60 touched pages on every
CPU but one, each counted after that CPU's counter folds), pin the measurement
to the remaining CPU, plant a 64 MiB buffer and free it. The expected value is
the test's own exact RSS growth around the plant (``statm``), never a TorchLens
output. Before the lag correction the reported peak was the residue short
(several MiB on a 32-CPU host). Hosts where the residue cannot be built (an
approximate ``/proc`` RSS, or too few CPUs) skip by design.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture import peak_memory

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
PLANT = 8 * 2**20  # float64 elements: 64 MiB, above glibc's mmap ceiling
PLANT_BYTES = PLANT * 8
MARGIN = 2**20
#: The residue must exceed the margin twice over for old code to fail clearly.
MIN_LAG = 2 * MARGIN
RESIDUE_PAGES = 60  # below the per-CPU fold batch (64 pages at 32 CPUs)
SAMPLES: dict[str, int] = {}

linux_procfs = pytest.mark.skipif(
    not (
        Path("/proc/self/stat").exists()
        and Path("/proc/self/statm").exists()
        and os.access("/proc/self/clear_refs", os.W_OK)
        and hasattr(os, "sched_setaffinity")
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


def _lag_bytes() -> int:
    return (_exact_pages() - _approximate_pages()) * PAGE


def _rss() -> int:
    return _exact_pages() * PAGE


def _leave_residue(cpu: int, held: list[mmap.mmap]) -> None:
    """Leave ``RESIDUE_PAGES`` unfolded pages in ``cpu``'s RSS counter."""

    os.sched_setaffinity(0, {cpu})
    region = mmap.mmap(-1, (128 + RESIDUE_PAGES) * PAGE)
    held.append(region)
    page = 0
    previous = _lag_bytes()
    while page < 128:  # touch until this CPU's counter folds (the lag drops)
        region[page * PAGE] = 1
        page += 1
        lag = _lag_bytes()
        if lag < previous:
            break
        previous = lag
    for offset in range(RESIDUE_PAGES):
        region[(page + offset) * PAGE] = 1


class _Lagged:
    """Pin to one CPU with a residue on every other; restore on exit."""

    def __enter__(self) -> _Lagged:
        self.home = os.sched_getaffinity(0)
        self.held: list[mmap.mmap] = []
        cpus = sorted(self.home)
        try:
            for cpu in cpus[1:]:
                _leave_residue(cpu, self.held)
            os.sched_setaffinity(0, {cpus[0]})
            self.lag = _lag_bytes()
        except BaseException:
            self.__exit__()
            raise
        if self.lag < MIN_LAG:
            self.__exit__()
            pytest.skip(f"per-CPU RSS counter lag reached only {self.lag} bytes on this host")
        return self

    def __exit__(self, *exc: Any) -> None:
        os.sched_setaffinity(0, self.home)
        for region in self.held:
            region.close()
        self.held.clear()


class Transient(nn.Module):
    """Holds the 64 MiB host buffer during the forward, frees it before returning."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        SAMPLES["entry"] = _rss()
        buffer = np.ones(PLANT)
        SAMPLES["alive"] = _rss()
        del buffer
        return self.fc(x)


@linux_procfs
def test_cpu_forward_peak_counts_a_freed_buffer_under_the_counter_lag() -> None:
    torch.manual_seed(0)
    tl.trace(Transient().eval(), torch.randn(2, 4)).cleanup()  # warm the wrappers
    model = Transient().eval()
    SAMPLES.clear()
    with _Lagged() as lagged:
        trace = tl.trace(model, torch.randn(2, 4))
    try:
        outside = SAMPLES["alive"] - SAMPLES["entry"]
        assert outside >= PLANT_BYTES - MARGIN
        assert lagged.lag >= MIN_LAG
        assert trace.forward_peak_memory is not None
        assert int(trace.forward_peak_memory) >= outside - MARGIN
    finally:
        trace.cleanup()


@linux_procfs
def test_cpu_backward_peak_counts_a_freed_buffer_under_the_counter_lag() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4).train()
    trace = tl.trace(model, torch.randn(2, 4))
    out = trace[trace.output_layers[0]].out
    samples: dict[str, int] = {}

    def plant(grad: torch.Tensor) -> None:
        samples["entry"] = _rss()
        buffer = np.ones(PLANT)
        samples["alive"] = _rss()
        del buffer

    out.register_hook(plant)
    try:
        trace.backward(out.sum(), retain_graph=True)  # warm the backward path
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
