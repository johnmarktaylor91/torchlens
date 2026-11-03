"""A dead trace is freed when its last reference goes, not by a later capture or gen-2 pass.

Three holders used to outlive a capture. Per-op closures in the emission path were
self-recursive, so each one sat in a reference cycle that kept the trace alive until
the cyclic collector's next full pass. The completeness patch wrappers installed on
``torch.Tensor`` and ``torch`` for the first capture in a process are copied into
torch's own import-time registries (sharded-op tables, dynamo's attribute map and rule
cache), and their closure state kept that first trace alive for the life of the
process. A predicate recording kept one context per op for its lifetime although its
backward reads only the selected ones.
"""

from __future__ import annotations

import gc
import inspect
import subprocess
import sys
import textwrap
import weakref
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.rng.monitor import host_nondeterminism_monitor
from torchlens.options import CaptureOptions
from torchlens.recording import dry_run


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.norm = nn.LayerNorm(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.fc(x)) + x)


class _Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.blocks[0](x)
        first, second = x.split(3, dim=-1)
        return self.head(torch.cat([second, first], dim=-1))


class _Chain(nn.Module):
    """``n`` elementwise ops in one module: a long forward with a tiny structure."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n):
            x = x + 1.0 if i % 2 else x * 1.0
        return x


#: Tracked objects a dead replayable capture may leave behind with the collector off.
#: The three fixed holders left about 7,000 (replayable) and 1,100 (plain) behind.
_DEAD_CAPTURE_OBJECT_FLOOR = 400


def _replayable_capture() -> Any:
    torch.manual_seed(0)
    return tl.trace(_Stack(), torch.randn(2, 6), capture=CaptureOptions(replayable=True))


def _death_by_refcount() -> tuple[bool, bool, int]:
    """Capture, drop the trace with the collector off, and report what is left.

    Returns ``(trace_dead, core_dead, tracked_object_delta)``.
    """

    gc.collect()
    gc.disable()
    try:
        before = len(gc.get_objects())
        trace = _replayable_capture()
        trace_ref = weakref.ref(trace)
        core_ref = weakref.ref(trace._trace_core)  # noqa: SLF001
        del trace
        trace_dead = trace_ref() is None
        core_dead = core_ref() is None
        delta = len(gc.get_objects()) - before
    finally:
        gc.enable()
    return trace_dead, core_dead, delta


def test_rng_monitor_window_is_freed_by_refcount() -> None:
    """A torn-down host RNG monitor dies with its last reference, not at the next collection.

    The replayable capture's monitor window used to leave a cycle (monitor -> profile
    hooks -> closures -> monitor, plus the owner-sync session) holding its per-window
    caches until a gen-2 collection; the monitor now holds its hooks weakly and drops the
    session at teardown.
    """

    model = _Stack().eval()
    x = torch.randn(2, 6)
    _replayable_capture()  # a warm process: lazy torch and torchlens imports are done
    monitor = host_nondeterminism_monitor(model)
    with monitor:
        model(x)
    ref = weakref.ref(monitor)
    gc.collect()
    gc.disable()
    try:
        del monitor
        assert ref() is None, "the monitor window survived its last reference (held by a cycle)"
    finally:
        gc.enable()


def test_dead_trace_is_freed_by_refcount() -> None:
    """``del trace`` frees the trace and its core without the cyclic collector."""

    _replayable_capture()  # not the first capture: that case is the subprocess test
    trace_dead, core_dead, delta = _death_by_refcount()
    assert trace_dead, "the trace survived its last reference (held by a cycle)"
    assert core_dead, "the trace core survived its last reference"
    assert delta < _DEAD_CAPTURE_OBJECT_FLOOR, f"{delta} tracked objects left by a dead capture"


_FIRST_CAPTURE_SCRIPT = textwrap.dedent(
    """
    import gc, sys, weakref
    import torch
    from torch import nn
    import torchlens as tl
    from torchlens.options import CaptureOptions
    from torchlens._data_classes.trace import Trace
    from torchlens._trace_core.core import TraceCore
    {models}

    torch.manual_seed(0)
    trace = tl.trace(_Stack(), torch.randn(2, 6), capture=CaptureOptions(replayable=True))
    trace_ref = weakref.ref(trace)
    core_ref = weakref.ref(trace._trace_core)
    del trace
    gc.collect()
    dead = trace_ref() is None and core_ref() is None
    live = sum(isinstance(obj, (Trace, TraceCore)) for obj in gc.get_objects())
    print("RESULT", int(dead), live)
    """
)


@pytest.mark.heavy
def test_first_capture_in_a_process_dies_after_collect() -> None:
    """The first replayable capture of a process is freed by ``gc.collect()``.

    Runs in a fresh interpreter: the holder is what torch's own registries copy
    from ``torch.Tensor`` while the first capture has its patches installed.
    """

    models = inspect.getsource(_Block) + "\n" + inspect.getsource(_Stack)
    code = _FIRST_CAPTURE_SCRIPT.format(models=models)
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=240
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT"))
    _, dead, live = line.split()
    assert dead == "1", "the first capture's trace survived gc.collect()"
    assert live == "0", f"{live} TorchLens trace objects alive after the capture died"


def _recording_state(recording: Any) -> Any:
    return object.__getattribute__(recording, "_recording_state")


def _every_hundredth_op(ctx: Any) -> bool:
    return ctx.kind == "op" and ctx.step_index is not None and ctx.step_index % 100 == 0


def test_recording_keeps_only_the_contexts_backward_reads() -> None:
    """A long recording retains the selected op contexts, not one per op."""

    from torchlens.backends.torch.backward import _selected_fastlog_forward_labels

    n_ops = 1000
    recording = tl.record(_Chain(n_ops), torch.randn(2, 3), save=_every_hundredth_op)
    state = _recording_state(recording)
    op_contexts = [ctx for ctx in state.all_contexts if ctx.kind == "op"]
    assert op_contexts, "the selected op contexts must stay available to backward"
    assert all(_every_hundredth_op(ctx) for ctx in op_contexts)
    assert len(state.all_contexts) < n_ops // 10, len(state.all_contexts)
    assert not state.pass_lookup.contexts_by_raw_label
    labels = _selected_fastlog_forward_labels(recording)
    assert all(ctx.label in labels for ctx in op_contexts)


def test_dry_run_still_lists_every_context() -> None:
    """A dry run's deliverable is the full context list, so it keeps every op."""

    n_ops = 200
    result = dry_run(_Chain(n_ops), torch.randn(2, 3), save=_every_hundredth_op)
    assert sum(ctx.kind == "op" for ctx in result.contexts) == n_ops
