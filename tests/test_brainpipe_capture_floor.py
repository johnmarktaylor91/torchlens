"""Capture-floor release-at-emission oracles (F20 W1a, brainpipe memo s3.3).

The brainpipe panel's central discovery: an instrumented forward that saves
nothing peaked at 27x the bare forward because capture bookkeeping pinned
every live intermediate for the whole pass. These oracles pin the fix at the
mechanism level (live-holder counts, deterministic) and at the claim level
(peak RSS in a subprocess, heavy tier). Per the memo's transferable lesson,
the tests are peak-shaped and holder-shaped -- never retained-bytes-shaped.
"""

from __future__ import annotations

import contextlib
import gc
import json
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

import torchlens as tl

N_CONVS = 12
ACT_NUMEL = 2 * 16 * 8 * 8


class _ConvStack(nn.Module):
    """Uniform conv/relu stack whose activations all share one size."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(16, 16, 3, padding=1) for _ in range(N_CONVS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x.mean()


def _count_live_activation_tensors() -> int:
    """Count gc-visible activation-sized 4-D tensors after a collect."""

    gc.collect()
    return sum(
        1
        for obj in gc.get_objects()
        if isinstance(obj, torch.Tensor) and obj.numel() == ACT_NUMEL and obj.dim() == 4
    )


class _MidForwardCounter(nn.Module):
    """Wrap the stack and count live activations near the end of the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.stack = _ConvStack()
        self.mid_forward_live: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.stack.convs):
            x = torch.relu(conv(x))
            if i == N_CONVS - 2:
                self.mid_forward_live = _count_live_activation_tensors()
        return x.mean()


@pytest.mark.smoke
def test_save_all_releases_live_sources_at_emission() -> None:
    """Retention holds copies; bookkeeping must not also pin live sources.

    At op ~2k of 2N (k convs + k relus done), a save-all inference capture
    may legitimately hold ~2k retained copies plus a bounded working set.
    Before the W1a fix the identity cache and the module-arg stash pinned
    every LIVE intermediate too (~4k live tensors, unreclaimable by gc).
    """

    model = _MidForwardCounter().eval()
    x = torch.randn(2, 16, 8, 8)
    tl.trace(model, x, capture=tl.options.CaptureOptions(inference_only=True))
    assert model.mid_forward_live is not None
    ops_done = 2 * (N_CONVS - 1)
    # Retained copies for completed ops, plus a small live working set
    # (current output, the counter's own view, input) and slack for
    # transiently alive frame locals. The pre-fix count exceeded
    # ops_done * 2 - a full second population of pinned live sources.
    ceiling = ops_done + 8
    assert model.mid_forward_live <= ceiling, (
        f"{model.mid_forward_live} live activation tensors mid-forward "
        f"(ceiling {ceiling}): capture bookkeeping is pinning live sources "
        "again (F20 W1a regression -- identity cache or module-arg stash)"
    )


@pytest.mark.smoke
def test_selective_save_holds_no_live_population_mid_forward() -> None:
    """A one-site selection must not pin the whole live activation set.

    The escrow may hold detached COPIES within its RAM budget (they spill to
    disk above it), but live forward intermediates must free at emission.
    """

    model = _MidForwardCounter().eval()
    x = torch.randn(2, 16, 8, 8)
    tl.trace(
        model,
        x,
        save=tl.func("mean"),
        capture=tl.options.CaptureOptions(inference_only=True),
    )
    assert model.mid_forward_live is not None
    ops_done = 2 * (N_CONVS - 1)
    # Escrow copies (bounded by its RAM budget; tiny here) plus working set.
    # The defect shape held ~2x ops_done: every live source AND every module
    # input pinned until finalize.
    ceiling = ops_done + 8
    assert model.mid_forward_live <= ceiling


@pytest.mark.smoke
def test_module_forward_arg_stash_carries_stubs_not_payloads() -> None:
    """The module-arg stash and enter events carry payload-free stubs.

    GC-11 nulls ``ModuleCall.forward_args`` before the trace is returned, so
    payloads stashed at module entry are never user-visible; carrying them
    was pure floor. The summary string must be unchanged by the stubbing.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    call = log.module_calls["0:1"]
    assert call.forward_args is None  # GC-11 contract, unchanged
    assert call.forward_args_summary == "[Tensor(shape=(2, 4), dtype=float32)]"


_PEAK_SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    import torch, torch.nn as nn

    class Big(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList(
                [nn.Conv2d(64, 64, 3, padding=1) for _ in range(30)]
            )
        def forward(self, x):
            for c in self.convs:
                x = torch.relu(c(x))
            return x.mean()

    mode = sys.argv[1]
    model = Big().eval()
    x = torch.randn(8, 64, 64, 64)  # ~8 MB per activation, 60 ops
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if mode == "bare":
        with torch.no_grad():
            model(x)
    else:
        import torchlens as tl
        tl.trace(
            model,
            x,
            save=tl.func("mean"),
            capture=tl.options.CaptureOptions(inference_only=True),
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"delta_mb": peak - base}))
    """
)


def _subprocess_peak_mb(mode: str) -> float:
    """Run one measurement mode in a clean subprocess and return its peak delta."""

    result = subprocess.run(
        [sys.executable, "-c", _PEAK_SCRIPT, mode],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return float(json.loads(result.stdout.strip().splitlines()[-1])["delta_mb"])


@pytest.mark.heavy
def test_selective_capture_peak_is_bounded_multiple_of_bare() -> None:
    """Peak oracle: one-site capture peak stays a small multiple of bare.

    Pre-fix this configuration peaked at ~6x the bare forward (332 MB vs
    55 MB: full-model module-arg pins plus escrow). Post-fix it is bare +
    escrow RAM budget (64 MiB) + slack. The ceiling of 3.5x holds a wide
    margin against allocator noise while staying far below the defect shape.
    """

    bare = _subprocess_peak_mb("bare")
    selective = _subprocess_peak_mb("selective")
    assert selective <= bare * 3.5 + 96, (
        f"selective-capture peak {selective:.0f} MB vs bare {bare:.0f} MB: "
        "the capture floor is unbounded again (F20 W1a regression)"
    )


@pytest.mark.smoke
def test_structure_only_pins_are_trace_scoped_not_process_leaks() -> None:
    """W1b narrowing (F20): structure_only's real-tensor pins die with the log.

    A structure-only capture retains ~one REAL activation-sized tensor per
    module call for the TRACE's lifetime (the W1b floor's persistent term;
    the holder is not reachable through __dict__/slots walks of the trace).
    This oracle pins the boundary that holds today: deleting the trace
    releases every one -- a trace-lifetime pin, never a process leak. The
    capture-time fix (release at emission like W1a) is the remainder.
    """

    import weakref

    model = _ConvStack().eval()
    x = torch.randn(2, 16, 8, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(structure_only=True))
    gc.collect()
    refs = []
    for obj in gc.get_objects():
        if (
            isinstance(obj, torch.Tensor)
            and obj.numel() == ACT_NUMEL
            and obj.dim() == 4
            and obj.device.type != "meta"
            and obj is not x
        ):
            with contextlib.suppress(TypeError):
                refs.append(weakref.ref(obj))
    del obj
    del log
    gc.collect()
    survivors = sum(1 for ref in refs if ref() is not None)
    assert survivors == 0, (
        f"{survivors} real activation-sized tensors outlive a deleted "
        "structure-only trace (W1b regression: trace-lifetime pin became a leak)"
    )
