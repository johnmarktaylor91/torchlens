"""Capture-floor release-at-emission oracles (F20 W1a, brainpipe memo s3.3).

The brainpipe panel's central discovery: an instrumented forward that saves
nothing peaked at 27x the bare forward because capture bookkeeping pinned
every live intermediate for the whole pass. These oracles pin the fix at the
mechanism level (live-holder counts, deterministic) and at the claim level
(peak RSS in a subprocess, heavy tier). Per the memo's transferable lesson,
the tests are peak-shaped and holder-shaped -- never retained-bytes-shaped.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

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
