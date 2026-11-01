"""Fork retention: calibrated measurement + regression ceilings.

The COW fork shares tensor payloads and sealed columns but is NOT near-free
in objects: measured ~61 gc-tracked objects (~13.5 KB of small allocations)
per op on a small conv/BN/relu stack -- ~68% of a steady-state capture's
tracked-object retention (disclosed in ``_trace_fork`` and ``Trace.fork``).

MEASUREMENT-BUG GUARDS (a prior probe over-counted retention 20x): the census
must first prove itself on a no-op (delta 0) and a known allocation counted
exactly. Two CPython traps make uncalibrated counts meaningless:
``gc.get_objects()`` sees only TRACKED objects (``object()`` instances never
appear), and a collection UNTRACKS atomic-only dicts/tuples (10k empty dicts
count as ~1 after a collect). Calibration uses lists, which stay tracked.
"""

import gc

import pytest
import torch
from torch import nn

import torchlens as tl


class _Net(nn.Module):
    """Small conv-free block stack (~83 ops when traced)."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()) for _ in range(8)]
        )
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block stack and head."""

        return self.head(self.blocks(x))


def _live_count() -> int:
    """Return the gc-tracked object count after a full collection."""

    gc.collect()
    gc.collect()
    return len(gc.get_objects())


# slow, not smoke: the gc.get_objects() census scales with accumulated session
# state -- 3.5s isolated but 97.6s charged min(wall, cpu) in the merged full
# not-slow session, past even the heavy ceiling.
@pytest.mark.slow
def test_fork_retention_measured_with_calibration_guards() -> None:
    """Fork object retention stays below a fresh capture (calibrated census)."""

    # Guard 1: a no-op census delta must be zero. Ambient churn (an unrelated
    # object dying between the two counts under a busy full-suite session,
    # round-3 gate: delta -1) is retried away; a genuinely mis-calibrated
    # census fails every attempt and still trips.
    for _ in range(5):
        base = _live_count()
        if _live_count() - base == 0:
            break
    else:
        pytest.fail("census never stabilized to a zero no-op delta")

    # Guard 2: a known tracked allocation must be counted exactly (same
    # bounded-retry discipline for ambient churn; a real over/under-count is
    # deterministic and fails all attempts).
    for _ in range(5):
        base = _live_count()
        hold = [[i] for i in range(10_000)]
        counted = _live_count() - base
        del hold
        if counted == 10_001:
            break
    else:
        pytest.fail(f"census mis-calibrated: {counted} != 10001")

    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(8, 32)
    # Steady-state baseline: the first capture pays one-time wrapping costs.
    warm = tl.trace(model, x, save=tl.func("relu"))
    warm.cleanup()
    del warm

    base = _live_count()
    trace = tl.trace(model, x, save=tl.func("relu"))
    capture_delta = _live_count() - base
    n_ops = len(trace._layer_list)
    assert n_ops > 50

    base = _live_count()
    fork = trace.fork()
    fork_delta = _live_count() - base

    # Regression ceilings, generous over the measured ~61/op so platform
    # variance never flakes: a fork must retain FEWER objects than a fresh
    # capture, and an accidental return to object-graph deep-copying
    # (~1000+/op, the deleted forkcopier regime) must trip loudly.
    assert fork_delta < capture_delta, (
        f"fork retained {fork_delta} objects >= fresh capture {capture_delta}"
    )
    assert fork_delta / n_ops < 200, (
        f"fork retention blew past the disclosed regime: {fork_delta / n_ops:.0f}/op"
    )
    del fork, trace
