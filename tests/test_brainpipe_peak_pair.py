"""Peak-pair instrumentation oracles (F20, brainpipe memo D-7 + build 3).

The measured peak is a PAIR (live allocation, resident high-water) with the
backend named -- CUDA and CPU instrumentation measure different physical
quantities. The sweep-scale instrument defect (the shipped peak property
reads 0 for every capture after the first in a process) is fixed on Linux
by scoping the host high-water mark per capture; the oracle here runs
multiple captures in ONE process and requires a non-degenerate reading on
the later ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

import torchlens as tl


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))


@pytest.mark.smoke
def test_peak_pair_shape_and_backend_label() -> None:
    """The pair carries live/resident/backend/resident_basis, backend named."""

    log = tl.trace(
        _model(),
        torch.randn(32, 64),
        capture=tl.options.CaptureOptions(inference_only=True),
    )
    pair = log.forward_peak_memory_pair
    assert pair is not None
    assert set(pair) == {"live", "resident", "backend", "resident_basis"}
    assert pair["backend"] in {"cpu:maxlive+rss", "mps:allocated+rss", "cuda:allocated+reserved"}
    # On the default CPU path the Python-allocation live peak is UNMEASURED
    # (the tracemalloc opt-in costs 1.7x-2.5x capture time): None, never a
    # fabricated zero.
    if pair["backend"] == "cpu:maxlive+rss":
        assert pair["live"] is None


@pytest.mark.smoke
def test_peak_pair_live_populates_under_the_tracemalloc_opt_in() -> None:
    """measure_python_peak_memory=True buys a positive live peak on CPU."""

    log = tl.trace(
        _model(),
        torch.randn(32, 64),
        capture=tl.options.CaptureOptions(inference_only=True, measure_python_peak_memory=True),
    )
    pair = log.forward_peak_memory_pair
    assert pair is not None
    assert pair["live"] is not None and pair["live"] > 0


@pytest.mark.smoke
def test_peak_pair_is_session_time_only() -> None:
    """The pair never survives save/load: loaded artifacts never re-measure."""

    import tempfile
    from pathlib import Path

    log = tl.trace(_model(), torch.randn(4, 64))
    assert log.forward_peak_memory_pair is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.tlspec"
        tl.save(log, path)
        loaded = tl.load(path)
    assert loaded.forward_peak_memory_pair is None


@pytest.mark.smoke
@pytest.mark.skipif(sys.platform != "linux", reason="VmHWM reset is procfs-only")
def test_multi_capture_resident_peak_is_non_degenerate() -> None:
    """Sweep-scale instrument oracle: capture N's resident peak is real.

    Pre-fix, the host high-water mark was the process-lifetime maximum, so
    every capture after the first read 0 -- useless for policing a
    hundred-capture sweep (brainpipe memo D-12 finding).

    The pre-forward RSS baseline is read via ``psutil`` (``process_rss_bytes``),
    an optional dependency deliberately absent from the lean per-PR smoke
    install (``.[dev,tabular,viz]`` in ``tests.yml``) while present in the
    nightly "Coverage floor" job's ``.[dev,test,tabular]`` install -- both are
    real, intentional CI environments, not a worker artifact. This oracle
    branches on the capability via ``psutil_available`` (never silently
    skips): where psutil is importable it proves the sweep-scale fix measures
    a real positive resident delta; where it is not, it proves the typed
    "unavailable" disclosure fires instead of a basis/value mismatch (the
    pre-fix bug the brainpipe memo's instrument defect would otherwise hide
    behind).
    """

    from torchlens.capture.peak_memory import psutil_available

    # Escalating widths: each capture retains strictly more than anything
    # the process allocated before, so the LAST capture's resident growth
    # cannot be served from cached allocator blocks (a warm allocator
    # legitimately reads 0 growth for a repeat-sized capture, which is
    # honest -- the defect under guard is the process-lifetime BASIS).
    readings = []
    for width in (128, 512, 1536):
        model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        x = torch.randn(64, width)
        log = tl.trace(model, x, capture=tl.options.CaptureOptions(inference_only=True))
        pair = log.forward_peak_memory_pair
        assert pair is not None
        readings.append((pair["resident"], pair["resident_basis"]))
    if psutil_available():
        assert all(basis == "per_capture" for _, basis in readings)
        assert readings[-1][0] is not None and readings[-1][0] > 0
    else:
        assert all(basis == "unavailable" for _, basis in readings)
        assert all(resident is None for resident, _ in readings)


_INFERENCE_ONLY_SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    import torch, torch.nn as nn
    import torchlens as tl

    class Big(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList(
                [nn.Conv2d(64, 64, 3, padding=1) for _ in range(20)]
            )
        def forward(self, x):
            for c in self.convs:
                x = torch.relu(c(x))
            return x.mean()

    mode = sys.argv[1]
    model = Big().eval()
    x = torch.randn(8, 64, 64, 64)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    opts = tl.options.CaptureOptions(inference_only=(mode == "inference"))
    tl.trace(model, x, capture=opts)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"delta_mb": peak - base}))
    """
)


@pytest.mark.heavy
def test_inference_only_never_costs_peak_memory() -> None:
    """D-13 claim gate, re-measured post-W1a: inference_only never costs peak.

    MEASUREMENT NOTE (F20): the memo's 2.9x figure (1891.6 -> 650.9 MB on
    real ResNet-50 batch 8) was taken on the PRE-W1a floor, where capture
    bookkeeping pinned every live intermediate and the autograd graph
    multiplied that pin. With the W1a release-at-emission fix in both modes,
    the host-RSS gap on this synthetic stack collapses to ~2% -- the floor
    fix subsumed most of what D-13 bought on CPU. The surviving claim gated
    here: the flag is never WORSE (the CUDA allocated-bytes win from
    dropping the autograd graph is real but cluster-gated per memo s5 --
    no CPU number ships as a GPU number).
    """

    def run(mode: str) -> float:
        result = subprocess.run(
            [sys.executable, "-c", _INFERENCE_ONLY_SCRIPT, mode],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        return float(json.loads(result.stdout.strip().splitlines()[-1])["delta_mb"])

    grad_peak = run("grad")
    inference_peak = run("inference")
    assert inference_peak <= grad_peak * 1.05 + 32, (
        f"inference_only peak {inference_peak:.0f} MB vs grad-mode "
        f"{grad_peak:.0f} MB: the flag now COSTS peak memory (regression)"
    )
