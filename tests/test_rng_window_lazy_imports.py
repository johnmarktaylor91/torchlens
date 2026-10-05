"""No import runs while the RNG monitor window is armed on a process's first capture.

An import inside the host-nondeterminism monitor window runs the
``sys.meta_path`` finders' ``find_spec`` frames, and the monitor's frame-reachable
inventory walks those frames as roots. A large object graph behind a finder
(pytest's assertion rewriter after a static-scan suite, an IPython hook) then
exhausted the 1,000,000-node deep-inventory budget
(``deep_inventory_budget_exhausted``), so the capture's runnable artifact was
ceilinged to UNVERIFIABLE depending on what else the process had loaded. Two
first-capture lazy imports did this: torch's own ``torch.backends.opt_einsum``
(imported inside ``torch.functional.einsum`` on torch 2.7) and TorchLens's
observability session reader (resolved on the first wrapped op). The monitor now
warms both before it arms.

The fresh-subprocess cases record every ``find_spec`` call made while the monitor
is armed, so they catch the next lazy import of this class wherever it comes from,
without depending on the node cap.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_EINSUM_ID = (
    "tests/test_tlspec_runnable_producer.py::"
    "test_runnable_internal_einsum_identity_save_load_and_run_verified[3]"
)

_RECORDING_FINDER_MODULE = """
import sys


class RecordingFinder:
    \"\"\"Record every lookup made while the RNG monitor window is armed.\"\"\"

    def __init__(self):
        self.in_window = []

    def find_spec(self, fullname, path=None, target=None):
        rng = sys.modules.get("torchlens.utils.rng")
        if rng is not None and getattr(rng, "_ACTIVE_MONITOR", None) is not None:
            self.in_window.append(fullname)
        return None
"""

_FIRST_CAPTURE = """
import sys

sys.path.insert(0, sys.argv[1])
from recording_finder import RecordingFinder  # a finder in a real source file

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

finder = RecordingFinder()
sys.meta_path.insert(0, finder)


class ThreeOperandEinsum(nn.Module):
    def forward(self, a, b, c):
        return torch.einsum("a,b,c->abc", a + 1, b + 2, c + 3)


class LinearRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return torch.relu(self.linear(x))


case = sys.argv[2]
if case == "einsum":
    model, inputs = ThreeOperandEinsum(), tuple(torch.arange(2.0) for _ in range(3))
else:
    model, inputs = LinearRelu(), torch.randn(2, 3)
trace = tl.trace(
    model,
    inputs,
    capture=CaptureOptions(
        intervention_ready=True, capture_container_structure=True, cache=False
    ),
)
assert finder.in_window == [], f"imports inside the RNG window: {finder.in_window}"
runnable = trace._runnable
assert runnable.rng_monitor_uncertain is False, runnable.rng_monitor_uncertain_detail
print("OK")
"""


def _child_environment() -> dict[str, str]:
    """Return the subprocess environment with the repo importable."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    return environment


@pytest.mark.heavy
@pytest.mark.parametrize("case", ("einsum", "linear_relu"))
def test_first_capture_runs_no_import_inside_the_rng_window(tmp_path: Path, case: str) -> None:
    """A fresh process's first runnable-capable capture imports nothing in-window."""

    (tmp_path / "recording_finder.py").write_text(textwrap.dedent(_RECORDING_FINDER_MODULE))
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_FIRST_CAPTURE), str(tmp_path), case],
        check=False,
        capture_output=True,
        text=True,
        env=_child_environment(),
        cwd=_REPO_ROOT,
        timeout=300,
    )
    assert completed.returncode == 0 and "OK" in completed.stdout, (
        f"fresh-process first capture ({case}) failed ({completed.returncode}):\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


@pytest.mark.slow
def test_static_scan_suite_then_einsum_stays_verified() -> None:
    """The deterministic order that failed: a static-scan suite, then einsum[3]."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_private_probe_gate.py",
            _EINSUM_ID,
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "-q",
            "--tb=short",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_child_environment(),
        cwd=_REPO_ROOT,
        timeout=600,
    )
    assert completed.returncode == 0, (
        f"static-scan suite then einsum failed ({completed.returncode}):\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n--- stderr ---\n{completed.stderr[-2000:]}"
    )
