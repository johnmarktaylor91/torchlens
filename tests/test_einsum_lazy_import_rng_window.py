"""Torch's lazy ``torch.backends.opt_einsum`` import never lands in the RNG monitor window.

``torch.functional.einsum`` imports ``torch.backends.opt_einsum`` inside the
function body, so the first ``torch.einsum`` of a process imported it while the
host-nondeterminism monitor was armed. The import machinery's frames (a
``sys.meta_path`` finder such as pytest's assertion rewriter) then became roots
of the monitor's frame-reachable inventory, and a large object graph behind the
finder exhausted the 1,000,000-node deep-inventory budget
(``deep_inventory_budget_exhausted``): the capture's runnable artifact was
ceilinged to UNVERIFIABLE depending on what else the process held (in the suite,
whenever a static-scan suite ran earlier on the same worker). The monitor now
warms that import before it arms, so no finder frame runs in-window.

Every case runs in a fresh subprocess: the bug fires only on the process's
first einsum.
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

_FIRST_EINSUM_BEHIND_A_HEAVY_FINDER = """
import sys

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

assert "torch.backends.opt_einsum" not in sys.modules, (
    "precondition broken: torch.backends.opt_einsum was imported before the first "
    "capture, so this child cannot exercise the lazy-import-in-window path"
)


class HeavyFinder:
    \"\"\"A meta-path finder holding an object graph larger than the walk budget.\"\"\"

    def __init__(self):
        self.payload = [object() for _ in range(1_100_000)]

    def find_spec(self, fullname, path=None, target=None):
        return None


sys.meta_path.insert(0, HeavyFinder())


class ThreeOperandEinsum(nn.Module):
    def forward(self, a, b, c):
        return torch.einsum("a,b,c->abc", a + 1, b + 2, c + 3)


inputs = tuple(torch.arange(2.0) for _ in range(3))
trace = tl.trace(
    ThreeOperandEinsum(),
    inputs,
    capture=CaptureOptions(
        intervention_ready=True, capture_container_structure=True, cache=False
    ),
)
runnable = trace._runnable
assert runnable.rng_monitor_uncertain is False, runnable.rng_monitor_uncertain_detail
print("OK")
"""


def _child_environment() -> dict[str, str]:
    """Return the subprocess environment with the repo importable."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    return environment


def test_first_einsum_behind_a_heavy_meta_path_finder_keeps_the_rng_window_certain() -> None:
    """A process's first einsum does not walk import-machinery frames in-window."""

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_FIRST_EINSUM_BEHIND_A_HEAVY_FINDER)],
        check=False,
        capture_output=True,
        text=True,
        env=_child_environment(),
        cwd=_REPO_ROOT,
        timeout=300,
    )
    assert completed.returncode == 0 and "OK" in completed.stdout, (
        f"fresh-process einsum capture failed ({completed.returncode}):\n"
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
