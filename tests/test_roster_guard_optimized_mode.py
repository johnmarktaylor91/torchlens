"""grind-r5 b7 R24: the roster contradiction guard must survive ``python -O``.

The plain-callable branch of the ``get_ignored_functions`` vs
``get_testing_overrides`` contradiction check was a bare ``assert`` +
``continue``: under ``-O`` the assert stripped, and a (fault-injected) future
torch listing an overridable function as ignored silently dropped it from the
3,350-entry wrapper roster -- an invisible capture gap on exactly the
torch-version drift the check exists to catch. The descriptor branch was
already a real raise; this pins BOTH branches under optimized execution.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_FAULT_INJECTION_SCRIPT = r"""
import torch
import torch.overrides as overrides

real_ignored = overrides.get_ignored_functions
real_testing = overrides.get_testing_overrides

def fault_ignored():
    contradicted = set(real_ignored())
    contradicted.add(torch.tensor)  # overridable func declared ignored: the drift
    return contradicted

def fault_testing():
    table = dict(real_testing())
    table.setdefault(torch.tensor, lambda *a, **k: None)
    return table

overrides.get_ignored_functions = fault_ignored
overrides.get_testing_overrides = fault_testing

try:
    # The roster builds at import time, so the guard fires on the import
    # itself when the contradiction is present.
    from torchlens import _constants as constants

    constants._get_torch_overridable_functions()
except RuntimeError as error:
    print(f"GUARD_FIRED: {error}")
else:
    print("GUARD_SILENT")
"""


def _run_fault_injection(optimized: bool) -> str:
    """Run the roster fault-injection probe, optionally under ``-O``."""

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.getcwd())
    argv = [sys.executable]
    if optimized:
        argv.append("-O")
    argv += ["-c", _FAULT_INJECTION_SCRIPT]
    completed = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout


@pytest.mark.heavy
def test_roster_contradiction_guard_fires_under_python_o() -> None:
    """The contradiction must raise even when asserts are stripped."""

    output = _run_fault_injection(optimized=True)
    assert "GUARD_FIRED" in output, (
        f"python -O silently dropped a contradicted roster entry: {output!r}"
    )


@pytest.mark.heavy
def test_roster_contradiction_guard_fires_in_normal_mode() -> None:
    """Sanity control: the guard fires identically without ``-O``."""

    output = _run_fault_injection(optimized=False)
    assert "GUARD_FIRED" in output
