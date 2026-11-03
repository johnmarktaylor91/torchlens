"""Builtin sidecar families exist in a fresh process, whatever was imported first.

The reserved ``torchlens.boundary_facts`` family's contract lives in
``torchlens.bundle._fed_by`` and registers at that module's import. The
sidecar door loads it on first consultation, so a fresh interpreter that
never imports the owner still lists, attaches and reads the family (the
order dependence first seen as the ``reference/harness_bundles.md`` doc
blocks failing alone). Each case runs in its own interpreter: in-process
state from earlier tests would hide the bug.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torchlens as tl

_OWNER = "torchlens.bundle._fed_by"
_FAMILY = "torchlens.boundary_facts"


def _run_fresh(source: str) -> str:
    """Run ``source`` in a fresh interpreter on this checkout; return stdout."""

    repo_root = Path(tl.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_fresh_process_attaches_and_reads_the_builtin_family() -> None:
    """Attach-first in a fresh process round-trips a validated descriptor."""

    out = _run_fresh(
        f"""
        import sys
        import torch
        import torchlens as tl
        from torchlens.io import AnalysisOnlySidecar
        trace = tl.trace(torch.nn.Linear(4, 2), torch.randn(1, 4))
        assert {_OWNER!r} not in sys.modules
        descriptor = {{"stream": "input", "batch_row": 0}}
        tl.io.attach_sidecar(trace, {_FAMILY!r}, descriptor)
        back = tl.io.read_sidecar(trace, {_FAMILY!r})
        print(isinstance(back, AnalysisOnlySidecar), back == descriptor)
        """
    )
    assert out.split() == ["False", "True"]
