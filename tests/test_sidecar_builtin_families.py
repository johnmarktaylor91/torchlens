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


def test_import_torchlens_does_not_load_the_family_owner() -> None:
    """``import torchlens`` stays free of the bundle family module."""

    out = _run_fresh(
        f"""
        import sys
        import torchlens
        print({_OWNER!r} in sys.modules)
        """
    )
    assert out.split() == ["False"]


def test_fresh_process_lists_the_builtin_family() -> None:
    """The public list door reports the builtin family without the owner imported."""

    out = _run_fresh(
        f"""
        import sys
        import torchlens as tl
        assert {_OWNER!r} not in sys.modules
        print({_FAMILY!r} in tl.io.list_sidecar_families())
        """
    )
    assert out.split() == ["True"]


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


def test_fresh_process_read_validates_the_builtin_family() -> None:
    """A read-first consultation validates instead of degrading analysis-only."""

    out = _run_fresh(
        f"""
        import sys
        import torch
        import torchlens as tl
        from torchlens._io.sidecar import SIDECAR_ANNOTATIONS_KEY, SidecarError
        trace = tl.trace(torch.nn.Linear(4, 2), torch.randn(1, 4))
        envelope = {{"schema_id": "token_boundary_v1", "version": 1,
                     "owner": "torchlens", "payload": {{"stream": "sideways"}}}}
        trace.annotations.setdefault(SIDECAR_ANNOTATIONS_KEY, {{}})[{_FAMILY!r}] = envelope
        assert {_OWNER!r} not in sys.modules
        try:
            tl.io.read_sidecar(trace, {_FAMILY!r})
        except SidecarError as exc:
            print(exc.fields["code"])
        """
    )
    assert out.split() == ["sidecar_payload_invalid"]


def test_owner_imported_first_registers_the_family_once() -> None:
    """Importing the owner directly (the reentrant path) registers exactly once."""

    out = _run_fresh(
        f"""
        import importlib
        import torchlens as tl
        importlib.import_module({_OWNER!r})
        families = tl.io.list_sidecar_families()
        print(families.count({_FAMILY!r}))
        """
    )
    assert out.split() == ["1"]
