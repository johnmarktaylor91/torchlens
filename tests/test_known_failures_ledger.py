"""Keep tests/known_failures.py the single, accurate source of tracked xfails.

Two things can silently drift once a ledger exists: a row can go STALE (the
test it names was renamed, removed, or never existed under that id -- the
strict xfail would then just vanish, un-noticed, since pytest cannot resolve
it to anything), and a SECOND xfail source can creep in (a file decorates one
of its own tests directly, so the same test effectively has two owners for
"is this expected to fail", one of which conftest.py's known-failures hook
never sees). Both are red here.

This file does not need the Weekly environment (torch 2.7.1+cpu):
``tests/test_real_world_models.py`` only imports ``torchvision`` at
collection time (``pytest.importorskip``), and every optional model
dependency (``timm``, ``transformers``, ...) is imported inside the test
functions themselves -- so collecting it, unlike running it, works with the
ordinary ``[test]`` extra.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from known_failures import KNOWN_FAILURES, duplicate_nodeids, stale_entries

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every file the ledger names, derived from the entries themselves so this
#: test never needs a second, hand-maintained file list to keep in sync.
_LEDGER_FILES: tuple[str, ...] = tuple(
    sorted({entry.nodeid.split("::", 1)[0] for entry in KNOWN_FAILURES})
)


def _collect_node_ids(target: str) -> set[str]:
    """Return the node ids pytest collects from one file, via a subprocess.

    A subprocess (not ``pytest.main`` in-process) so this test's own
    collection/session state can never leak into, or be polluted by, the
    collection it is inspecting.
    """

    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    ids = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if "::" in line and line.split("::", 1)[0].endswith(".py"):
            ids.add(line)
    return ids


@pytest.mark.smoke
def test_known_failures_ledger_has_no_duplicate_nodeids() -> None:
    """Each tracked node id appears at most once (a second row would be dead code)."""

    duplicates = duplicate_nodeids()
    assert not duplicates, f"duplicate known_failures entries: {duplicates}"


@pytest.mark.heavy
def test_known_failures_match_collected_node_ids() -> None:
    """Every ledger entry must resolve to a node id pytest actually collects."""

    if not KNOWN_FAILURES:
        pytest.skip("ledger is empty; nothing to cross-check")

    collected: set[str] = set()
    for target in _LEDGER_FILES:
        collected |= _collect_node_ids(target)

    stale = stale_entries(collected)
    assert not stale, (
        "tests/known_failures.py has entries pytest cannot collect (renamed, "
        "removed, or never-existing node ids) -- fix or drop each:\n  "
        + "\n  ".join(f"{entry.nodeid} [{entry.reason}]" for entry in stale)
    )


@pytest.mark.smoke
def test_known_failures_ledger_is_the_only_xfail_source_in_its_files() -> None:
    """No file the ledger names may decorate a test with ``xfail`` directly.

    The ledger (via ``tests/conftest.py::apply_xfail_marks``) is the only
    mechanism that should mark these tests ``xfail`` -- a direct decorator in
    the test file itself would be a second, invisible-to-the-ledger source of
    the same contract and could silently diverge from it (different reason,
    different strictness, or a test the ledger considers fixed and removed
    while the decorator lingers).
    """

    offenders = []
    for target in _LEDGER_FILES:
        tree = ast.parse((REPO_ROOT / target).read_text(encoding="utf-8"), filename=target)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if "xfail" in ast.dump(decorator):
                    offenders.append(f"{target}::{node.name}")

    assert not offenders, (
        "these tests decorate xfail directly; known failures belong ONLY in "
        "tests/known_failures.py so the ledger stays the single source:\n  "
        + "\n  ".join(offenders)
    )
