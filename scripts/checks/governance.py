"""The repository-policy checks that moved out of ``tests/`` into this folder.

Each name in ``MOVED`` is a module here with a ``main()`` that runs its checks
and returns an exit status. :func:`run` calls one; :func:`as_check` adapts one
to the ``(repo) -> Outcome`` callable that ``run_all.py``'s ``CHECKS`` holds.
The checks read the checkout that contains this folder; for any other
``repo`` the adapter reports a skip.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from collections.abc import Callable
from pathlib import Path

_CHECKS_DIR = Path(__file__).resolve().parent
if str(_CHECKS_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKS_DIR))

from _common import FAIL, OK, SKIP, Outcome  # noqa: E402

MOVED: tuple[str, ...] = (
    "internal_id_lint",
    "mypy_ratchet",
    "not_mvp_audit",
    "filename_convention_lint",
    "deprecation_inventory",
    "file_size_ratchet",
    "claims_lint",
    "arch_spine_layer_lint",
    "factcore_import_lint",
    "arch_tail_vocab_lint",
    "arch_tail_seam_census",
    "arch_tail_surface_audit",
    "state_access_ratchet",
    "error_contract_lockstep",
    "docs_lockstep_names",
    "terminology_lint",
    "removed_spelling_lint",
    "ruff_ratchet",
    "known_failures_ledger",
    "skip_audit",
    "schema_lockstep",
    "golden_governance_lint",
)


def run(name: str) -> int:
    """Import the moved check module ``name``, run its ``main()``, return its status."""

    return int(importlib.import_module(name).main())


def as_check(name: str) -> Callable[[Path], Outcome]:
    """Return a ``(repo) -> Outcome`` callable for ``run_all.py``'s ``CHECKS`` list."""

    def check(repo: Path) -> Outcome:
        if repo.resolve() != _CHECKS_DIR.parents[1]:
            # The moved checks read this checkout only; another tree (a fixture) is
            # not theirs to judge. Strict mode (TORCHLENS_CHECKS_STRICT=1) fails a skip.
            return Outcome(SKIP, f"{name}: checks only the checkout that holds scripts/checks")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = run(name)
        lines = tuple(line for line in buffer.getvalue().splitlines() if line)
        summary = lines[-1] if lines else name
        failures = tuple(line for line in lines if line.startswith(("FAIL", "SKIP")))
        return Outcome(OK if status == 0 else FAIL, summary, failures)

    check.__name__ = name
    return check
