"""Shared runner for the repository-policy checks in ``scripts/checks/``.

Each check module defines ``CHECKS``, a tuple of zero-argument callables that
raise ``AssertionError`` on a policy violation, and a ``main()`` that calls
:func:`run_checks`. The checks use plain ``assert`` statements, so the runner
refuses to start under ``python -O``, where those statements would vanish.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
PACKAGE_ROOT = REPO_ROOT / "torchlens"


class CheckSkipped(Exception):
    """Raised by a check whose precondition is absent (the former ``pytest.skip``)."""


def require_assertions() -> None:
    """Exit when assertions are disabled; the checks are written as asserts."""

    if not __debug__:
        sys.exit("scripts/checks: run without -O; the checks are assert statements")


def use_test_helpers() -> None:
    """Make the shared helpers in ``tests/`` (``_source_corpus`` and friends) importable."""

    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))


def run_checks(name: str, checks: Sequence[Callable[[], None]]) -> int:
    """Run every check, print one line per failure, and return an exit status.

    Parameters
    ----------
    name:
        Check module name used in the printed lines.
    checks:
        Zero-argument callables that raise ``AssertionError`` on a violation.

    Returns
    -------
    int
        ``0`` when every check passed, else ``1``.
    """

    require_assertions()
    failed = 0
    for check in checks:
        try:
            check()
        except CheckSkipped as skipped:
            print(f"SKIP {name}::{check.__name__}: {skipped}")
        except Exception as error:  # noqa: BLE001 - every failure is reported, none hidden
            failed += 1
            print(f"FAIL {name}::{check.__name__}: {type(error).__name__}")
            print(
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                file=sys.stderr,
            )
    print(f"{name}: {len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0
