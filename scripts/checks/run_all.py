"""Run every repository-policy check in ``scripts/checks/``.

``CHECKS`` lists ``(name, callable)`` pairs; each callable returns an exit
status (``0`` passes). ``python scripts/checks/run_all.py`` prints one line per
check and exits non-zero when any check fails. Pass check names as arguments to
run a subset.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path

_CHECKS_DIR = Path(__file__).resolve().parent
if str(_CHECKS_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKS_DIR))

import governance  # noqa: E402

CHECKS: list[tuple[str, Callable[[], int]]] = [
    # Moved from tests/ (governance checks run as scripts, not by the test runner).
    (name, functools.partial(governance.run, name))
    for name in governance.MOVED
]


def main(argv: list[str] | None = None) -> int:
    """Run the selected checks (all by default) and return the exit status."""

    wanted = set(sys.argv[1:] if argv is None else argv)
    unknown = wanted - {name for name, _ in CHECKS}
    if unknown:
        print(f"unknown checks: {sorted(unknown)}")
        return 2
    failed = []
    for name, check in CHECKS:
        if wanted and name not in wanted:
            continue
        try:
            status = check()
        except Exception as error:  # noqa: BLE001 - a crashing check is a failing check
            print(f"{name}: crashed: {type(error).__name__}: {error}")
            status = 1
        print(f"{'ok  ' if status == 0 else 'FAIL'} {name}")
        if status != 0:
            failed.append(name)
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
