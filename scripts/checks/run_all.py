"""Run every repository-policy check in ``scripts/checks/``.

``CHECKS`` lists ``(name, callable)`` pairs; each callable returns an exit
status (``0`` passes). ``python scripts/checks/run_all.py`` prints one line per
check and exits non-zero when any check fails. Pass check names as arguments to
run a subset.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

_CHECKS_DIR = Path(__file__).resolve().parent
if str(_CHECKS_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKS_DIR))


def _module_main(module_name: str) -> Callable[[], int]:
    """Return a callable that imports ``module_name`` and runs its ``main()``."""

    def run() -> int:
        return int(importlib.import_module(module_name).main())

    run.__name__ = module_name
    return run


CHECKS: list[tuple[str, Callable[[], int]]] = [
    # Moved from tests/ (governance checks run as scripts, not by the test runner).
    ("internal_id_lint", _module_main("internal_id_lint")),
    ("mypy_ratchet", _module_main("mypy_ratchet")),
    ("not_mvp_audit", _module_main("not_mvp_audit")),
    ("filename_convention_lint", _module_main("filename_convention_lint")),
    ("deprecation_inventory", _module_main("deprecation_inventory")),
    ("file_size_ratchet", _module_main("file_size_ratchet")),
    ("claims_lint", _module_main("claims_lint")),
    ("arch_spine_layer_lint", _module_main("arch_spine_layer_lint")),
    ("factcore_import_lint", _module_main("factcore_import_lint")),
    ("arch_tail_vocab_lint", _module_main("arch_tail_vocab_lint")),
    ("arch_tail_seam_census", _module_main("arch_tail_seam_census")),
    ("arch_tail_surface_audit", _module_main("arch_tail_surface_audit")),
    ("state_access_ratchet", _module_main("state_access_ratchet")),
    ("error_contract_lockstep", _module_main("error_contract_lockstep")),
    ("docs_lockstep_names", _module_main("docs_lockstep_names")),
    ("terminology_lint", _module_main("terminology_lint")),
    ("removed_spelling_lint", _module_main("removed_spelling_lint")),
    ("ruff_ratchet", _module_main("ruff_ratchet")),
    ("known_failures_ledger", _module_main("known_failures_ledger")),
    ("skip_audit", _module_main("skip_audit")),
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
