"""Run every repository policy check and print one line each.

Each entry in ``CHECKS`` is ``(name, callable)``; the callable takes the
repository root and returns an ``Outcome``. Register a new check by appending
to ``CHECKS``. The process exits 1 when any check fails.

Usage: ``python scripts/checks/run_all.py [--repo PATH] [--verbose]``.
Set ``TORCHLENS_CHECKS_STRICT=1`` to make a skipped check (a missing optional
tool) count as a failure, as a CI leg that installs the tools should.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import file_cap  # noqa: E402
import governance  # noqa: E402
import id_lint  # noqa: E402
import layering  # noqa: E402
from _common import FAIL, SKIP, Outcome  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CHECKS: list[tuple[str, Callable[[Path], Outcome]]] = [
    ("layering", layering.check),
    ("import-contracts", layering.check_contracts),
    ("file-cap", file_cap.check),
    ("id-lint", id_lint.check),
    # Policy checks moved out of tests/ (each module's own checks; see governance.py).
    ("internal-id-lint", governance.as_check("internal_id_lint")),
    ("mypy-ratchet", governance.as_check("mypy_ratchet")),
    ("not-mvp-audit", governance.as_check("not_mvp_audit")),
    ("filename-convention-lint", governance.as_check("filename_convention_lint")),
    ("deprecation-inventory", governance.as_check("deprecation_inventory")),
    ("file-size-ratchet", governance.as_check("file_size_ratchet")),
    ("claims-lint", governance.as_check("claims_lint")),
    ("arch-spine-layer-lint", governance.as_check("arch_spine_layer_lint")),
    ("factcore-import-lint", governance.as_check("factcore_import_lint")),
    ("arch-tail-vocab-lint", governance.as_check("arch_tail_vocab_lint")),
    ("arch-tail-seam-census", governance.as_check("arch_tail_seam_census")),
    ("arch-tail-surface-audit", governance.as_check("arch_tail_surface_audit")),
    ("state-access-ratchet", governance.as_check("state_access_ratchet")),
    ("error-contract-lockstep", governance.as_check("error_contract_lockstep")),
    ("docs-lockstep-names", governance.as_check("docs_lockstep_names")),
    ("terminology-lint", governance.as_check("terminology_lint")),
    ("removed-spelling-lint", governance.as_check("removed_spelling_lint")),
    ("ruff-ratchet", governance.as_check("ruff_ratchet")),
    ("known-failures-ledger", governance.as_check("known_failures_ledger")),
    ("skip-audit", governance.as_check("skip_audit")),
    ("schema-lockstep", governance.as_check("schema_lockstep")),
    ("golden-governance-lint", governance.as_check("golden_governance_lint")),
]


def run(repo: Path, verbose: bool = False, strict: bool = False) -> int:
    """Run all checks against ``repo``; return the process exit code."""

    failed = 0
    for name, check in CHECKS:
        try:
            outcome = check(repo)
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            outcome = Outcome(FAIL, f"{name}: crashed: {type(exc).__name__}: {exc}")
        if outcome.status == SKIP and strict:
            outcome = Outcome(FAIL, f"{outcome.message} (strict mode)", outcome.details)
        print(f"[{outcome.status.upper():4}] {outcome.message}")
        if verbose or outcome.failed:
            for line in outcome.details:
                print(f"       {line}")
        failed += outcome.failed
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--verbose", action="store_true", help="print details for every check")
    args = parser.parse_args(argv)
    strict = os.environ.get("TORCHLENS_CHECKS_STRICT") == "1"
    return run(args.repo.resolve(), verbose=args.verbose, strict=strict)


if __name__ == "__main__":
    sys.exit(main())
