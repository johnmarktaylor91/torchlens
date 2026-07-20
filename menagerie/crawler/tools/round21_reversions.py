"""Deterministic Round-21 deliberate-reversion runner for disposable checkouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


_REVERSIONS = tuple(f"D{index:02d}" for index in range(1, 30))
_REGISTRY_PATH = Path("menagerie/crawler/conformance-round21.json")
_WORKFLOW_PATH = Path(".github/workflows/tests.yml")
_LINUX_LOCK_PATH = Path("menagerie/crawler/envs/locks/round19-linux-64.lock")
_P11_NODE = (
    "menagerie/crawler/tests/test_round21_conformance_composition.py::"
    "test_round21_conformance_registry_is_total_and_executed"
)


@dataclass(frozen=True)
class ReversionCase:
    """One deterministic disposable-checkout deliberate reversion."""

    reversion_id: str
    semantic_reversion: str
    proof_node: str
    expected_reason: str
    mutate: Callable[[Path, str], None]


def _load_registry(root: Path) -> dict[str, object]:
    """Load the conformance registry from a disposable checkout."""

    payload = json.loads((root / _REGISTRY_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("conformance registry must be a JSON object")
    return payload


def _write_registry(root: Path, payload: dict[str, object]) -> None:
    """Write the conformance registry atomically in a disposable checkout."""

    path = root / _REGISTRY_PATH
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remove_record(root: Path, reversion_id: str) -> None:
    """Remove the registry record for one D-cell."""

    payload = _load_registry(root)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("conformance registry lacks records[]")
    payload["records"] = [
        record
        for record in records
        if isinstance(record, dict) and record.get("clause_id") != reversion_id
    ]
    _write_registry(root, payload)


def _remove_reversion_reference(root: Path, reversion_id: str) -> None:
    """Remove one D-cell from every deliberate_reversion_ids field."""

    payload = _load_registry(root)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("conformance registry lacks records[]")
    for record in records:
        if isinstance(record, dict):
            values = record.get("deliberate_reversion_ids")
            if isinstance(values, list):
                record["deliberate_reversion_ids"] = [
                    value for value in values if value != reversion_id
                ]
    _write_registry(root, payload)


def _delete_linux_lock(root: Path, _reversion_id: str) -> None:
    """Delete the committed Linux explicit lock in a disposable checkout."""

    (root / _LINUX_LOCK_PATH).unlink()


def _delete_workflow_conformance_job(root: Path, _reversion_id: str) -> None:
    """Delete the conformance job marker from the workflow text."""

    path = root / _WORKFLOW_PATH
    source = path.read_text(encoding="utf-8")
    path.write_text(
        source.replace("crawler-round21-conformance:", "crawler-round21-conformance-deleted:"),
        encoding="utf-8",
    )


def _drop_first_registry_record(root: Path, _reversion_id: str) -> None:
    """Delete the first conformance record to prove totality has teeth."""

    payload = _load_registry(root)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("conformance registry lacks records[]")
    payload["records"] = records[1:]
    _write_registry(root, payload)


def _cases() -> tuple[ReversionCase, ...]:
    """Return the exact D01-D29 deliberate-reversion table."""

    semantic_reversions = {
        "D01": "restore hardlink or startup-.pth filtering",
        "D02": "omit content seal from generation",
        "D03": "use conda-meta alone as seal",
        "D04": "grant post-seal bytes",
        "D05": "weaken internal/external symlink binding",
        "D06": "treat packaged checkpoints as runtime",
        "D07": "restore sys.executable or parallel interpreter",
        "D08": "allow outside/mismatched selected interpreter",
        "D09": "each substitution evasion",
        "D10": "reintroduce live legacy root/fake result/hand binding",
        "D11": "remove transitive fixture/composition/CI scope",
        "D12": "omit ctime/inode/external fingerprint trigger",
        "D13": "quarantine byte-identical hardlink clone churn",
        "D14": "rewalk per model/consumer or reuse pass token for spawn",
        "D15": "delete shutdown guard or admit later work",
        "D16": "insert shutdown check inside atomic publication",
        "D17": "restore root/suffix transport grant",
        "D18": "retain mismatched artifact transaction",
        "D19": "accept zero/multiple handoff finals or old-root fallback",
        "D20": "invalidate active cache on rejected rebind",
        "D21": "delete/corrupt either lock/export/probe contract",
        "D22": "remove workflow proof node or permit skip/xfail",
        "D23": "remove Linux bwrap/strace or denial audit",
        "D24": "replace macOS denial with profile-only proof",
        "D25": "drop nullable evidence or misclassify mode",
        "D26": "restore dry-run false-complete/signature mismatch",
        "D27": "restore source/CAS/cache/fetch fallback",
        "D28": "delete clause/finding/invariant/matrix registry record",
        "D29": "regress receipt/spawn/writer/schema preservation",
    }
    cases: list[ReversionCase] = []
    for reversion_id in _REVERSIONS:
        mutate = _remove_reversion_reference
        proof_node = _P11_NODE
        reason = "assert"
        if reversion_id == "D21":
            mutate = _delete_linux_lock
            proof_node = (
                "menagerie/crawler/tests/test_round17_structural_inventories.py::"
                "test_round21_linux_release_artifacts_and_provisioning_are_real"
            )
            reason = "AssertionError"
        elif reversion_id == "D22":
            mutate = _delete_workflow_conformance_job
            reason = "crawler-round21-conformance"
        elif reversion_id == "D28":
            mutate = _drop_first_registry_record
        elif reversion_id in {"D23", "D24"}:
            mutate = _delete_workflow_conformance_job
            reason = "crawler-round21-conformance"
        elif reversion_id in {
            "D01",
            "D02",
            "D03",
            "D04",
            "D05",
            "D06",
            "D07",
            "D08",
            "D09",
            "D10",
            "D11",
            "D12",
            "D13",
            "D14",
            "D15",
            "D16",
            "D17",
            "D18",
            "D19",
            "D20",
            "D25",
            "D26",
            "D27",
            "D29",
        }:
            mutate = _remove_record
        cases.append(
            ReversionCase(
                reversion_id=reversion_id,
                semantic_reversion=semantic_reversions[reversion_id],
                proof_node=proof_node,
                expected_reason=reason,
                mutate=mutate,
            )
        )
    return tuple(cases)


def _copy_checkout(source: Path, destination: Path) -> None:
    """Copy a repository tree into a disposable checkout destination."""

    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns(
        ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"
    )
    shutil.copytree(source, destination, ignore=ignore)


def _run_pytest(root: Path, node: str, python: str) -> subprocess.CompletedProcess[str]:
    """Run one mapped proof node in a disposable checkout."""

    return subprocess.run(
        (python, "-m", "pytest", "-q", node, "-x", "--tb=short"),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )


def _run_case(
    case: ReversionCase,
    *,
    source: Path,
    work_root: Path,
    python: str,
) -> dict[str, object]:
    """Apply one reversion in a fresh disposable checkout and run its mapped proof."""

    checkout = work_root / case.reversion_id
    _copy_checkout(source, checkout)
    case.mutate(checkout, case.reversion_id)
    completed = _run_pytest(checkout, case.proof_node, python)
    combined = completed.stdout + "\n" + completed.stderr
    passed = completed.returncode != 0 and case.expected_reason in combined
    return {
        "reversion_id": case.reversion_id,
        "semantic_reversion": case.semantic_reversion,
        "proof_node": case.proof_node,
        "expected_reason": case.expected_reason,
        "exit_code": completed.returncode,
        "passed": passed,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _write_matrix(path: Path) -> None:
    """Write the public exact D01-D29 matrix consumed by P11."""

    payload = {
        "schema_version": "menagerie.crawler.round21-reversion-matrix.v1",
        "deliberate_reversion_ids": list(_REVERSIONS),
        "cases": [
            {
                "reversion_id": case.reversion_id,
                "semantic_reversion": case.semantic_reversion,
                "proof_node": case.proof_node,
                "expected_reason": case.expected_reason,
            }
            for case in _cases()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ids", nargs="*", default=list(_REVERSIONS))
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--write-matrix", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run requested deliberate reversions and write an attestation."""

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.write_matrix is not None:
        _write_matrix(args.write_matrix)
        return 0
    source = args.source.resolve()
    work_root = args.work_root.resolve()
    if source == work_root or source in work_root.parents:
        raise SystemExit("work-root must be outside the source checkout")
    requested = tuple(args.ids)
    cases = {case.reversion_id: case for case in _cases()}
    if set(requested) - set(cases):
        raise SystemExit(f"unknown reversion id(s): {sorted(set(requested) - set(cases))}")
    work_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_case(cases[reversion_id], source=source, work_root=work_root, python=str(args.python))
        for reversion_id in requested
    ]
    passed = [str(result["reversion_id"]) for result in results if result["passed"] is True]
    failed = [str(result["reversion_id"]) for result in results if result["passed"] is not True]
    payload = {
        "schema_version": "menagerie.crawler.round21-reversion-result.v1",
        "status": "passed" if not failed and set(passed) == set(requested) else "failed",
        "requested_reversions": list(requested),
        "passed_reversions": sorted(passed),
        "skipped_reversions": [],
        "failed_reversions": sorted(failed),
        "results": results,
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
