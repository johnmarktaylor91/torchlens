"""Headline real-forward end-to-end acceptance dry-run through the public CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import materialize_current
from menagerie.crawler.tests.dry_run_support import (
    DRY_RUN_CASES,
    DRY_RUN_ITEMS,
    dry_run_paths,
    read_notification_summaries,
)
from .support import repository_root


def _run_cli(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run the crawler module in a real child process and capture its public output."""

    return subprocess.run(
        [sys.executable, "-m", "menagerie.crawler", "--repo-root", str(repo_root), *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse the CLI's final one-line JSON payload."""

    value = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(value, dict)
    return value


def _ledger_prefixes(paths: Sequence[Path]) -> dict[Path, bytes]:
    """Capture existing immutable ledger bytes before a resume appends more facts."""

    return {path: path.read_bytes() for path in paths if path.is_file()}


def _assert_receipt_observations(campaign_root: Path, stable_id_to_name: Mapping[str, str]) -> None:
    """Assert real worker receipts match every tiny model's intended behavior."""

    expected = {case.name: case for case in DRY_RUN_CASES}
    receipt_paths = sorted(
        (campaign_root / "runtime" / "work").glob("*/forward/cold-*/*/result/receipt.json")
    )
    receipts = [
        (path.parts[-3], json.loads(path.read_text(encoding="utf-8"))) for path in receipt_paths
    ]
    assert len(receipts) == 22
    observed_by_name: dict[str, list[dict[str, Any]]] = {}
    for requested_mode, receipt in receipts:
        name = stable_id_to_name[str(receipt["stable_id"])]
        observed_by_name.setdefault(name, []).append(receipt)
        assert receipt["awards_runs"] is False
        assert requested_mode in receipt["per_mode"]
        assert receipt["per_mode"][requested_mode]["forward_completed"]
        assert receipt["per_mode"][requested_mode]["input_kind"] == ("standard-typed-dummy-call")
        if set(receipt["per_mode"]) == {"train", "eval"}:
            assert receipt["train_eval_divergence"] == expected[name].divergence
    item_counts = {
        name: sum(item_name == name for item_name, _variant in DRY_RUN_ITEMS)
        for name in {item_name for item_name, _variant in DRY_RUN_ITEMS}
    }
    assert {name: len(values) for name, values in observed_by_name.items()} == {
        case.name: 2 * (item_counts[case.name] + int(case.fidelity_required))
        for case in DRY_RUN_CASES
    }


def test_cli_dry_run_real_forward_checkpoint_resume_and_milestone(tmp_path: Path) -> None:
    """The real driver reaches runs only after isolated train/eval forwards and review."""

    repo_root = repository_root()
    campaign_root = tmp_path / "campaign"
    common = [
        "--dry-run",
        "--dry-run-root",
        str(campaign_root),
        "--review-checkpoint-at",
        "2",
        "--progress-milestones",
        "3",
        "--run-id",
        "slice-h-dry-run",
    ]
    first = _run_cli(repo_root, ["run", *common])
    assert first.returncode == 4, first.stderr
    first_payload = _payload(first)
    assert first_payload["status"] == "paused:review-checkpoint"
    assert first_payload["terminal_models"] == 2
    assert first_payload["funnel"]["status:runs"] == 2

    snapshot_root = next((campaign_root / "intake").glob("intake-*"))
    snapshot = load_intake_snapshot(snapshot_root)
    paths = dry_run_paths(campaign_root, snapshot)
    ledger_paths = (
        paths.ledgers.models,
        paths.ledgers.attempts,
        paths.ledgers.gates,
        paths.operational_ledger,
    )
    prefixes = _ledger_prefixes(ledger_paths)
    assert len(scan_jsonl(paths.ledgers.models)) == 2
    assert [event["event_kind"] for event in scan_jsonl(paths.operational_ledger)] == [
        "checkpoint-review",
        "notification-delivery",
    ]

    resumed = _run_cli(repo_root, ["resume", *common, "--after-review"])
    assert resumed.returncode == 0, resumed.stderr
    resumed_payload = _payload(resumed)
    assert resumed_payload["status"] == "complete"
    assert resumed_payload["terminal_models"] == 10
    assert resumed_payload["models_reduced"] == 8
    assert resumed_payload["funnel"] == {
        "framework:pytorch": 10,
        "metadata:accepted": 10,
        "mode:eval": 10,
        "mode:train": 10,
        "models:total": 10,
        "rung:R1_LIBRARY": 9,
        "rung:R3_PORT": 1,
        "status:runs": 10,
    }
    status = _run_cli(
        repo_root,
        [
            "status",
            "--intake",
            str(snapshot.root),
            "--records-root",
            str(campaign_root / "records"),
            "--verify-partition",
        ],
    )
    assert status.returncode == 0, status.stderr
    status_payload = _payload(status)
    assert status_payload["terminal"] == 10
    assert status_payload["partition_valid"] is True
    assert status_payload["funnel"] == resumed_payload["funnel"]

    for path, prefix in prefixes.items():
        assert path.read_bytes().startswith(prefix)
    current = materialize_current(paths.ledgers)
    stable_id_to_name = {item.stable_id: item.name for item in snapshot.items}
    expected_by_name = {case.name: case for case in DRY_RUN_CASES}
    assert len(current) == 10
    for stable_id, record in current.items():
        case = expected_by_name[stable_id_to_name[stable_id]]
        assert record["status"]["code"] == "runs"
        assert record["status"]["environment"] == "core"
        assert record["revised_by"] == {"actor": "driver"}
        assert record["modes"]["train_eval_divergence"] == case.divergence
        assert set(record["modes"]["per_mode_run"]) == {"train", "eval"}
        assert record["observed"]["input_kind"] == "standard-typed-dummy-call"
        assert record["accuracy_gate"]["verdict"] == "accurate"
    structural_id = next(
        stable_id
        for stable_id, name in stable_id_to_name.items()
        if name == "DryRunStructuralBranch"
    )
    assert current[structural_id]["fidelity"]["verdict"] == "match"

    attempts = scan_jsonl(paths.ledgers.attempts)
    assert len(attempts) == 22
    assert all(attempt["result"] == "succeeded" for attempt in attempts)
    assert all(attempt["worker_receipt"]["mode"] in {"train", "eval"} for attempt in attempts)
    _assert_receipt_observations(campaign_root, stable_id_to_name)

    events = scan_jsonl(paths.operational_ledger)
    event_kinds = [event["event_kind"] for event in events]
    assert event_kinds.count("checkpoint-review") == 1
    assert event_kinds.count("review-signoff") == 1
    assert event_kinds.count("progress-notification") == 1
    assert event_kinds.count("notification-delivery") == 2
    progress = next(event for event in events if event["event_kind"] == "progress-notification")
    assert progress["milestone"] == 3
    summaries = read_notification_summaries(campaign_root / "notifications.jsonl")
    assert sum("review checkpoint" in summary for summary in summaries) == 1
    assert sum("milestone 3" in summary for summary in summaries) == 1
