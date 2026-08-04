"""Headline real-forward end-to-end acceptance dry-run through the public CLI."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.cli import (
    EXIT_OK,
    EXIT_REVIEW_PAUSED,
    _persisted_environment_generations,
)
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import materialize_current
from menagerie.crawler.status import assert_partition, funnel_counts
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.dry_run_support import (
    DRY_RUN_CASES,
    DRY_RUN_ITEMS,
    dry_run_paths,
    read_notification_summaries,
)
from .support import repository_root


def _run_cli(
    repo_root: Path,
    environment_prefix: Path,
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run the crawler module in a real child process and capture its public output.

    The child runs under the bound prefix's own interpreter, exactly as the documented
    dry-run procedure does, so the awards it grants come from the real environment
    rather than from whichever interpreter happens to host pytest.

    Parameters
    ----------
    repo_root, environment_prefix:
        Checked-out source root and the materialized real prefix the driver binds.
    arguments:
        CLI arguments following the shared ``--repo-root`` selection.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured real CLI process result.
    """

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    # Fail closed in the child: an unmet prerequisite must surface as a failure here, not
    # as a quietly degraded run that still reports a clean acceptance payload.
    environment["MENAGERIE_RELEASE_GATE"] = "1"
    return subprocess.run(
        [
            str(environment_prefix / "bin" / "python"),
            "-B",
            "-m",
            "menagerie.crawler",
            "--repo-root",
            str(repo_root),
            *arguments,
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def _payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse the CLI's final one-line JSON payload."""

    value = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(value, dict)
    return value


#: Every intake item is forwarded once per mode in each of two independent cold rounds.
#: The repeat is what lets the driver tell a real train/eval divergence from run-to-run
#: noise, so the count is a contract, not incidental: an item that skipped its second cold
#: round would still produce a divergence verdict, just an unsupported one.
_COLD_ROUNDS = 2
_FORWARD_MODES = 2
_FORWARDS_PER_ITEM = _COLD_ROUNDS * _FORWARD_MODES

#: Exact worker-result envelope the on-disk receipt assertions below read.
_WORKER_RESULT_VERSION = "menagerie.crawler.worker-result.v3"

#: Operational events this acceptance test is about. The operational ledger also carries
#: routine worker-lease bookkeeping, which is not part of the review/notification contract;
#: selecting these kinds keeps the ordered assertions exact without pinning unrelated rows.
_REVIEW_EVENT_KINDS = frozenset(
    {
        "checkpoint-review",
        "review-signoff",
        "progress-notification",
        "notification-delivery",
    }
)


def _review_event_kinds(operational_ledger: Path) -> list[str]:
    """Return the review and notification event kinds in exact append order.

    Parameters
    ----------
    operational_ledger:
        Canonical operational ledger path for the campaign.

    Returns
    -------
    list[str]
        Ordered review/notification event kinds, excluding worker-lease bookkeeping.
    """

    return [
        str(event["event_kind"])
        for event in scan_jsonl(operational_ledger)
        if str(event["event_kind"]) in _REVIEW_EVENT_KINDS
    ]


def _current_records(campaign_root: Path, snapshot: IntakeSnapshot) -> Mapping[str, Any]:
    """Materialize the authenticated current projection for one dry-run campaign.

    The projection is authority-scoped: it is only meaningful against the exact intake
    snapshot and agent identities the run recorded, so the context is rebuilt from those
    facts rather than assumed.

    Parameters
    ----------
    campaign_root, snapshot:
        Disposable campaign root and the intake snapshot the run consumed.

    Returns
    -------
    Mapping[str, Any]
        Highest valid dependency-current revision per stable ID.
    """

    paths = dry_run_paths(campaign_root, snapshot)
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model="fake-claude",
        author_version="dry-run",
        checker_model="fake-codex",
        checker_version="dry-run",
    )
    context = replace(
        context,
        environment_generations=_persisted_environment_generations(
            scan_jsonl(paths.ledgers.attempts)
        ),
    )
    return materialize_current(paths.ledgers, context=context)


def _ledger_prefixes(paths: Sequence[Path]) -> dict[Path, bytes]:
    """Capture existing immutable ledger bytes before a resume appends more facts."""

    return {path: path.read_bytes() for path in paths if path.is_file()}


def _assert_receipt_observations(campaign_root: Path, stable_id_to_name: Mapping[str, str]) -> None:
    """Assert real worker receipts match every tiny model's intended behavior."""

    receipt_paths = sorted(
        (campaign_root / "runtime" / "work").glob("*/forward/cold-*/*/result/receipt.json")
    )
    receipts = [
        (path.parts[-3], json.loads(path.read_text(encoding="utf-8"))) for path in receipt_paths
    ]
    assert len(receipts) == _FORWARDS_PER_ITEM * len(DRY_RUN_ITEMS)
    observed_by_name: dict[str, list[dict[str, Any]]] = {}
    modes_by_stable_id: dict[str, list[str]] = {}
    for requested_mode, envelope in receipts:
        # Pinned: a worker-result schema bump must be looked at here rather than silently
        # reinterpreted, because every assertion below reads that exact shape.
        assert envelope["result_version"] == _WORKER_RESULT_VERSION
        receipt = envelope["raw_award_receipt"]
        observation = receipt["observation"]
        # A worker receipt is an observation, never an award. The runs decision belongs to
        # the driver after both modes and review, so no award claim may appear here.
        assert "awards_runs" not in receipt
        assert "awards_runs" not in observation
        stable_id = str(receipt["stable_id"])
        name = stable_id_to_name[stable_id]
        observed_by_name.setdefault(name, []).append(receipt)
        modes_by_stable_id.setdefault(stable_id, []).append(requested_mode)
        # One receipt observes exactly one requested mode, in its own isolated forward.
        assert receipt["requested_mode"] == requested_mode
        assert observation["mode"] == requested_mode
        assert observation["forward_completed"]
        assert observation["input_kind"] == "standard-typed-dummy-call"
    item_counts = {
        name: sum(item_name == name for item_name, _variant in DRY_RUN_ITEMS)
        for name in {item_name for item_name, _variant in DRY_RUN_ITEMS}
    }
    assert {name: len(values) for name, values in observed_by_name.items()} == {
        case.name: _FORWARDS_PER_ITEM * item_counts[case.name] for case in DRY_RUN_CASES
    }
    assert len(modes_by_stable_id) == len(DRY_RUN_ITEMS)
    expected_modes = sorted(["train", "eval"] * _COLD_ROUNDS)
    assert all(sorted(modes) == expected_modes for modes in modes_by_stable_id.values())


def test_cli_dry_run_real_forward_checkpoint_resume_and_milestone(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """The real driver reaches runs only after isolated train/eval forwards and review."""

    repo_root = repository_root()
    campaign_root = tmp_path / "campaign"
    environment_prefix = real_environment_fixture.prefix
    common = [
        "--dry-run",
        "--dry-run-root",
        str(campaign_root),
        "--dry-run-environment-prefix",
        str(environment_prefix),
        "--review-checkpoint-at",
        "2",
        "--progress-milestones",
        "3",
        "--run-id",
        "slice-h-dry-run",
    ]
    first = _run_cli(repo_root, environment_prefix, ["run", *common])
    # The named constant, not a literal: a review pause got its own exit code apart from
    # the generic pause, and a literal 4 would now accept the wrong kind of pause.
    assert first.returncode == EXIT_REVIEW_PAUSED, first.stderr
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
    assert _review_event_kinds(paths.operational_ledger) == [
        "checkpoint-review",
        "notification-delivery",
    ]

    resumed = _run_cli(repo_root, environment_prefix, ["resume", *common, "--after-review"])
    assert resumed.returncode == EXIT_OK, resumed.stderr
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
        # Every tiny model is authored as a staged typed port, so the corpus is uniformly
        # R3_PORT; the earlier 9/1 split predates the staged-code author.
        "rung:R3_PORT": 10,
        "status:runs": 10,
    }
    status = _run_cli(
        repo_root,
        environment_prefix,
        [
            "status",
            "--intake",
            str(snapshot.root),
            "--records-root",
            str(campaign_root / "records"),
            "--verify-partition",
        ],
    )
    assert status.returncode == EXIT_OK, status.stderr
    status_payload = _payload(status)
    # A record is only current for the agent identities its caller declares, and the
    # production status command declares the production ones. This campaign was authored
    # by the dry-run fake lanes, so status must attribute NONE of it rather than count
    # records it cannot authenticate. Pinning the refusal keeps that authority gate
    # visible: a regression that let status adopt unattributable records fails here.
    assert status_payload["terminal"] == 0
    assert status_payload["partition_valid"] is False
    assert sorted(status_payload["missing"]) == sorted(item.stable_id for item in snapshot.items)

    for path, prefix in prefixes.items():
        assert path.read_bytes().startswith(prefix)
    current = _current_records(campaign_root, snapshot)
    # Under the authority the driver actually ran with, the independently materialized
    # projection must reproduce the driver's own funnel and cover the intake exactly.
    assert funnel_counts(current) == resumed_payload["funnel"]
    partition = assert_partition((item.stable_id for item in snapshot.items), current)
    assert partition.valid
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
    assert len(attempts) == _FORWARDS_PER_ITEM * len(DRY_RUN_ITEMS)
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
