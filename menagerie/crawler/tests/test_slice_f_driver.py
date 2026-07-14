"""Slice F single-writer driver, award, pause, and notification tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.checker_dispatch import CheckerBackoffSignal
from menagerie.crawler.constants import (
    CheckerPauseReason,
)
from menagerie.crawler.driver import (
    AuthorArtifact,
    AuthorLane,
    CheckerLane,
    CheckerOutcome,
    CommandNotifier,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverPaths,
    EnvironmentLane,
    ForwardLane,
    Notifier,
    UsagePauseScheduler,
    WorkItem,
)
from menagerie.crawler.envs import EnvironmentIntent, load_environment_registry
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.metadata import MANDATORY_EXTERNAL_FIELDS
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.conftest import (
    HASH,
    NOW,
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
)
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupBackend,
    WakeupManager,
)


class InjectedKill(RuntimeError):
    """Simulate an uncatchable process boundary failure in a deterministic test."""


class FakeAuthor(AuthorLane):
    """Return complete synthetic proposals without a live author session."""

    def __init__(self) -> None:
        """Initialize per-model invocation counts."""

        self.calls: dict[str, int] = {}

    def author(self, item: WorkItem, work_root: Path, config: DriverConfig) -> AuthorArtifact:
        """Return one accepted two-mode proposal."""

        del config
        self.calls[item.stable_id] = self.calls.get(item.stable_id, 0) + 1
        proposal = make_author_proposal(item.stable_id)
        proposal["work_id"] = f"work-{item.stable_id}"
        facts = proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["train", "eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
        model_dir = work_root / item.stable_id / "fake-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return AuthorArtifact(proposal, {"sources": []}, model_dir)


class FakeChecker(CheckerLane):
    """Return accurate synthetic gates or one configured quota signal."""

    def __init__(self, *, quota: bool = False) -> None:
        """Configure whether metadata checking reports quota exhaustion."""

        self.quota = quota
        self.metadata_calls = 0
        self.fidelity_calls = 0

    def check_metadata(
        self, artifacts: Sequence[AuthorArtifact], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return one exhaustive accurate metadata gate."""

        del work_root, config
        self.metadata_calls += 1
        if self.quota:
            return CheckerOutcome(
                backoff=CheckerBackoffSignal(
                    CheckerPauseReason.QUOTA_EXHAUSTED,
                    None,
                    "2026-07-14T13:00:00Z",
                    "quota exhausted",
                )
            )
        stable_ids = [str(artifact.proposal["stable_id"]) for artifact in artifacts]
        gate = make_gate(stable_ids, gate_id=f"gate-metadata-{stable_ids[0]}")
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        for item in gate["items"]:
            item["field_checks"] = [
                {
                    "field": f"external_metadata.{field}",
                    "verdict": "accurate",
                    "evidence_ids": ["evidence-1"],
                    "checked_source_ids": ["source-1"],
                    "reason": "supported",
                    "required_repair": None,
                }
                for field in MANDATORY_EXTERNAL_FIELDS
            ]
        return CheckerOutcome(gate=gate)

    def check_fidelity(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Return an accurate match fidelity gate."""

        del work_root, config
        self.fidelity_calls += 1
        stable_id = str(artifact.proposal["stable_id"])
        gate = make_gate(
            [stable_id],
            gate_id=f"gate-fidelity-{stable_id}",
            gate_kind="fidelity",
            fidelity_identity=str(artifact.proposal["fidelity_identity"]),
        )
        gate.pop("ledger_seq", None)
        gate.pop("payload_sha256", None)
        return CheckerOutcome(gate=gate)


class FakeForward(ForwardLane):
    """Return schema-valid attempts for every mode/cold invocation."""

    def __init__(self) -> None:
        """Initialize per-model invocation counts."""

        self.calls: dict[str, int] = {}

    def forward(
        self,
        artifact: AuthorArtifact,
        environment_prefix: Path,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Build clean train/eval attempts with complete model output signatures."""

        del environment_prefix, work_root
        stable_id = str(artifact.proposal["stable_id"])
        self.calls[stable_id] = self.calls.get(stable_id, 0) + 1
        output_signature = make_model()["observed"]["output_signature"]
        attempts: list[dict[str, Any]] = []
        attempt_no = 0
        for cold in range(cold_runs):
            for mode in ("train", "eval"):
                attempt_no += 1
                attempt = make_attempt(
                    stable_id,
                    attempt_id=f"attempt-{stable_id}-{cold}-{mode}",
                    mode=mode,
                )
                attempt["attempt_no"] = attempt_no
                attempt.pop("ledger_seq", None)
                attempt.pop("payload_sha256", None)
                attempt["worker_receipt"]["output_signature"] = output_signature
                attempts.append(attempt)
        return attempts


class FakeEnvironments(EnvironmentLane):
    """Exercise environment callbacks sequentially without conda."""

    def __init__(self, root: Path) -> None:
        """Store the fake prefix root and lifecycle event log."""

        self.root = root
        self.events: list[str] = []
        self.active = 0

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Call use under one active fake prefix and always tear down."""

        assert self.active == 0
        self.active += 1
        self.events.append(f"create:{intent.name}")
        prefix = self.root / intent.name
        prefix.mkdir(parents=True, exist_ok=True)
        try:
            use(prefix)
        finally:
            self.events.append(f"remove:{intent.name}")
            self.active -= 1
        return object()


class FakeNotifier(Notifier):
    """Capture ASCII summaries without external messaging."""

    def __init__(self) -> None:
        """Initialize the delivered message list."""

        self.messages: list[str] = []

    def notify(self, summary: str) -> bool:
        """Capture one summary and report success."""

        summary.encode("ascii")
        self.messages.append(summary)
        return True


class FakePauseScheduler(UsagePauseScheduler):
    """Use the real idempotent wakeup records with a no-op OS activator."""

    def __init__(self, root: Path) -> None:
        """Store the fake wakeup definition root and call count."""

        self.root = root
        self.calls = 0

    def schedule(
        self,
        signal: CheckerBackoffSignal,
        operational: JsonlLedger,
        context: OperationalContext,
        created_at: str,
        reset_at: str,
    ) -> None:
        """Record pause/wakeup events without installing an OS scheduler."""

        self.calls += 1
        manager = WakeupManager(
            self.root,
            operational,
            ["python", "-m", "menagerie.crawler", "run", "--resume"],
            backend=WakeupBackend.CRON,
            activator=lambda _spec: None,
        )
        manager.record_pause_and_schedule(
            provider="openai",
            observed_response=signal.response_excerpt,
            reset_at=reset_at,
            context=context,
            created_at=created_at,
        )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write small discovery inputs for an intake snapshot."""

    path.write_text("".join(json.dumps(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _snapshot(tmp_path: Path, count: int = 10) -> IntakeSnapshot:
    """Create one immutable synthetic intake snapshot."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(
        master,
        [
            {"name": f"Example{index}", "zoo": "fixtures", "variant": "base"}
            for index in range(count)
        ],
    )
    _write_jsonl(deferred, [])
    return create_intake_snapshot(master, deferred, tmp_path / "intake")


def _paths(tmp_path: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return isolated runtime and canonical ledger paths."""

    return DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            tmp_path / "records" / "models.jsonl",
            tmp_path / "records" / "attempts.jsonl",
            tmp_path / "records" / "gates.jsonl",
        ),
    )


def _driver(
    tmp_path: Path,
    snapshot: IntakeSnapshot,
    *,
    author: Optional[FakeAuthor] = None,
    checker: Optional[FakeChecker] = None,
    forward: Optional[FakeForward] = None,
    notifier: Optional[FakeNotifier] = None,
    boundary: Any = lambda _boundary, _stable_id: None,
    review_at: Optional[int] = None,
    milestones: tuple[int, ...] = (),
    pause_scheduler: Optional[FakePauseScheduler] = None,
) -> CrawlerDriver:
    """Build a fully fake deterministic driver."""

    dependencies = DriverDependencies(
        author or FakeAuthor(),
        checker or FakeChecker(),
        forward or FakeForward(),
        FakeEnvironments(tmp_path / "fake-envs"),
        notifier or FakeNotifier(),
        lambda: NOW,
        boundary,
        pause_scheduler,
    )
    return CrawlerDriver(
        _paths(tmp_path, snapshot),
        DriverConfig(
            target="osx-arm64",
            run_id="run-test",
            machine_id="machine-test",
            review_checkpoint_at=review_at,
            progress_milestones=milestones,
        ),
        dependencies,
        registry=load_environment_registry(target="osx-arm64"),
    )


@pytest.mark.parametrize(
    "boundary", ["after-author", "after-gate", "after-forward", "after-reduce"]
)
def test_resume_is_lock_safe_and_duplicate_free_at_critical_boundaries(
    tmp_path: Path, boundary: str
) -> None:
    """A kill after every durable boundary resumes the first unsatisfied identity."""

    snapshot = _snapshot(tmp_path)
    killed = False

    def kill_once(observed: str, stable_id: str) -> None:
        """Raise exactly once at the selected boundary."""

        nonlocal killed
        del stable_id
        if observed == boundary and not killed:
            killed = True
            raise InjectedKill(boundary)

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_once).run()
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 10
    assert len(scan_jsonl(paths.ledgers.attempts)) == 20
    assert len(scan_jsonl(paths.ledgers.gates)) == 1


def test_only_driver_awards_runs_after_gate_receipts_and_both_modes(tmp_path: Path) -> None:
    """Gates and worker attempts alone never create a canonical runs record."""

    snapshot = _snapshot(tmp_path)
    paths = _paths(tmp_path, snapshot)
    checker = FakeChecker()
    author = FakeAuthor()
    item_driver = _driver(tmp_path, snapshot, author=author, checker=checker)
    work = item_driver._ordered_work(snapshot, {})
    artifacts = [author.author(item, paths.work_root, item_driver.config) for item in work]
    artifact = artifacts[0]
    gate = checker.check_metadata(artifacts, paths.work_root, item_driver.config).gate
    assert gate is not None
    gate["ledger_seq"] = 1
    gate["payload_sha256"] = HASH
    attempts = FakeForward().forward(artifact, tmp_path / "env", 1, paths.work_root)
    with CanonicalReducer(paths.ledgers, [item.stable_id for item in snapshot.items]) as reducer:
        reducer.append_gate(gate)
        for attempt in attempts:
            reducer.append_attempt(attempt)
        assert reducer.current_records == {}
    result = _driver(tmp_path, snapshot, author=author, checker=checker).run()
    assert result.status == "complete"
    assert scan_jsonl(paths.ledgers.models)[0]["status"]["code"] == "runs"


def test_quota_pause_records_event_wakeup_and_no_partial_award(tmp_path: Path) -> None:
    """A checker quota signal visibly pauses, schedules once, and awards no run."""

    snapshot = _snapshot(tmp_path)
    scheduler = FakePauseScheduler(tmp_path / "wakeups")
    result = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        pause_scheduler=scheduler,
    ).run()
    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1
    paths = _paths(tmp_path, snapshot)
    assert scan_jsonl(paths.ledgers.models) == []
    events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in events] == ["usage-pause", "wakeup"]


def test_review_checkpoint_blocks_notifies_and_is_one_shot(tmp_path: Path) -> None:
    """The configured terminal count blocks until explicit sign-off, then never re-blocks."""

    snapshot = _snapshot(tmp_path, count=10)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, review_at=1).run()
    assert first.status == "paused:review-checkpoint"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 1
    events = scan_jsonl(paths.operational_ledger)
    review = next(event for event in events if event["event_kind"] == "checkpoint-review")
    assert Path(review["report_path"]).is_file()
    assert len(notifier.messages) == 1
    still_paused = _driver(tmp_path, snapshot, review_at=1).run()
    assert still_paused.status == "paused:review-checkpoint"
    resumed = _driver(tmp_path, snapshot, review_at=1).run(after_review=True)
    assert resumed.status == "complete"
    assert len(scan_jsonl(paths.ledgers.models)) == 10
    final_events = scan_jsonl(paths.operational_ledger)
    assert [event["event_kind"] for event in final_events].count("checkpoint-review") == 1
    assert [event["event_kind"] for event in final_events].count("review-signoff") == 1


def test_progress_milestones_fire_once_without_pausing(tmp_path: Path) -> None:
    """Every crossed milestone emits one event/message and resume does not re-fire it."""

    snapshot = _snapshot(tmp_path, count=10)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, milestones=(1, 2)).run()
    assert first.status == "complete"
    assert len(notifier.messages) == 2
    second_notifier = FakeNotifier()
    second = _driver(tmp_path, snapshot, notifier=second_notifier, milestones=(1, 2)).run()
    assert second.status == "complete"
    assert second_notifier.messages == []
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    assert [
        event["milestone"] for event in events if event["event_kind"] == "progress-notification"
    ] == [1, 2]


def test_empty_milestones_and_missing_notifier_never_block(tmp_path: Path) -> None:
    """No milestones means no pings; a missing notifier remains a nonfatal fallback."""

    snapshot = _snapshot(tmp_path)
    result = _driver(tmp_path, snapshot, milestones=()).run()
    assert result.status == "complete"
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    assert all(event["event_kind"] != "progress-notification" for event in events)
    assert CommandNotifier("/definitely/missing/send-to-jmt.sh").notify("milestone 1") is False
