"""Author-queue lane, typed author backoff, and the author usage-pause path.

Covers W1.3: the ``QueueAuthorLane`` file-queue RPC that bridges the engine's
subprocess author contract to an in-session Claude subagent pool, the
``AuthorBackoffSignal`` pause path that closes the ``failed:source`` quota hole
(``PLAN.md`` acceptance test 14), the LP-13.2 author effort caps, and the
transient-versus-permanent operator classification behind risk R8.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.author_dispatch import (
    AUTHOR_EXIT_BACKOFF,
    AUTHOR_EXIT_PERMANENT,
    AUTHOR_EXIT_RETRYABLE,
    AUTHOR_EXIT_UNAVAILABLE,
    AuthorBackoffSignal,
    AuthorDispatchError,
    AuthorEffortGrant,
    classify_author_response,
    parse_author_reset_at,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.checker_dispatch import classify_checker_response
from menagerie.crawler.cli import EXIT_OPERATOR_OUTAGE, main as cli_main
from menagerie.crawler.constants import (
    AUTHOR_MAX_FETCH_TARGETS,
    AUTHOR_MAX_TOOL_CALLS,
    AUTHOR_QUEUE_STALL_SECONDS,
    AUTHOR_SESSION_WALL_SECONDS,
    USAGE_LIMIT_PROVIDERS,
    AuthorPauseReason,
    CheckerPauseReason,
    OperationalEventKind,
)
import menagerie.crawler.driver as driver_module
from menagerie.crawler.driver import (
    AuthorArtifact,
    AuthorLane,
    CommandAuthorLane,
    DriverConfig,
    DriverIntegrationError,
    QueueAuthorLane,
)
from menagerie.crawler.driver_admission import classify_author_exit
from menagerie.crawler.driver_contracts import (
    AuthorBackoffError,
    AuthorEffortCapExceeded,
    AuthorQueueStalled,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakePauseScheduler,
    ScriptedAuthor,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# fake operator pool
# ---------------------------------------------------------------------------


class FakePool:
    """Service the author queue the way the managing Claude session would.

    The pool runs on the lane's poll tick, so a queue round trip is deterministic
    with no threads and no wall-clock sleeping.
    """

    def __init__(
        self,
        queue_root: Path,
        *,
        result: Optional[Mapping[str, Any]] = None,
        backoff: Optional[Mapping[str, Any]] = None,
        failure: Optional[Mapping[str, Any]] = None,
        consumption: Optional[Mapping[str, Any]] = None,
        claim_only: bool = False,
        nonce_override: Optional[str] = None,
    ) -> None:
        """Store the queue root and the scripted single response."""

        self.queue_root = queue_root
        self.result = result
        self.backoff = backoff
        self.failure = failure
        self.consumption = consumption or {
            "tool_calls": 3,
            "fetch_targets": 1,
            "wall_seconds": 12.0,
        }
        self.claim_only = claim_only
        self.nonce_override = nonce_override
        self.ticks = 0
        self.served: list[dict[str, Any]] = []

    def __call__(self, _interval: float) -> None:
        """Serve every not-yet-served pending job."""

        self.ticks += 1
        for pending in sorted((self.queue_root / "pending").glob("*.json")):
            job = json.loads(pending.read_text(encoding="utf-8"))
            if any(entry["job_id"] == job["job_id"] for entry in self.served):
                continue
            self.served.append(job)
            self._serve(job)

    def _serve(self, job: Mapping[str, Any]) -> None:
        """Publish one scripted response for a claimed job."""

        nonce = self.nonce_override or job["attempt_nonce"]
        self._write(Path(job["claim_path"]), {"job_id": job["job_id"], "attempt_nonce": nonce})
        if self.claim_only:
            return
        if self.backoff is not None:
            self._write(
                Path(job["backoff_path"]),
                {"job_id": job["job_id"], "attempt_nonce": nonce, **self.backoff},
            )
            return
        if self.failure is not None:
            self._write(
                Path(job["failure_path"]),
                {"job_id": job["job_id"], "attempt_nonce": nonce, **self.failure},
            )
            return
        assert self.result is not None
        self._write(Path(job["required_output_path"]), dict(self.result))
        self._write(
            Path(job["receipt_path"]),
            {
                "job_id": job["job_id"],
                "attempt_nonce": nonce,
                "consumption": dict(self.consumption),
            },
        )

    @staticmethod
    def _write(path: Path, payload: Mapping[str, Any]) -> None:
        """Write one JSON payload, creating parents."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload)), encoding="utf-8")


class _StepClock:
    """Monotonic clock that advances a fixed step per read."""

    def __init__(self, step: float = 1.0) -> None:
        """Initialize the elapsed counter and per-read step."""

        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        """Return the next monotonic instant."""

        value = self.now
        self.now += self.step
        return value


def _queue_lane(
    queue_root: Path,
    pool: Optional[Callable[[float], None]] = None,
    *,
    grant: Optional[AuthorEffortGrant] = None,
    stall_timeout_seconds: float = 100.0,
    monotonic_step: float = 1.0,
) -> QueueAuthorLane:
    """Build a deterministic queue lane wired to a fake pool."""

    return QueueAuthorLane(
        queue_root,
        effort_grant=grant,
        stall_timeout_seconds=stall_timeout_seconds,
        poll_interval_seconds=0.001,
        monotonic=_StepClock(monotonic_step),
        sleep=pool or (lambda _interval: None),
        nonce_factory=lambda: "nonce-fixed",
    )


def _source_targets(count: int = 1) -> dict[str, Any]:
    """Return a source-target payload with ``count`` pinned entries."""

    return {
        "sources": [
            {
                "source_id": f"src-{index}",
                "url": f"https://example.invalid/{index}",
                "revision": "main",
                "expected_sha256": "sha256:" + "0" * 64,
                "media_type": "text/plain",
            }
            for index in range(count)
        ]
    }


def _dispatch_source_request(
    lane: QueueAuthorLane,
    item: WorkItem,
    root: Path,
) -> None:
    """Run one raw queue round trip for the source-request envelope."""

    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request.json"
    request_path.write_text(json.dumps({"envelope_version": "test"}), encoding="utf-8")
    lane._dispatch(  # noqa: SLF001 -- exercising the queue transport directly
        kind="source-request",
        item=item,
        config=None,
        work_id=f"work-{item.stable_id}",
        request_path=request_path,
        output_path=root / "source-targets.json",
    )


def _stub_controlled_fetch(monkeypatch: Any) -> None:
    """Replace the controlled fetch with a deterministic offline manifest."""

    monkeypatch.setattr(
        driver_module,
        "fetch_targets",
        lambda targets, root: {
            "sources": [
                {"source_id": target.source_id, "url": target.url, "cas_path": str(root)}
                for target in targets
            ]
        },
    )


def _work_item(tmp_path: Path) -> WorkItem:
    """Return one routed synthetic work item."""

    snapshot = _snapshot(tmp_path, count=1)
    return _driver(tmp_path, snapshot)._ordered_work(snapshot, {})[0]  # noqa: SLF001


# ---------------------------------------------------------------------------
# queue transport
# ---------------------------------------------------------------------------


def test_queue_lane_publishes_a_job_the_pool_can_serve(tmp_path: Path) -> None:
    """The lane enqueues one closed job descriptor and consumes the pool's result."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, result=_source_targets())
    lane = _queue_lane(queue_root, pool)

    _dispatch_source_request(lane, item, tmp_path / "work")

    assert len(pool.served) == 1
    job = pool.served[0]
    assert job["envelope_version"] == "menagerie.crawler.author-queue-job.v1"
    assert job["kind"] == "source-request"
    assert job["stable_id"] == item.stable_id
    assert Path(job["request_path"]).is_absolute()
    assert Path(job["required_output_path"]).is_absolute()
    assert job["effort_grant"] == {
        "tool_calls": AUTHOR_MAX_TOOL_CALLS,
        "fetch_targets": AUTHOR_MAX_FETCH_TARGETS,
        "wall_seconds": float(AUTHOR_SESSION_WALL_SECONDS),
    }
    assert job["stall_timeout_seconds"] == 100.0
    # The queue's own control files never survive a completed round trip.
    assert list((queue_root / "pending").glob("*.json")) == []
    assert list((queue_root / "receipts").glob("*.json")) == []
    assert Path(job["required_output_path"]).is_file()


def test_queue_lane_rejects_completion_without_a_published_result(tmp_path: Path) -> None:
    """A receipt without the exact required output is a protocol violation."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"

    class ReceiptOnlyPool(FakePool):
        """Publish a receipt but never the result."""

        def _serve(self, job: Mapping[str, Any]) -> None:
            """Write only the completion receipt."""

            self._write(
                Path(job["receipt_path"]),
                {
                    "job_id": job["job_id"],
                    "attempt_nonce": job["attempt_nonce"],
                    "consumption": dict(self.consumption),
                },
            )

    pool = ReceiptOnlyPool(queue_root, result=_source_targets())
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(DriverIntegrationError, match="reported completion without a result"):
        _dispatch_source_request(lane, item, tmp_path / "work")


def test_queue_lane_ignores_a_superseded_attempt_receipt(tmp_path: Path) -> None:
    """A late file from an earlier attempt is never mistaken for this answer."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, result=_source_targets(), nonce_override="stale-nonce")
    lane = _queue_lane(queue_root, pool, stall_timeout_seconds=5.0)

    with pytest.raises(AuthorQueueStalled):
        _dispatch_source_request(lane, item, tmp_path / "work")


# ---------------------------------------------------------------------------
# stall guard (risk R6)
# ---------------------------------------------------------------------------


def test_queue_stall_is_retryable_infrastructure_not_a_model_failure(tmp_path: Path) -> None:
    """A dead managing session looks like stalled infrastructure, never a failed model."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    lane = _queue_lane(queue_root, None, stall_timeout_seconds=3.0)

    with pytest.raises(AuthorQueueStalled, match="never claimed") as raised:
        _dispatch_source_request(lane, item, tmp_path / "work")

    assert isinstance(raised.value, RetryableOperatorError)
    markers = tuple((queue_root / "watchdog").glob("*.stall.json"))
    assert len(markers) == 1
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    assert marker["stable_id"] == item.stable_id
    assert marker["claimed"] is False
    driver = _driver(tmp_path, _snapshot(tmp_path, count=1))
    assert driver._is_infrastructure_error(raised.value) is True  # noqa: SLF001


def test_queue_stall_reports_a_claimed_but_unfinished_job(tmp_path: Path) -> None:
    """A claimed-then-silent job is observably distinct from an unserviced queue."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, claim_only=True)
    lane = _queue_lane(queue_root, pool, stall_timeout_seconds=4.0)

    with pytest.raises(AuthorQueueStalled, match="claimed but unfinished"):
        _dispatch_source_request(lane, item, tmp_path / "work")


def test_queue_stall_default_matches_the_planned_forty_five_minute_deadline() -> None:
    """The shipped stall deadline is the planned 45 minute outer bound."""

    assert AUTHOR_QUEUE_STALL_SECONDS == 45 * 60


# ---------------------------------------------------------------------------
# typed operator failures (risk R8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retryable", [True, False])
def test_pool_failure_signal_honors_its_declared_retryability(
    tmp_path: Path, retryable: bool
) -> None:
    """A declared-transient failure is retryable; a declared-permanent one is not."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(
        queue_root,
        failure={"retryable": retryable, "reason": "operator-crash", "detail": "boom"},
    )
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(DriverIntegrationError) as raised:
        _dispatch_source_request(lane, item, tmp_path / "work")

    assert isinstance(raised.value, RetryableOperatorError) is retryable
    driver = _driver(tmp_path, _snapshot(tmp_path, count=1))
    assert driver._is_infrastructure_error(raised.value) is retryable  # noqa: SLF001


def test_pool_failure_without_a_classification_is_refused(tmp_path: Path) -> None:
    """An unclassified failure signal is a protocol violation, not a silent guess."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, failure={"reason": "operator-crash", "detail": "boom"})
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(DriverIntegrationError, match="boolean retryable classification"):
        _dispatch_source_request(lane, item, tmp_path / "work")


@pytest.mark.parametrize("exit_code", [AUTHOR_EXIT_RETRYABLE, AUTHOR_EXIT_UNAVAILABLE])
def test_command_lane_transient_exit_is_retryable(exit_code: int) -> None:
    """Operator exits 75/78 are transient and must not fail the model."""

    with pytest.raises(RetryableOperatorError):
        classify_author_exit("author", "m_x", exit_code, "", "upstream unavailable")


def test_command_lane_permanent_exit_is_not_retried(tmp_path: Path) -> None:
    """Operator exit 64 is a declared contract rejection with no retry."""

    with pytest.raises(DriverIntegrationError) as raised:
        classify_author_exit("author", "m_x", AUTHOR_EXIT_PERMANENT, "", "bad envelope")

    assert not isinstance(raised.value, RetryableOperatorError)
    driver = _driver(tmp_path, _snapshot(tmp_path, count=1))
    assert driver._is_infrastructure_error(raised.value) is False  # noqa: SLF001


def test_command_lane_unclassified_exit_keeps_its_retryable_prefix(tmp_path: Path) -> None:
    """An unknown nonzero exit stays one retryable transport failure, as before."""

    with pytest.raises(DriverIntegrationError, match="author command failed") as raised:
        classify_author_exit("author", "m_x", 1, "stdout tail", "")

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1))
    assert driver._is_infrastructure_error(raised.value) is True  # noqa: SLF001


def test_command_lane_reads_quota_text_from_stdout(tmp_path: Path, monkeypatch: Any) -> None:
    """Structured provider errors on stdout are not masked by a nonempty stderr."""

    item = _work_item(tmp_path)

    def fake_run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Report quota exhaustion on stdout with unrelated stderr noise."""

        del kwargs
        return subprocess.CompletedProcess(
            list(argv),
            1,
            "Claude usage limit reached. try again at 2026-07-27T18:00:00Z.",
            "Reading additional input from stdin...",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    lane = CommandAuthorLane(("fake-author",))

    with pytest.raises(AuthorBackoffError) as raised:
        lane._fetch_author_sources(item, tmp_path / "author")  # noqa: SLF001

    assert raised.value.signal.reason is AuthorPauseReason.QUOTA_EXHAUSTED
    assert raised.value.signal.provider == "anthropic"
    assert raised.value.signal.reset_at == "2026-07-27T18:00:00Z"


# ---------------------------------------------------------------------------
# effort caps (PLAN.md LP-13.2, PLAN_RECONCILED 3.4)
# ---------------------------------------------------------------------------


def test_fetch_target_cap_is_enforced_by_the_lane(tmp_path: Path) -> None:
    """The lane refuses more pinned sources than its controlled-fetch grant."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, result=_source_targets(count=AUTHOR_MAX_FETCH_TARGETS + 1))
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(AuthorEffortCapExceeded, match="controlled-fetch grant"):
        lane._fetch_author_sources(item, tmp_path / "work" / "author")  # noqa: SLF001


def test_fetch_target_cap_admits_the_full_grant(tmp_path: Path, monkeypatch: Any) -> None:
    """Exactly the granted number of targets is inside the cap, not over it."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, result=_source_targets(count=AUTHOR_MAX_FETCH_TARGETS))
    lane = _queue_lane(queue_root, pool)
    _stub_controlled_fetch(monkeypatch)

    manifest = lane._fetch_author_sources(item, tmp_path / "work" / "author")  # noqa: SLF001

    assert len(manifest["sources"]) == AUTHOR_MAX_FETCH_TARGETS


def test_source_request_publishes_the_fetch_grant_to_the_operator(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The operator is told its controlled-fetch ceiling, not left to guess it."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, result=_source_targets())
    lane = _queue_lane(queue_root, pool)
    root = tmp_path / "work" / "author"
    _stub_controlled_fetch(monkeypatch)

    lane._fetch_author_sources(item, root)  # noqa: SLF001

    request = json.loads((root / "source-request.json").read_text(encoding="utf-8"))
    assert request["max_sources"] == AUTHOR_MAX_FETCH_TARGETS


@pytest.mark.parametrize(
    "metric, value",
    [
        ("tool_calls", AUTHOR_MAX_TOOL_CALLS + 1),
        ("fetch_targets", AUTHOR_MAX_FETCH_TARGETS + 1),
        ("wall_seconds", float(AUTHOR_SESSION_WALL_SECONDS) + 1.0),
    ],
)
def test_declared_consumption_over_grant_is_a_cap_failure(
    tmp_path: Path, metric: str, value: float
) -> None:
    """The lane audits what the pool declares it spent against the published grant."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    consumption = {"tool_calls": 1, "fetch_targets": 1, "wall_seconds": 1.0, metric: value}
    pool = FakePool(queue_root, result=_source_targets(), consumption=consumption)
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(AuthorEffortCapExceeded, match=metric):
        _dispatch_source_request(lane, item, tmp_path / "work")


def test_receipt_without_declared_consumption_is_refused(tmp_path: Path) -> None:
    """An operator that does not declare its spend cannot be audited, so it is refused."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"

    class SilentPool(FakePool):
        """Publish a result and a receipt that declares nothing."""

        def _serve(self, job: Mapping[str, Any]) -> None:
            """Write the result and a consumption-free receipt."""

            assert self.result is not None
            self._write(Path(job["required_output_path"]), dict(self.result))
            self._write(
                Path(job["receipt_path"]),
                {"job_id": job["job_id"], "attempt_nonce": job["attempt_nonce"]},
            )

    lane = _queue_lane(queue_root, SilentPool(queue_root, result=_source_targets()))

    with pytest.raises(DriverIntegrationError, match="omits declared effort consumption"):
        _dispatch_source_request(lane, item, tmp_path / "work")


def test_effort_grant_defaults_match_the_plan() -> None:
    """The shipped default grant is PLAN.md LP-13.2's 30/20/30-minute ceiling."""

    grant = AuthorEffortGrant()

    assert (grant.tool_calls, grant.fetch_targets) == (30, 20)
    assert grant.wall_seconds == 30 * 60
    with pytest.raises(AuthorDispatchError):
        AuthorEffortGrant(tool_calls=0)


# ---------------------------------------------------------------------------
# typed backoff classification
# ---------------------------------------------------------------------------


def test_classify_author_response_detects_quota_and_extracts_a_reset(tmp_path: Path) -> None:
    """A usage-limit response becomes a quota pause carrying its reset instant."""

    del tmp_path
    signal = classify_author_response(
        1, "Claude usage limit reached. try again at 2026-07-27T18:00:00Z."
    )

    assert signal is not None
    assert signal.reason is AuthorPauseReason.QUOTA_EXHAUSTED
    assert signal.provider == "anthropic"
    assert signal.reset_at == "2026-07-27T18:00:00Z"


def test_classify_author_response_falls_back_when_no_reset_is_named() -> None:
    """An unparseable reset leaves ``reset_at`` unset for the one-hour re-check."""

    signal = classify_author_response(AUTHOR_EXIT_BACKOFF, "429 rate limit; slow down")

    assert signal is not None
    assert signal.reason is AuthorPauseReason.RATE_LIMIT
    assert signal.reset_at is None
    assert parse_author_reset_at("no reset named here") is None


def test_classify_author_response_ignores_ordinary_failures() -> None:
    """A plain contract error is not a pause signal."""

    assert classify_author_response(1, "invalid_request_error: bad schema") is None


def test_checker_backoff_still_reports_openai() -> None:
    """Deriving the provider from the signal did not move the checker off openai."""

    signal = classify_checker_response(429, "rate limit exceeded")

    assert signal is not None
    assert signal.provider == "openai"
    assert signal.reason is CheckerPauseReason.RATE_LIMIT
    assert {"anthropic", "openai"} == set(USAGE_LIMIT_PROVIDERS)


def test_queue_backoff_sidecar_rejects_a_foreign_provider(tmp_path: Path) -> None:
    """A pool cannot invent a provider outside the closed usage-limit vocabulary."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(queue_root, backoff={"provider": "acme", "reason": "quota-exhausted"})
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(DriverIntegrationError, match="unsupported usage-limit provider"):
        _dispatch_source_request(lane, item, tmp_path / "work")


def test_queue_backoff_sidecar_becomes_a_typed_author_pause(tmp_path: Path) -> None:
    """The pool's backoff sidecar surfaces as ``AuthorBackoffError``, not a failure."""

    item = _work_item(tmp_path)
    queue_root = tmp_path / "author-queue"
    pool = FakePool(
        queue_root,
        backoff={
            "provider": "anthropic",
            "reason": "quota-exhausted",
            "reset_at": "2026-07-27T18:00:00Z",
            "response_excerpt": "Claude usage limit reached",
        },
    )
    lane = _queue_lane(queue_root, pool)

    with pytest.raises(AuthorBackoffError) as raised:
        _dispatch_source_request(lane, item, tmp_path / "work")

    signal = raised.value.signal
    assert signal.reason is AuthorPauseReason.QUOTA_EXHAUSTED
    assert signal.provider == "anthropic"
    assert signal.reset_at == "2026-07-27T18:00:00Z"
    # The pause unwinds the job cleanly so a resumed attempt is not served a stale one.
    assert list((queue_root / "pending").glob("*.json")) == []


# ---------------------------------------------------------------------------
# the driver pause path (PLAN.md acceptance test 14)
# ---------------------------------------------------------------------------


class BackoffAuthor(AuthorLane):
    """Author lane that reports one provider pause instead of a proposal."""

    def __init__(self, signal: AuthorBackoffSignal) -> None:
        """Store the pause signal and the call counter."""

        self.signal = signal
        self.calls = 0

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise the typed pause rather than returning an artifact."""

        del item, work_root, config, context
        self.calls += 1
        raise AuthorBackoffError(self.signal)


def test_claude_quota_pauses_the_campaign_instead_of_failing_the_model(
    tmp_path: Path,
) -> None:
    """Acceptance 14: Claude usage exhaustion pauses with a reset time, never fails a model."""

    snapshot = _snapshot(tmp_path, count=1)
    author = BackoffAuthor(
        AuthorBackoffSignal(
            reason=AuthorPauseReason.QUOTA_EXHAUSTED,
            retry_after_seconds=None,
            reset_at="2026-07-27T18:00:00Z",
            response_excerpt="Claude usage limit reached",
            provider="anthropic",
        )
    )
    paths = _paths(tmp_path, snapshot)
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        pause_scheduler=FakePauseScheduler(paths.wakeup_root),
    ).run()

    assert result.status == "paused:usage-limit"
    assert result.paused_reason == AuthorPauseReason.QUOTA_EXHAUSTED.value
    # No model record was created at all; the quota hole used to write failed:source here.
    assert scan_jsonl(paths.ledgers.models) == []
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state == {
        "status": "paused:usage-limit",
        "provider": "anthropic",
        "reset_at": "2026-07-27T18:00:00Z",
    }


def test_author_pause_records_an_anthropic_usage_event_and_wake_episode(
    tmp_path: Path,
) -> None:
    """The recorded pause names anthropic and schedules its reset wake episode."""

    snapshot = _snapshot(tmp_path, count=1)
    author = BackoffAuthor(
        AuthorBackoffSignal(
            reason=AuthorPauseReason.QUOTA_EXHAUSTED,
            retry_after_seconds=None,
            reset_at="2026-07-27T18:00:00Z",
            response_excerpt="Claude usage limit reached",
            provider="anthropic",
        )
    )
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    _driver(tmp_path, snapshot, author=author, pause_scheduler=scheduler).run()

    assert scheduler.calls == 1
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == OperationalEventKind.USAGE_PAUSE.value
    ]
    assert [event["provider"] for event in events] == ["anthropic"]
    assert [event["reset_at"] for event in events] == ["2026-07-27T18:00:00Z"]


def test_author_pause_without_a_reset_falls_back_to_a_one_hour_recheck(
    tmp_path: Path,
) -> None:
    """An unobserved reset schedules a re-check instead of blocking the campaign."""

    snapshot = _snapshot(tmp_path, count=1)
    author = BackoffAuthor(
        AuthorBackoffSignal(
            reason=AuthorPauseReason.RATE_LIMIT,
            retry_after_seconds=None,
            reset_at=None,
            response_excerpt="rate limit",
            provider="anthropic",
        )
    )
    paths = _paths(tmp_path, snapshot)
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        pause_scheduler=FakePauseScheduler(paths.wakeup_root),
    ).run()

    assert result.status == "paused:usage-limit"
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["provider"] == "anthropic"
    assert state["reset_at"] != ""


class FlakyAuthor(ScriptedAuthor):
    """Fail transiently on the first call, then author normally."""

    def __init__(self) -> None:
        """Initialize the injected-fault counter."""

        super().__init__()
        self.transient_faults = 0

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise one retryable operator failure before succeeding."""

        if self.transient_faults == 0:
            self.transient_faults += 1
            raise RetryableOperatorError("author queue job stalled once")
        return super().author(item, work_root, config, context)


def test_injected_transient_author_fault_never_becomes_a_permanent_failure(
    tmp_path: Path,
) -> None:
    """R8 gate: an injected transient is retried and creates no permanent failure."""

    snapshot = _snapshot(tmp_path, count=1)
    author = FlakyAuthor()
    paths = _paths(tmp_path, snapshot)

    _driver(tmp_path, snapshot, author=author).run()

    assert author.transient_faults == 1
    statuses = [record["status"]["code"] for record in scan_jsonl(paths.ledgers.models)]
    assert statuses
    assert "failed:source" not in statuses


class StalledAuthor(AuthorLane):
    """Always report a dead managing-session queue."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise retryable infrastructure without producing model authority."""

        del item, work_root, config, context
        raise AuthorQueueStalled("author queue stalled beyond 45 minutes")


def test_exhausted_queue_stall_surfaces_as_campaign_infrastructure(
    tmp_path: Path,
) -> None:
    """Two bounded transport attempts never turn the stalled model into a failure."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)

    with pytest.raises(AuthorQueueStalled):
        _driver(tmp_path, snapshot, author=StalledAuthor()).run()

    assert scan_jsonl(paths.ledgers.models) == []
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["status"] == "retryable:infrastructure"


def test_cli_maps_queue_stall_to_operator_outage_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The supervisor receives exit 6 for a dead managing session."""

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1), author=StalledAuthor())

    assert (
        cli_main(
            ["--repo-root", str(tmp_path), "run"],
            driver_factory=lambda _args: driver,
        )
        == EXIT_OPERATOR_OUTAGE
    )
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "retryable-infrastructure"


class CapExhaustedAuthor(AuthorLane):
    """Author lane whose session exceeds its published effort grant."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise the typed cap failure instead of returning an artifact."""

        del item, work_root, config, context
        raise AuthorEffortCapExceeded("author session consumed tool_calls 31, exceeding its 30")


def test_effort_cap_exhaustion_records_its_own_reason_code(tmp_path: Path) -> None:
    """LP-13.2: cap exhaustion is failed:source/effort-cap-exhausted, not identity-unresolved."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)

    _driver(tmp_path, snapshot, author=CapExhaustedAuthor()).run()

    records = scan_jsonl(paths.ledgers.models)
    assert [record["status"]["code"] for record in records] == ["failed:source"]
    assert [record["status"]["reason_code"] for record in records] == ["effort-cap-exhausted"]


def test_cli_selects_the_queue_lane_and_drops_the_author_command_requirement(
    tmp_path: Path,
) -> None:
    """``--author-queue`` selects the in-session pool bridge over a wrapper command."""

    import argparse

    from menagerie.crawler.cli import _optional_author_queue_root

    assert _optional_author_queue_root(argparse.Namespace(author_queue=None)) is None
    assert _optional_author_queue_root(argparse.Namespace(author_queue="  ")) is None
    resolved = _optional_author_queue_root(
        argparse.Namespace(author_queue=str(tmp_path / "author-queue"))
    )
    assert resolved == (tmp_path / "author-queue").resolve()
