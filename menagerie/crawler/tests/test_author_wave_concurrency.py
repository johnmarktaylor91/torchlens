"""Bounded concurrent author sessions with a single serial canonical writer.

The author lane is roughly three quarters of the campaign's projected work, and a
serial lane capped a four-campaign fleet at four concurrent sessions -- structurally
below the reconciled schedule. ``_AuthorWavePool`` overlaps the *sessions* only.

These tests are evidence, not assertion. They prove, in order:

* sessions genuinely overlap in wall-clock time, and never exceed the configured
  bound (a rendezvous that cannot complete under a serial lane, plus measured
  interval overlap);
* one sibling failing or pausing neither cancels nor corrupts the others, and a
  paused wave preserves every already-finished sibling instead of discarding it;
* the wave's canonical output is identical to serial execution even when sessions
  finish in exactly reverse order, including family-variant derivation.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    MAX_AUTHOR_WAVE_CONCURRENCY,
)
from menagerie.crawler.driver import AuthorArtifact, DriverConfig, QueueAuthorLane
from menagerie.crawler.driver_contracts import (
    AuthorBackoffError,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.author_dispatch import AuthorBackoffSignal
from menagerie.crawler.constants import AuthorPauseReason
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakePauseScheduler,
    ForwardScript,
    ScriptedAuthor,
    ScriptedForward,
    _driver,
    _family_snapshot,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# instrumented author lanes
# ---------------------------------------------------------------------------


@dataclass
class _Interval:
    """One author session's observed wall-clock occupancy."""

    stable_id: str
    started: float
    finished: float


class ObservedAuthor(ScriptedAuthor):
    """Canonical synthetic author that records each session's start and end.

    ``hold`` runs while the session is "in progress", so it is the exact window a
    concurrent lane is supposed to overlap. Intervals are appended under a lock and
    are the test's primary evidence: overlap is measured, never inferred from the
    number of results that came back.
    """

    def __init__(
        self,
        *,
        hold: Optional[Callable[[str], None]] = None,
        after: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Bind the in-session hold, the post-completion hook, and observation state.

        ``after`` fires once this session's completion has been recorded, which is
        what lets a test CHAIN sessions into an exact completion order instead of
        hoping a sleep produces one.
        """

        super().__init__(**kwargs)
        self._hold = hold or (lambda _stable_id: None)
        self._after = after or (lambda _stable_id: None)
        self._lock = threading.Lock()
        self._live = 0
        self.intervals: list[_Interval] = []
        self.peak_concurrency = 0
        self.completed: list[str] = []

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Author one model while recording its occupancy window."""

        started = time.monotonic()
        with self._lock:
            self._live += 1
            self.peak_concurrency = max(self.peak_concurrency, self._live)
        try:
            self._hold(item.stable_id)
            return super().author(item, work_root, config, context)
        finally:
            with self._lock:
                self._live -= 1
                self.intervals.append(_Interval(item.stable_id, started, time.monotonic()))
                self.completed.append(item.stable_id)
            # Strictly after the completion is recorded, so a waiter released here
            # can never race ahead of the record it is waiting on.
            self._after(item.stable_id)


def _max_overlap(intervals: list[_Interval]) -> int:
    """Return the largest number of intervals that were open simultaneously."""

    edges = sorted(
        [(interval.started, 1) for interval in intervals]
        + [(interval.finished, -1) for interval in intervals]
    )
    live = 0
    peak = 0
    for _instant, delta in edges:
        live += delta
        peak = max(peak, live)
    return peak


# ---------------------------------------------------------------------------
# 1. sessions genuinely overlap, within the configured bound
# ---------------------------------------------------------------------------


def test_author_sessions_overlap_up_to_the_configured_bound(tmp_path: Path) -> None:
    """Exactly ``degree`` sessions must be able to rendezvous inside one wave.

    The hold is a barrier of width ``degree``. Under the historical serial lane the
    first session waits for peers that cannot exist and the barrier breaks, so this
    test cannot pass without real overlap. The measured interval overlap is then
    asserted independently, and is also asserted not to exceed the bound.
    """

    degree = 4
    models = 8
    barrier = threading.Barrier(degree, timeout=30.0)
    snapshot = _snapshot(tmp_path, count=models)

    def rendezvous(_stable_id: str) -> None:
        """Hold this session until ``degree`` peers are live at the same instant."""

        barrier.wait()

    author = ObservedAuthor(hold=rendezvous)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=degree,
    ).run()

    assert result.status == "complete"
    assert sorted(author.calls) == sorted(item.stable_id for item in snapshot.items)
    # Structural evidence: the rendezvous completed, so `degree` sessions were live
    # at the same instant. Measured evidence: the recorded windows overlap.
    assert author.peak_concurrency == degree
    assert _max_overlap(author.intervals) == degree
    # And the bound is a bound: never more than `degree` sessions in flight.
    assert author.peak_concurrency <= degree


def test_serial_configuration_keeps_exactly_one_session_in_flight(tmp_path: Path) -> None:
    """``author_concurrency=1`` must restore the historical fully serial lane."""

    snapshot = _snapshot(tmp_path, count=4)
    author = ObservedAuthor(hold=lambda _stable_id: time.sleep(0.02))

    result = _driver(tmp_path, snapshot, author=author, author_concurrency=1).run()

    assert result.status == "complete"
    assert author.peak_concurrency == 1
    assert _max_overlap(author.intervals) == 1
    # A serial lane also completes in exactly scheduled order.
    assert author.completed == [item.stable_id for item in snapshot.items]


def test_author_concurrency_is_a_validated_closed_range() -> None:
    """The fan-out bound must be an explicit, validated number, never unbounded."""

    assert DriverConfig().author_concurrency == DEFAULT_AUTHOR_WAVE_CONCURRENCY
    with pytest.raises(ValueError, match="author_concurrency"):
        DriverConfig(author_concurrency=0)
    with pytest.raises(ValueError, match="author_concurrency"):
        DriverConfig(author_concurrency=MAX_AUTHOR_WAVE_CONCURRENCY + 1)


# ---------------------------------------------------------------------------
# 2. failure isolation
# ---------------------------------------------------------------------------


class SiblingFailureAuthor(ObservedAuthor):
    """Fail exactly one model while its siblings are still in flight."""

    def __init__(self, failing_id: str, error: BaseException, **kwargs: Any) -> None:
        """Bind the single failing model and the typed exception it raises."""

        super().__init__(**kwargs)
        self._failing_id = failing_id
        self._error = error

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise for the designated model, author every other one normally."""

        if item.stable_id == self._failing_id:
            with self._lock:
                self.completed.append(item.stable_id)
            raise self._error
        return super().author(item, work_root, config, context)


class ResearchToolsFailureAuthor(ObservedAuthor):
    """Report the queue protocol's typed research-tool failure for selected models."""

    def __init__(self, failing_ids: set[str], **kwargs: Any) -> None:
        """Bind the models whose research sessions cannot reach their tools."""

        super().__init__(**kwargs)
        self._failing_ids = failing_ids

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise the exact error produced by a retryable queue failure sidecar."""

        if item.stable_id not in self._failing_ids:
            return super().author(item, work_root, config, context)
        del work_root, config, context
        with self._lock:
            self.completed.append(item.stable_id)
        lane = object.__new__(QueueAuthorLane)
        lane._raise_operator_failure(  # noqa: SLF001
            SimpleNamespace(job_id=f"author-{item.stable_id}", stable_id=item.stable_id),
            {
                "reason": "research-tools-unavailable",
                "retryable": True,
                "detail": "Exa MCP provider is unreachable",
            },
        )
        raise AssertionError("queue failure classifier did not raise")


def test_one_failing_sibling_does_not_take_down_the_rest_of_the_wave(tmp_path: Path) -> None:
    """A per-model author failure must terminalize only its own model.

    Every sibling still runs, still reaches ``runs``, and the failing model records
    exactly the ``failed:source`` terminal a serial lane would have recorded.
    """

    snapshot = _snapshot(tmp_path, count=6)
    failing_id = snapshot.items[2].stable_id
    gate = threading.Barrier(3, timeout=30.0)

    def hold(stable_id: str) -> None:
        """Keep the first few siblings in flight while the failure lands."""

        if stable_id != failing_id:
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                return

    author = SiblingFailureAuthor(
        failing_id,
        RuntimeError("synthetic single-sibling author failure"),
        hold=hold,
    )

    result = _driver(tmp_path, snapshot, author=author, author_concurrency=4).run()

    # One model failed on its own, so the partition is complete-with-terminals; the
    # run did not abort and no sibling inherited the failure.
    assert result.status == "terminal-partition-complete"
    current = {
        str(model["stable_id"]): model
        for model in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert current[failing_id]["status"]["kind"].startswith("failed")
    survivors = [item.stable_id for item in snapshot.items if item.stable_id != failing_id]
    assert survivors, "the fixture must contain siblings for the failure to spare"
    for stable_id in survivors:
        assert current[stable_id]["status"]["kind"] == "runs", stable_id
        # A sibling's author session ran to completion; it was never cancelled.
        assert stable_id in author.calls


def test_one_retryable_sibling_still_lets_its_peers_finish_their_sessions(
    tmp_path: Path,
) -> None:
    """A retryable operator failure aborts the wave without orphaning siblings.

    The abort is the historical one -- ``RetryableOperatorError`` unwinds the wave --
    but every session already started is drained, so nothing is left running behind
    the driver and no sibling's provider spend is thrown away.
    """

    snapshot = _snapshot(tmp_path, count=4)
    failing_id = snapshot.items[0].stable_id
    author = SiblingFailureAuthor(
        failing_id,
        RetryableOperatorError("author command failed for synthetic transport"),
        hold=lambda _stable_id: time.sleep(0.02),
    )

    with pytest.raises(RetryableOperatorError):
        _driver(tmp_path, snapshot, author=author, author_concurrency=4).run()

    # Every session the pool started reached a terminal state before the abort
    # propagated: `completed` is appended in the lane's own `finally`, so an
    # orphaned or cancelled sibling would be missing from it. The failing model
    # appears twice because the retryable classification still gets its one
    # bounded infrastructure retry, exactly as under a serial lane.
    assert author.completed.count(failing_id) == 2
    assert set(author.calls) <= set(author.completed)
    work_root = _paths(tmp_path, snapshot).work_root
    for stable_id in author.calls:
        # A drained sibling's validated result is preserved, never discarded.
        assert (work_root / stable_id / "driver-author-artifact.json").is_file()


def test_one_research_tool_failure_remains_retryable_without_pausing(
    tmp_path: Path,
) -> None:
    """One model's tool failure is typed and retryable, not a campaign pause."""

    snapshot = _snapshot(tmp_path, count=4)
    failing_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    author = ResearchToolsFailureAuthor(
        {failing_id},
        hold=lambda _stable_id: time.sleep(0.02),
    )

    with pytest.raises(RetryableOperatorError) as raised:
        _driver(tmp_path, snapshot, author=author, author_concurrency=4).run()

    assert type(raised.value).__name__ == "ResearchToolsUnavailableError"
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["status"] == "retryable:infrastructure"
    assert not state["status"].startswith("paused:")
    assert scan_jsonl(paths.ledgers.models) == []


def test_three_consecutive_research_tool_failures_pause_with_actionable_reason(
    tmp_path: Path,
) -> None:
    """Three different failed models establish an outage and pause the campaign."""

    snapshot = _snapshot(tmp_path, count=4)
    failing_ids = {item.stable_id for item in snapshot.items[:3]}
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    result = _driver(
        tmp_path,
        snapshot,
        author=ResearchToolsFailureAuthor(failing_ids),
        author_concurrency=4,
        pause_scheduler=scheduler,
    ).run()

    assert result.status == "paused:usage-limit"
    assert result.paused_reason == "research-tools-unavailable"
    assert scheduler.calls == 1
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["provider"] == "research-tools"
    assert state["reason"] == "research-tools-unavailable"
    assert "restore" in state["detail"].lower()
    assert "research" in state["detail"].lower()
    assert scan_jsonl(paths.ledgers.models) == []


def test_research_outage_pause_keeps_successes_and_failed_models_retryable(
    tmp_path: Path,
) -> None:
    """Resume reuses successful sibling work and retries every outage-affected model."""

    snapshot = _snapshot(tmp_path, count=5)
    failing_ids = {item.stable_id for item in snapshot.items[:3]}
    successful_ids = {item.stable_id for item in snapshot.items[3:]}
    paths = _paths(tmp_path, snapshot)
    paused_author = ResearchToolsFailureAuthor(
        failing_ids,
        hold=lambda _stable_id: time.sleep(0.02),
    )

    paused = _driver(
        tmp_path,
        snapshot,
        author=paused_author,
        author_concurrency=5,
        pause_scheduler=FakePauseScheduler(paths.wakeup_root),
    ).run()
    assert paused.status == "paused:usage-limit"
    for stable_id in successful_ids:
        assert (paths.work_root / stable_id / "driver-author-artifact.json").is_file()

    resumed_author = ObservedAuthor()
    resumed = _driver(
        tmp_path,
        snapshot,
        author=resumed_author,
        author_concurrency=5,
    ).run()

    assert resumed.status == "complete"
    assert failing_ids <= set(resumed_author.calls)
    assert successful_ids.isdisjoint(resumed_author.calls)


class PausingAuthor(ObservedAuthor):
    """Report a provider usage pause for exactly one model."""

    def __init__(self, pausing_id: str, **kwargs: Any) -> None:
        """Bind the single model whose session hits the provider quota."""

        super().__init__(**kwargs)
        self._pausing_id = pausing_id

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Raise a typed backoff for the designated model only."""

        if item.stable_id == self._pausing_id:
            with self._lock:
                self.completed.append(item.stable_id)
            raise AuthorBackoffError(
                AuthorBackoffSignal(
                    reason=AuthorPauseReason.QUOTA_EXHAUSTED,
                    retry_after_seconds=None,
                    reset_at="2026-07-28T18:00:00Z",
                    response_excerpt="Claude usage limit reached",
                    provider="anthropic",
                )
            )
        return super().author(item, work_root, config, context)


def test_usage_pause_with_siblings_in_flight_pauses_without_discarding_them(
    tmp_path: Path,
) -> None:
    """A quota pause must reach the driver and preserve every finished sibling.

    Decision under N in flight: the pause keeps its historical meaning -- the first
    pausing model *in work order* records the pause once and unwinds the wave, so the
    canonical ledger sequence is byte-identical to serial. What changes is that no
    started session is cancelled and no finished session is thrown away: siblings are
    drained and their validated results are written to the disposable per-model
    cache, so the resume after the reset reloads them instead of re-authoring.
    """

    snapshot = _snapshot(tmp_path, count=5)
    pausing_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    author = PausingAuthor(pausing_id, hold=lambda _stable_id: time.sleep(0.02))

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=4,
        pause_scheduler=scheduler,
    ).run()

    assert result.status == "paused:usage-limit"
    # The pause reached the driver exactly once, from the first model in work order.
    assert scheduler.calls == 1

    work_root = paths.work_root
    dispatched_siblings = [
        stable_id for stable_id in author.completed if stable_id != pausing_id
    ]
    assert dispatched_siblings, "the fixture must keep siblings in flight at pause time"
    for stable_id in dispatched_siblings:
        cache = work_root / stable_id / "driver-author-artifact.json"
        assert cache.is_file(), f"paused wave discarded a finished sibling: {stable_id}"

    # No canonical record was written for a paused or an uncommitted sibling: the
    # cache is disposable custody, not authority.
    assert not scan_jsonl(paths.ledgers.models)

    # Resuming re-uses the preserved sessions instead of re-authoring them.
    resumed_author = ObservedAuthor()
    resumed = _driver(
        tmp_path,
        snapshot,
        author=resumed_author,
        author_concurrency=4,
    ).run()
    assert resumed.status == "complete"
    for stable_id in dispatched_siblings:
        assert stable_id not in resumed_author.calls


# ---------------------------------------------------------------------------
# 3. determinism: identical to serial regardless of completion order
# ---------------------------------------------------------------------------


class _ReverseCompletionChain:
    """Force a wave's sessions to complete in exactly reverse work order.

    Ordering threads by sleeping is not synchronization: it only biases a race, and
    under load the bias loses. An earlier revision of this fixture used a sleep
    ladder and was flaky at ~1-in-2 under a full-tier run -- the *precondition*
    failed, so the equality assertion it exists to set up never even ran.

    This chains the sessions instead. Session ``i`` blocks until session ``i + 1``
    has recorded its completion, so the completion order is reverse work order by
    construction, with no timing assumption at all. The chain is also a concurrency
    assertion in its own right: it can only resolve if every session is in flight at
    once, so a serial lane fails here loudly (timeout) rather than passing quietly.
    """

    def __init__(self, order: dict[str, int], *, timeout: float = 60.0) -> None:
        """Bind the scheduled work order and the per-link deadline."""

        self._order = order
        self._timeout = timeout
        self._done = [threading.Event() for _ in order]

    def hold(self, stable_id: str) -> None:
        """Block until every later-scheduled session has already completed."""

        successor = self._order[stable_id] + 1
        if successor >= len(self._done):
            return
        if not self._done[successor].wait(timeout=self._timeout):
            raise AssertionError(
                f"session {self._order[stable_id]} waited {self._timeout:g}s for its "
                f"successor: the wave is not running its sessions concurrently"
            )

    def after(self, stable_id: str) -> None:
        """Release the predecessor once this session's completion is recorded."""

        self._done[self._order[stable_id]].set()


_DIGEST_PATTERN = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{24,}")


def _canonical_projection(paths: Any, root: Path) -> dict[str, Any]:
    """Return every canonical ledger, normalized for its campaign root only.

    Two campaigns rooted at different absolute paths cannot produce equal content
    digests, so a literal byte comparison would be vacuous. Instead the campaign's
    root is erased and every remaining digest is replaced by the index of its FIRST
    appearance in the campaign's own canonical text. That keeps the comparison
    strict where it matters: two campaigns match only when their ledgers agree
    record for record, in append order, AND the digest *aliasing* agrees -- so a
    variant bound to the wrong representative revision, a record derived from a
    different parent, or a reordered append would all still diverge.

    Parameters
    ----------
    paths:
        Finished campaign's driver paths.
    root:
        Campaign root whose absolute spelling must be erased.

    Returns
    -------
    dict[str, Any]
        Normalized text of each canonical ledger, keyed by ledger name.
    """

    aliases: dict[str, str] = {}

    def alias(match: re.Match[str]) -> str:
        """Return one digest's stable first-appearance alias."""

        return aliases.setdefault(match.group(0), f"<digest-{len(aliases)}>")

    def normalized(ledger: Path) -> list[str]:
        """Return one ledger's records with root and digest spellings normalized."""

        return [
            _DIGEST_PATTERN.sub(
                alias, canonical_json_bytes(record).decode("utf-8").replace(str(root), "<root>")
            )
            for record in scan_jsonl(ledger)
        ]

    return {
        "models": normalized(paths.ledgers.models),
        "attempts": normalized(paths.ledgers.attempts),
        "gates": normalized(paths.ledgers.gates),
    }


def test_reverse_completion_order_yields_the_serial_result_exactly(tmp_path: Path) -> None:
    """A wave whose sessions finish backwards must record the serial result.

    The concurrent run's sessions are held so that the last scheduled model finishes
    first. If any canonical effect were driven by completion order rather than work
    order, the record revisions, the ledger append order, or the terminal statuses
    would diverge from the serial baseline.
    """

    serial_root = tmp_path / "serial"
    concurrent_root = tmp_path / "concurrent"
    serial_root.mkdir()
    concurrent_root.mkdir()

    serial_snapshot = _snapshot(serial_root, count=6)
    concurrent_snapshot = _snapshot(concurrent_root, count=6)
    order = {
        item.stable_id: index for index, item in enumerate(concurrent_snapshot.items)
    }

    serial = _driver(
        serial_root,
        serial_snapshot,
        author=ObservedAuthor(),
        author_concurrency=1,
    ).run()
    chain = _ReverseCompletionChain(order)
    concurrent_author = ObservedAuthor(hold=chain.hold, after=chain.after)
    concurrent = _driver(
        concurrent_root,
        concurrent_snapshot,
        author=concurrent_author,
        author_concurrency=len(order),
    ).run()

    assert serial.status == concurrent.status == "complete"
    # Guaranteed by construction rather than by timing: the chain cannot resolve in
    # any other order. Asserted anyway so a broken chain can never silently weaken
    # the equality check below into a same-order comparison.
    scheduled = [item.stable_id for item in concurrent_snapshot.items]
    assert concurrent_author.completed == list(reversed(scheduled))

    assert _canonical_projection(
        _paths(concurrent_root, concurrent_snapshot), concurrent_root
    ) == _canonical_projection(_paths(serial_root, serial_snapshot), serial_root)


def test_family_variants_resolve_from_their_representative_under_concurrency(
    tmp_path: Path,
) -> None:
    """Representatives must still be resolved before the variants they seed.

    Family variants are scheduled as their own later wave, and a variant must consume
    the representative's artifact rather than opening a session of its own. Running
    the representative wave concurrently must not change that: the concurrent
    campaign has to match the serial one exactly, including the derivation binding.
    """

    serial_root = tmp_path / "serial"
    concurrent_root = tmp_path / "concurrent"
    serial_root.mkdir()
    concurrent_root.mkdir()

    serial_snapshot, serial_representative, serial_variants = _family_snapshot(serial_root)
    snapshot, representative_id, variant_ids = _family_snapshot(concurrent_root)
    counts = {representative_id: 10, variant_ids[0]: 20, variant_ids[1]: 30}
    serial_counts = {
        serial_representative: 10,
        serial_variants[0]: 20,
        serial_variants[1]: 30,
    }

    serial_author = ObservedAuthor()
    assert (
        _driver(
            serial_root,
            serial_snapshot,
            author=serial_author,
            forward=ScriptedForward(ForwardScript(parameter_counts=serial_counts)),
            author_concurrency=1,
        )
        .run()
        .status
        == "complete"
    )
    concurrent_author = ObservedAuthor(hold=lambda _stable_id: time.sleep(0.02))
    assert (
        _driver(
            concurrent_root,
            snapshot,
            author=concurrent_author,
            forward=ScriptedForward(ForwardScript(parameter_counts=counts)),
            author_concurrency=8,
        )
        .run()
        .status
        == "complete"
    )

    # Exactly one session for the representative; the variants are templated.
    assert serial_author.calls == {serial_representative: 1}
    assert concurrent_author.calls == {representative_id: 1}

    assert _canonical_projection(
        _paths(concurrent_root, snapshot), concurrent_root
    ) == _canonical_projection(_paths(serial_root, serial_snapshot), serial_root)

    current = {
        str(model["stable_id"]): model
        for model in scan_jsonl(_paths(concurrent_root, snapshot).ledgers.models)
    }
    for variant_id in variant_ids:
        variant = current[variant_id]
        assert variant["status"]["kind"] == "runs"
        assert variant["budget"]["author_sessions_used"] == 0
        derivation = variant["family_variant_derivation"]
        # The variant is bound to the representative revision this campaign wrote,
        # so a concurrently authored representative still seeds it exactly.
        assert (
            derivation["template_source_revision"] == current[representative_id]["record_revision"]
        )
