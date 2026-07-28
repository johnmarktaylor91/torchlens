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

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    DEFAULT_AUTHOR_WAVE_CONCURRENCY,
    MAX_AUTHOR_WAVE_CONCURRENCY,
)
from menagerie.crawler.driver import AuthorArtifact, DriverConfig
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
        **kwargs: Any,
    ) -> None:
        """Bind the in-session hold and the shared observation state."""

        super().__init__(**kwargs)
        self._hold = hold or (lambda _stable_id: None)
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


def _reverse_completion_hold(order: dict[str, int], unit: float = 0.03) -> Callable[[str], None]:
    """Return a hold that makes sessions finish in exactly reverse work order."""

    def hold(stable_id: str) -> None:
        """Sleep longer the earlier the model is scheduled."""

        time.sleep(unit * (len(order) - order[stable_id]))

    return hold


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
    concurrent_author = ObservedAuthor(hold=_reverse_completion_hold(order))
    concurrent = _driver(
        concurrent_root,
        concurrent_snapshot,
        author=concurrent_author,
        author_concurrency=6,
    ).run()

    assert serial.status == concurrent.status == "complete"
    # The hold really did invert completion order relative to scheduling order.
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
