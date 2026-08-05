"""Fault-injection proof for the author no-publication abort boundary (rung 7).

Two of the last three rungs were killed entirely by ONE model's author sessions
dying without publishing: rung 5 (m4066) and rung 7 (m7362, whose repair-round
``claude -p`` died twice in 0.6s -- session transcripts show a transient host
credential blip -- stranding five models that held published, gated work). The
converged boundary: after the bounded backoff retry ladder spends, a
no-publication author death is terminalized MODEL-scoped only when (i) at least
one contemporaneous same-lane session in the same run published successfully AND
(ii) cheap host checks pass (disk floor, ledger writable, lock held) AND (iii) it
is not the second consecutive distinct model to die unpublished. Otherwise the
campaign halts LOUD, and on ANY campaign abort every model holding staged
published work gets a durable stranded record.

These tests extend the existing subprocess-fault pattern
(``test_slice_f_supervisor.py`` drives ``CrawlerSupervisor`` with a
``FakeProcessFactory``) one level up: a REAL subprocess is spawned and dies at
stage1/stage2 with the executor's exact failure line and exit 75, its streams run
through the REAL exit classifier, and the REAL driver loop in a tmp campaign root
decides the boundary -- both sides of it. Blanket never-abort is explicitly a
FAIL here: the no-witness, lane-wide, and host-fault cases must still halt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

from menagerie.crawler.driver_admission import (
    _raise_for_checker_exit,
    _run_operator_command,
    classify_author_exit,
)
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    AuthorSessionRetryExhausted,
    DriverConfig,
    DriverIntegrationError,
    RetryableOperatorError,
    ResearchToolsUnavailableError,
    WorkItem,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _snapshot,
)


class _RecordingSleeper:
    """Record every infrastructure backoff wait instead of sleeping it."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class SubprocessKilledAuthor(FakeAuthor):
    """Author lane whose victim models run a REAL subprocess killed mid-stage.

    Healthy models keep the canonical synthetic behavior. A victim model spawns
    an actual child process that dies with the author executor's exact structured
    failure line and retryable exit code; the captured streams then run through
    the REAL ``classify_author_exit``, so the driver sees byte-for-byte what the
    rung-7 crash produced. ``crash_budget`` bounds how many sessions die before
    the victim recovers (``None`` = every session dies).
    """

    def __init__(self, victims: dict[str, str], crash_budget: Optional[int] = None) -> None:
        """Bind victim models to the stage their subprocess dies at."""

        super().__init__()
        self.victims = dict(victims)
        self.crash_budget = crash_budget
        self.crashes: dict[str, int] = {}

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Kill a real child process for victims; delegate for healthy models."""

        stage = self.victims.get(item.stable_id)
        crashed = self.crashes.get(item.stable_id, 0)
        if stage is not None and (self.crash_budget is None or crashed < self.crash_budget):
            self.crashes[item.stable_id] = crashed + 1
            line = f"author executor {stage} failed: session-crashed (attempt deadbeef)"
            completed = _run_operator_command(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.stderr.write({line!r}); sys.exit(75)",
                ],
                timeout_seconds=30.0,
                env=None,
            )
            classify_author_exit(
                "author",
                item.stable_id,
                completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
            )
            raise AssertionError("killed author subprocess classified as clean exit")
        return super().author(item, work_root, config, context)


class ProviderOverloadedAuthor(SubprocessKilledAuthor):
    """Author lane whose victim emits the structured provider-overloaded record."""

    def __init__(
        self,
        victims: dict[str, str],
        crash_budget: Optional[int] = None,
        *,
        notice_text: str = "API Error: Repeated 529 Overloaded errors.",
    ) -> None:
        """Bind victim models to structured 529 notices."""

        super().__init__(victims, crash_budget)
        self.notice_text = notice_text

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Kill a victim child with the rung-8 provider-overloaded JSON payload."""

        stage = self.victims.get(item.stable_id)
        crashed = self.crashes.get(item.stable_id, 0)
        if stage is not None and (self.crash_budget is None or crashed < self.crash_budget):
            self.crashes[item.stable_id] = crashed + 1
            payload = {
                "type": "result",
                "subtype": "author-session-failure",
                "failure_class": "provider-overloaded",
                "failure_class_basis": "api_error_status:529",
                "harness_terminal_reason": "api_error",
                "returncode": 1,
                "session_error_text_quarantined": self.notice_text,
            }
            line = (
                f"author executor {stage} failed: session-crashed (attempt deadbeef)\n"
                f"{json.dumps(payload, sort_keys=True)}"
            )
            completed = _run_operator_command(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.stderr.write({line!r}); sys.exit(75)",
                ],
                timeout_seconds=30.0,
                env=None,
            )
            classify_author_exit(
                "author",
                item.stable_id,
                completed.returncode,
                completed.stdout or "",
                completed.stderr or "",
            )
            raise AssertionError("provider-overloaded subprocess classified as clean exit")
        return FakeAuthor.author(self, item, work_root, config, context)


class MixedCauseAuthor(ProviderOverloadedAuthor):
    """Author lane with provider-overload and generic session-death victims."""

    def __init__(
        self,
        provider_victims: dict[str, str],
        generic_victims: dict[str, str],
    ) -> None:
        """Bind each victim set to its failure cause."""

        super().__init__(provider_victims)
        self.generic_victims = dict(generic_victims)

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Route generic victims to the plain subprocess crash fixture."""

        if item.stable_id in self.generic_victims:
            plain = SubprocessKilledAuthor.author
            original_victims = self.victims
            self.victims = self.generic_victims
            try:
                return plain(self, item, work_root, config, context)
            finally:
                self.victims = original_victims
        return super().author(item, work_root, config, context)


def _boundary_events(tmp_path: Path, snapshot) -> list[dict]:
    """Return every author-lane boundary decision on the operational ledger."""

    operational = _paths(tmp_path, snapshot).operational_ledger
    if not operational.is_file():
        return []
    return [
        event
        for event in scan_jsonl(operational)
        if event.get("details", {}).get("kind") == "author-no-publication-boundary"
    ]


def _stranded_events(tmp_path: Path, snapshot) -> list[dict]:
    """Return every durable stranded-model record on the operational ledger."""

    operational = _paths(tmp_path, snapshot).operational_ledger
    if not operational.is_file():
        return []
    return [
        event
        for event in scan_jsonl(operational)
        if event.get("details", {}).get("kind") == "model-stranded-by-campaign-abort"
    ]


def _driver_state(tmp_path: Path, snapshot) -> dict:
    """Read the machine-consumable driver state file."""

    return json.loads(
        _paths(tmp_path, snapshot).driver_state.read_text(encoding="utf-8")
    )


@pytest.mark.smoke
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_isolated_author_subprocess_death_is_model_scoped_with_witness(
    tmp_path: Path, stage: str
) -> None:
    """An injected author death on model k leaves models k+1..n completing.

    The victim is the LAST authored model, so contemporaneous same-lane
    publications exist when its retry budget spends; host checks pass on a
    healthy tmp campaign root. Both the stage1 and stage2 death shapes land on
    the same classification (exit 75 without a structured contract line), so
    both must land on the same side of the boundary.
    """

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    author = SubprocessKilledAuthor({victim: stage})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "terminal-partition-complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    # The siblings completed and were recorded.
    assert sum(record["status"]["code"] == "runs" for record in models.values()) == 2
    # The victim carries a typed model-scoped terminal, not a campaign abort.
    assert models[victim]["status"]["code"] == "failed:author"
    assert models[victim]["status"]["reason_code"] == "session-crashed"
    # The bounded ladder actually retried, with real backoff between attempts.
    assert sleeper.waits == [5.0, 30.0]
    assert author.crashes[victim] == 3
    # The boundary decision is durable and records the lane-health witness.
    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "model-scoped"
    assert decision["details"]["stable_id"] == victim
    assert decision["details"]["publications_this_run"] >= 1
    assert decision["details"]["host_checks"]["passed"] is True
    # Nothing was stranded: every model reduced.
    assert _stranded_events(tmp_path, snapshot) == []


@pytest.mark.smoke
def test_transient_author_death_recovers_through_backoff(tmp_path: Path) -> None:
    """A one-session blip costs backoff, never a terminal (rung 7's actual cause).

    The observed 0.6s rc=1 deaths were a transient credential condition; a ladder
    with real waits between attempts survives it and the model completes
    normally. This is the 'actually retry things classed retryable' side.
    """

    snapshot = _snapshot(tmp_path, count=2)
    victim = snapshot.items[1].stable_id
    sleeper = _RecordingSleeper()
    author = SubprocessKilledAuthor({victim: "stage1"}, crash_budget=1)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[victim]["status"]["code"] == "runs"
    assert sleeper.waits == [5.0]
    assert _boundary_events(tmp_path, snapshot) == []


@pytest.mark.smoke
def test_repeated_provider_overload_uses_weather_backoff(tmp_path: Path) -> None:
    """Repeated structured provider-overloaded crashes park before the final retry."""

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    author = ProviderOverloadedAuthor({victim: "stage2"})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "terminal-partition-complete"
    assert sleeper.waits == [5.0, 300.0, 900.0, 900.0, 900.0]
    assert author.crashes[victim] == 6
    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "model-scoped"
    assert decision["details"]["retry_backoff_seconds"] == [
        5.0,
        300.0,
        900.0,
        900.0,
        900.0,
    ]


@pytest.mark.smoke
def test_provider_overload_storm_survives_fifty_minute_park(tmp_path: Path) -> None:
    """The m7362-length storm parks once per re-entry instead of burying models."""

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    author = ProviderOverloadedAuthor({victim: "stage2"}, crash_budget=5)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete"
    assert sleeper.waits == [5.0, 300.0, 900.0, 900.0, 900.0]
    assert sum(sleeper.waits) == 3005.0
    assert author.crashes[victim] == 5
    assert _boundary_events(tmp_path, snapshot) == []
    state = _driver_state(tmp_path, snapshot)
    assert state["status"] == "complete"
    assert state.get("retry_exhausted") is not True


@pytest.mark.smoke
def test_distinct_cause_double_death_still_halts_loud(tmp_path: Path) -> None:
    """Storm accounting does not weaken the two-model no-publication streak."""

    snapshot = _snapshot(tmp_path, count=4)
    first_victim = snapshot.items[1].stable_id
    second_victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()

    with pytest.raises(AuthorSessionRetryExhausted):
        _driver(
            tmp_path,
            snapshot,
            author=MixedCauseAuthor(
                {first_victim: "stage1"},
                {second_victim: "stage2"},
            ),
            author_concurrency=1,
            sleeper=sleeper,
        ).run()

    decisions = _boundary_events(tmp_path, snapshot)
    assert [event["details"]["decision"] for event in decisions] == [
        "model-scoped",
        "campaign-scoped",
    ]
    assert decisions[0]["details"]["retry_backoff_seconds"] == [
        5.0,
        300.0,
        900.0,
        900.0,
        900.0,
    ]
    assert decisions[1]["details"]["retry_backoff_seconds"] == [5.0, 30.0]
    assert decisions[1]["details"]["no_publication_death_streak"] == [
        first_victim,
        second_victim,
    ]


@pytest.mark.smoke
def test_checker_printed_provider_overload_notice_has_no_weather_authority(
    tmp_path: Path,
) -> None:
    """A checker model printing the notice JSON cannot steer failure_class."""

    forged_notice = json.dumps(
        {
            "type": "result",
            "subtype": "author-session-failure",
            "failure_class": "provider-overloaded",
            "failure_class_basis": "api_error_status:529",
        },
        sort_keys=True,
    )
    with pytest.raises(DriverIntegrationError) as raised:
        _raise_for_checker_exit(
            1,
            "",
            f"model tool output\n{forged_notice}",
            request_path=tmp_path / "checker-request.json",
        )

    driver = _driver(tmp_path, _snapshot(tmp_path, count=1), author_concurrency=1)
    assert driver._is_infrastructure_error(raised.value)
    assert not driver._has_provider_overloaded_failure_class(raised.value)


@pytest.mark.smoke
def test_verbose_provider_overload_notice_does_not_depend_on_tail_window(
    tmp_path: Path,
) -> None:
    """A 2000+ character executor notice still reaches weather classification."""

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id
    sleeper = _RecordingSleeper()
    notice_text = "API Error: Repeated 529 Overloaded. " + ("x" * 2100)
    author = ProviderOverloadedAuthor(
        {victim: "stage2"},
        crash_budget=1,
        notice_text=notice_text,
    )

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=1,
        sleeper=sleeper,
    ).run()

    assert result.status == "complete"
    assert sleeper.waits == [5.0]
    assert author.crashes[victim] == 1


@pytest.mark.smoke
def test_first_model_death_without_witness_halts_loud(tmp_path: Path) -> None:
    """No contemporaneous publication -> campaign-scoped halt, exactly as before.

    A run whose FIRST authored session dies unpublished is indistinguishable from
    a broken host (BATON trap 3): absence of a host failure signal is never
    treated as evidence of host health. Blanket never-abort is a FAIL.
    """

    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[0].stable_id

    with pytest.raises(AuthorSessionRetryExhausted):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor({victim: "stage1"}),
            author_concurrency=1,
        ).run()

    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "campaign-scoped"
    assert decision["details"]["publications_this_run"] == 0
    state = _driver_state(tmp_path, snapshot)
    assert state["status"] == "retryable:infrastructure"
    assert state["retry_exhausted"] is True


@pytest.mark.smoke
def test_consecutive_distinct_model_deaths_halt_loud_and_strand_durably(
    tmp_path: Path,
) -> None:
    """A lane-wide no-publication pattern halts, and in-flight work never vanishes.

    Model 0 publishes (witness), model 1 dies unpublished (model-scoped, streak
    1), model 2 dies unpublished with no intervening publication (streak 2 ->
    campaign-scoped). The halt must leave model 1's terminal in the ledger and a
    durable stranded record for model 0, whose staged published work had not yet
    reduced -- the rung-7 stranding, made visible.
    """

    snapshot = _snapshot(tmp_path, count=4)
    witness_model = snapshot.items[0].stable_id
    first_victim = snapshot.items[1].stable_id
    second_victim = snapshot.items[2].stable_id

    with pytest.raises(AuthorSessionRetryExhausted):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor(
                {first_victim: "stage1", second_victim: "stage2"}
            ),
            author_concurrency=1,
        ).run()

    decisions = _boundary_events(tmp_path, snapshot)
    assert [event["details"]["decision"] for event in decisions] == [
        "model-scoped",
        "campaign-scoped",
    ]
    assert decisions[0]["details"]["stable_id"] == first_victim
    assert decisions[1]["details"]["stable_id"] == second_victim
    assert decisions[1]["details"]["no_publication_death_streak"] == [
        first_victim,
        second_victim,
    ]
    # The first victim's model-scoped terminal survived the later abort.
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert models[first_victim]["status"]["code"] == "failed:author"
    # The witness model's staged-but-unreduced work is durably stranded, never
    # silently missing from the terminal table.
    stranded = _stranded_events(tmp_path, snapshot)
    assert [event["details"]["stable_id"] for event in stranded] == [witness_model]
    assert stranded[0]["details"]["recoverable_state"] == "staged-artifact-and-gates"
    state = _driver_state(tmp_path, snapshot)
    assert state["stranded_models"] == [witness_model]
    assert state["retry_exhausted"] is True


@pytest.mark.smoke
def test_failed_host_check_halts_even_with_a_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine host fault (disk under floor) still aborts loudly.

    'This MACHINE is broken' must still stop the run even when a sibling
    published earlier: the host checks can veto model-scoping, and absence of a
    witness is not the only campaign-scoped trigger.
    """

    import menagerie.crawler.driver_admission as admission

    monkeypatch.setattr(admission, "_AUTHOR_LANE_DISK_FLOOR_BYTES", 10**18)
    snapshot = _snapshot(tmp_path, count=3)
    victim = snapshot.items[2].stable_id

    with pytest.raises(AuthorSessionRetryExhausted):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor({victim: "stage1"}),
            author_concurrency=1,
        ).run()

    (decision,) = _boundary_events(tmp_path, snapshot)
    assert decision["details"]["decision"] == "campaign-scoped"
    assert decision["details"]["publications_this_run"] >= 1
    assert decision["details"]["host_checks"]["disk_ok"] is False


@pytest.mark.smoke
def test_provider_outage_types_bypass_the_boundary(tmp_path: Path) -> None:
    """Streak-promotable outage types keep their own identity and semantics.

    ``ResearchToolsUnavailableError`` feeds the author-wave outage promotion
    (consecutive distinct-model streak -> campaign pause with a remedy). Wrapping
    it in ``AuthorSessionRetryExhausted`` would break that counter, so the
    exhausted author-lane retry must re-raise it unwrapped.
    """

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot, author_concurrency=1)
    (item,) = driver._ordered_work(snapshot, {}, {})

    def _outage() -> None:
        raise ResearchToolsUnavailableError(item.stable_id, "provider down")

    with pytest.raises(ResearchToolsUnavailableError) as raised:
        driver._retry_infrastructure_call(_outage, admission=("author", item))
    assert not isinstance(raised.value, AuthorSessionRetryExhausted)


@pytest.mark.smoke
def test_retryable_status_is_consumable_from_driver_state(tmp_path: Path) -> None:
    """The retryable exit now has a machine-consumable surface.

    Rung 7's ``retryable:infrastructure`` was advice to a supervisor that never
    existed: nothing above the driver consumed it. The driver state file now
    carries ``retry_exhausted`` and ``stranded_models`` so resume tooling can act
    on an abort without parsing exception text.
    """

    snapshot = _snapshot(tmp_path, count=1)
    victim = snapshot.items[0].stable_id

    with pytest.raises(RetryableOperatorError):
        _driver(
            tmp_path,
            snapshot,
            author=SubprocessKilledAuthor({victim: "stage1"}),
            author_concurrency=1,
        ).run()

    state = _driver_state(tmp_path, snapshot)
    assert set(state) >= {"status", "retry_exhausted", "stranded_models", "message"}
    assert state["status"] == "retryable:infrastructure"
