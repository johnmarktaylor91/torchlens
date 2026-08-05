"""The author wall budget is single-sourced and coherent end to end.

Two budget systems used to describe the same thing and the stricter one won
silently: the driver's lane killed every author wrapper at the DEFAULT 30-minute
grant, while the executor believed c3-classics had 60 minutes. c3 is exactly the
campaign that needs the long budget -- the no-prior-code classics -- so the lane
truncated the tail the grant existed to cover, recorded the truncation as a
timeout, retried blind into the same wall, and corrupted the C3 p95 the month's
go/no-go depends on.

These tests pin the replacement: ONE resolution point, an explicitly derived and
strictly ordered relationship between the grant and the lane's stall bound, and
a startup refusal when a stale environment value disagrees.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

from menagerie.crawler.author_dispatch import AuthorEffortGrant
from menagerie.crawler.author_executor import ExecutorConfig
from menagerie.crawler.campaign_config import (
    CampaignConfig,
    load_campaign_config,
    write_campaign_config,
)
from menagerie.crawler.constants import (
    AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET,
    AUTHOR_SESSION_WALL_SECONDS,
    AUTHOR_WALL_EXTERNAL_KILL_FACTOR,
    AUTHOR_WALL_SECONDS_ENV,
    TIER_CAMPAIGN_IDS,
    author_lane_wall_bound,
    resolve_author_wall_seconds,
)

pytestmark = pytest.mark.smoke


def test_c3_classics_carries_the_long_grant_and_the_others_the_default() -> None:
    """The per-campaign table is the single source of the grant."""

    assert resolve_author_wall_seconds("c3-classics") == 60.0 * 60.0
    for campaign in ("c1-mech", "c2-disco", "c4-native", None):
        assert resolve_author_wall_seconds(campaign) == float(AUTHOR_SESSION_WALL_SECONDS)


def test_an_explicit_override_wins_and_a_malformed_one_is_refused() -> None:
    """An operator grant is honoured; a nonsense one never becomes a budget."""

    assert resolve_author_wall_seconds("c3-classics", 900.0) == 900.0
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            resolve_author_wall_seconds("c3-classics", bad)


@pytest.mark.parametrize("campaign", sorted(TIER_CAMPAIGN_IDS) + [None])
def test_the_limit_ladder_is_strictly_ordered_for_every_campaign(campaign: object) -> None:
    """grant < executor kill < lane stall bound, with no ties.

    This is the invariant whose violation was the defect. The lane bound is the
    OUTERMOST limit; if it ever slides inside the executor's own kill, the
    executor loses the chance to record a typed, recoverable outcome and every
    overrun degrades into an opaque SIGKILL again.
    """

    grant = resolve_author_wall_seconds(campaign if isinstance(campaign, str) else None)
    executor_kill = grant * AUTHOR_WALL_EXTERNAL_KILL_FACTOR
    lane_bound = author_lane_wall_bound(grant)
    assert grant < executor_kill < lane_bound


def test_the_lane_bound_covers_the_executors_worst_case_invocation() -> None:
    """One stage-2 round trip may legitimately run 2.5 grant-sized sessions."""

    grant = resolve_author_wall_seconds("c3-classics")
    worst_case = grant * AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET
    assert author_lane_wall_bound(grant) > worst_case * AUTHOR_WALL_EXTERNAL_KILL_FACTOR


def test_the_session_budget_matches_the_executors_actual_call_sites() -> None:
    """Structural tripwire on the multiplier the lane bound is derived from.

    ``AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET`` is only honest while it
    describes the sessions the executor really runs. Adding a fourth stage-2
    session, or widening the supplementary round's half grant, silently makes
    the lane bound too tight again -- the exact shape of the original defect.
    Any change to a session's wall argument fails here and forces the budget to
    be re-derived deliberately.
    """

    source = Path(__file__).resolve().parents[1] / "author_executor.py"
    observed: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if getattr(inner.func, "id", "") != "run_claude_session":
                continue
            for keyword in inner.keywords:
                if keyword.arg == "wall_seconds":
                    observed.setdefault(node.name, []).append(ast.unparse(keyword.value))

    # Stage 1 is its own lane round trip, so it is bounded independently. The
    # capability probe carries its own short deadline and is not grant-derived.
    # The pre-publication gate's continuation rounds run through EXACTLY ONE
    # call site whose grant is remaining-budget-bounded (see the derivation
    # test below), so they consume only wall the primary session left unused
    # and the 2.5 multiplier stays valid without a constant change -- the
    # census revoked the 2.5 -> 3.0 bump on evidence.
    assert observed == {
        "serve_source_request": ["config.wall_seconds()"],
        "serve_author": ["config.wall_seconds()", "config.wall_seconds()"],
        "_maybe_supplement_round": ["config.wall_seconds() / 2"],
        "_stage2_continuation": ["wall_seconds"],
        "serve_capability_probe": ["deadline"],
    }
    stage2_multiplier = 1.0 + 1.0 + 0.5
    assert AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET == stage2_multiplier


def test_a_continuation_round_only_spends_the_attempts_unused_wall() -> None:
    """The gate's rounds are bounded by ``grant - spent``, never a new budget.

    This is the derivation the call-site tripwire above points at: the wall
    argument reaching ``_stage2_continuation`` comes from
    ``_continuation_wall_seconds``, whose ceiling is the wall the attempt's own
    grant has left. An attempt's sessions therefore never total more than one
    grant, and the invocation budget the lane bound is derived from is
    unchanged.
    """

    from menagerie.crawler.author_executor import (
        _CONTINUATION_MIN_WALL_SECONDS,
        _continuation_wall_seconds,
    )

    grant = 1800.0
    # The rung-8 shape: a session that bailed at minute ~4 leaves its own wall.
    assert _continuation_wall_seconds(grant, 250.0) == pytest.approx(1550.0)
    # spent + continuation never exceeds the grant.
    for spent in (0.0, 250.0, 900.0, 1650.0, 1799.0, 1800.0, 2500.0):
        wall = _continuation_wall_seconds(grant, spent)
        assert wall >= 0.0
        assert spent + wall <= grant + 1e-9 or wall == 0.0
    # Below the usefulness floor no round is dispatched at all.
    assert _continuation_wall_seconds(grant, grant - _CONTINUATION_MIN_WALL_SECONDS + 1.0) == 0.0
    assert _continuation_wall_seconds(grant, grant) == 0.0
    assert _continuation_wall_seconds(grant, grant + 100.0) == 0.0


def test_the_executor_resolves_the_same_grant_the_lane_sized_its_bound_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subprocess and the driver read one table, not two."""

    monkeypatch.delenv(AUTHOR_WALL_SECONDS_ENV, raising=False)
    config = ExecutorConfig.from_env(campaign_id="c3-classics")
    assert config.wall_seconds() == resolve_author_wall_seconds("c3-classics")
    assert config.wall_seconds() == 3600.0


def test_resolving_the_grant_never_mutates_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution is a pure read (regression).

    Writing the grant into the driver's own ``os.environ`` made resolution
    order-dependent and process-wide: the value outlived its caller, so the next
    driver built in the same process inherited a foreign campaign's grant and
    this module's own startup guard refused a perfectly correct configuration.
    A test that merely ran later in the file was enough to trip it.
    """

    from menagerie.crawler.cli import _resolve_author_wall

    monkeypatch.delenv(AUTHOR_WALL_SECONDS_ENV, raising=False)
    before = dict(os.environ)

    assert _resolve_author_wall(_namespace(author_wall_seconds=None), "c3-classics") == 3600.0

    assert AUTHOR_WALL_SECONDS_ENV not in os.environ
    assert dict(os.environ) == before
    # ... so a later, differently-budgeted campaign still resolves its own grant.
    assert _resolve_author_wall(_namespace(author_wall_seconds=None), "c1-mech") == 1800.0


def test_the_grant_travels_in_the_wrapper_childs_environment() -> None:
    """The executor's grant is subprocess-scoped, explicit, and exact."""

    from menagerie.crawler.driver_admission import CommandAuthorLane

    grant = resolve_author_wall_seconds("c3-classics")
    lane = CommandAuthorLane(
        (sys.executable, "-c", "pass"),
        effort_grant=AuthorEffortGrant(wall_seconds=grant),
    )
    child = lane._child_environment()

    assert child[AUTHOR_WALL_SECONDS_ENV] == repr(3600.0)
    # The child inherits the rest of the environment, and only the child.
    assert child["PATH"] == os.environ["PATH"]
    assert AUTHOR_WALL_SECONDS_ENV not in os.environ


def test_the_child_grant_round_trips_through_the_executors_own_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the lane hands the child is exactly what the executor resolves.

    A lossy hand-off would reintroduce a disagreement between the number the
    lane sized its stall bound from and the number the session was told.
    """

    from menagerie.crawler.driver_admission import CommandAuthorLane

    lane = CommandAuthorLane(
        (sys.executable, "-c", "pass"),
        effort_grant=AuthorEffortGrant(wall_seconds=resolve_author_wall_seconds("c3-classics")),
    )
    child = lane._child_environment()

    # Stand in for the subprocess: the executor reads the env it was handed.
    monkeypatch.setenv(AUTHOR_WALL_SECONDS_ENV, child[AUTHOR_WALL_SECONDS_ENV])
    # Even a campaign whose OWN default is 1800 honours the handed-down grant.
    assert ExecutorConfig.from_env(campaign_id="c1-mech").wall_seconds() == 3600.0


def test_a_conflicting_environment_grant_refuses_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale variable is a loud startup failure, not a silent winner.

    Failing here is the whole point: the alternative is discovering the
    disagreement at minute 30 of a 60-minute session, as a timeout.
    """

    from menagerie.crawler.cli import OperatorOutageError, _resolve_author_wall

    monkeypatch.setenv(AUTHOR_WALL_SECONDS_ENV, "1800")
    with pytest.raises(OperatorOutageError) as raised:
        _resolve_author_wall(_namespace(author_wall_seconds=None), "c3-classics")
    assert "conflicts with the resolved" in str(raised.value)

    monkeypatch.setenv(AUTHOR_WALL_SECONDS_ENV, "not-a-number")
    with pytest.raises(OperatorOutageError):
        _resolve_author_wall(_namespace(author_wall_seconds=None), "c3-classics")


def test_resolving_one_campaign_does_not_poison_the_next_driver_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact cross-test ordering failure, pinned.

    Reproduces what the mid tier caught: resolve a 3600s c3 grant, then build a
    driver for a 1800s campaign in the SAME process, as a supervised driver
    handling successive campaigns would. Under the defect the second resolution
    saw the first's leaked variable and raised
    ``OperatorOutageError: ...=3600 conflicts with the resolved 1800s grant``.
    """

    from menagerie.crawler.cli import _resolve_author_wall

    monkeypatch.delenv(AUTHOR_WALL_SECONDS_ENV, raising=False)
    assert _resolve_author_wall(_namespace(author_wall_seconds=None), "c3-classics") == 3600.0
    # No campaign at all -- the case the wake-callback test exercises.
    assert _resolve_author_wall(_namespace(author_wall_seconds=None), None) == 1800.0
    assert _resolve_author_wall(_namespace(author_wall_seconds=None), "c3-classics") == 3600.0


def test_a_matching_environment_grant_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agreement is not an error -- only disagreement is."""

    from menagerie.crawler.cli import _resolve_author_wall

    monkeypatch.setenv(AUTHOR_WALL_SECONDS_ENV, "3600")
    assert _resolve_author_wall(_namespace(author_wall_seconds=None), "c3-classics") == 3600.0


def test_a_tuned_grant_survives_a_supervised_restart(tmp_path: Path) -> None:
    """The concurrency knob's drop-on-restart hole, closed for the wall.

    ``supervisor.py`` rebuilds the driver's resume argv from the frozen campaign
    config, so anything absent from that file reverts to a default mid-campaign
    without a word. A tuned wall grant must round-trip.
    """

    config = _campaign_config(tmp_path, author_wall_seconds=5400.0)
    path = tmp_path / "campaign.json"
    write_campaign_config(path, config)
    assert load_campaign_config(path).author_wall_seconds == 5400.0

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["author_wall_seconds"] == 5400.0
    assert payload["format"].endswith(".v3")


def test_an_unset_grant_round_trips_as_the_campaign_default(tmp_path: Path) -> None:
    """``None`` means "take the campaign's own grant", and stays ``None``."""

    path = tmp_path / "campaign.json"
    write_campaign_config(path, _campaign_config(tmp_path, author_wall_seconds=None))
    assert load_campaign_config(path).author_wall_seconds is None


def test_a_persisted_nonsense_grant_is_refused_at_load(tmp_path: Path) -> None:
    """A corrupt budget never silently becomes a month-long campaign's ceiling."""

    path = tmp_path / "campaign.json"
    write_campaign_config(path, _campaign_config(tmp_path, author_wall_seconds=5400.0))
    payload = json.loads(path.read_text(encoding="utf-8"))
    for bad in (0, -1, "3600", True):
        payload["author_wall_seconds"] = bad
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)
        with pytest.raises(ValueError):
            load_campaign_config(path)


def test_the_lane_does_not_kill_a_wrapper_inside_its_stall_bound(tmp_path: Path) -> None:
    """Behavioural regression: a session outliving the bare grant survives.

    Under the defect the lane's timeout WAS the grant, so a wrapper that ran one
    second past it was killed and reported as a timeout. Here the wrapper sleeps
    well past its grant but far inside the derived stall bound, and must be
    allowed to finish.
    """

    from menagerie.crawler.driver_admission import CommandAuthorLane

    grant = 1.0
    assert author_lane_wall_bound(grant) > 100.0  # the bound is not the grant

    wrapper = "import time\ntime.sleep(3.0)\n"
    lane = CommandAuthorLane(
        (sys.executable, "-c", wrapper),
        effort_grant=AuthorEffortGrant(wall_seconds=grant),
    )
    root = tmp_path / "slow-author"
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request.json"
    request_path.write_text(json.dumps({"envelope_version": "x"}), encoding="utf-8")

    # No exception: three seconds is 3x the grant and still inside the bound.
    lane._dispatch(
        kind="source-request",
        item=_item(),
        config=None,
        work_id="work-slow",
        request_path=request_path,
        output_path=root / "result.json",
    )


def test_an_injected_stall_bound_cannot_truncate_the_session_budget() -> None:
    """The injectable bound is not an escape hatch back into the defect.

    A bound at or inside the executor's own kill is refused at construction:
    that is precisely the configuration -- an outer limit tighter than the inner
    budget -- that killed c3 sessions at minute 30.
    """

    from menagerie.crawler.driver_admission import CommandAuthorLane

    grant = AuthorEffortGrant(wall_seconds=100.0)
    executor_kill = 100.0 * AUTHOR_WALL_EXTERNAL_KILL_FACTOR
    for bound in (1.0, 100.0, executor_kill):
        with pytest.raises(ValueError, match="truncating the session budget"):
            CommandAuthorLane(
                (sys.executable, "-c", "pass"),
                effort_grant=grant,
                stall_bound_seconds=bound,
            )

    lane = CommandAuthorLane(
        (sys.executable, "-c", "pass"),
        effort_grant=grant,
        stall_bound_seconds=executor_kill + 0.5,
    )
    assert lane.stall_bound_seconds == executor_kill + 0.5


def test_the_production_lane_takes_the_derived_bound() -> None:
    """No injected bound means the derivation governs, with nothing in between."""

    from menagerie.crawler.driver_admission import CommandAuthorLane

    lane = CommandAuthorLane(
        (sys.executable, "-c", "pass"),
        effort_grant=AuthorEffortGrant(wall_seconds=resolve_author_wall_seconds("c3-classics")),
    )
    assert lane.stall_bound_seconds is None
    assert lane.effort_grant.wall_seconds == 3600.0


def _namespace(**values: object) -> object:
    """Return a minimal argparse-like namespace."""

    import argparse

    return argparse.Namespace(**values)


def _item() -> object:
    """Return a minimal work item carrying only the fields the lane reads."""

    import argparse

    return argparse.Namespace(stable_id="model-under-test")


def _campaign_config(tmp_path: Path, *, author_wall_seconds: float | None) -> CampaignConfig:
    """Return a valid campaign config with the given wall grant."""

    executable = Path(os.path.realpath(sys.executable))
    return CampaignConfig(
        repo_root=tmp_path,
        intake_root=tmp_path / "intake",
        target="osx-arm64",
        run_id="crawler-run",
        author_queue_root=None,
        author_command=(str(executable), "-c", "pass"),
        checker_command=(str(executable), "-c", "pass"),
        environment_command=(str(executable), "-c", "pass"),
        notify_command=None,
        public_mirror=tmp_path / "public",
        private_mirror=tmp_path / "private",
        review_checkpoint_at=1000,
        progress_milestones=(900, 950, 1000),
        phase=None,
        only_status=None,
        author_wall_seconds=author_wall_seconds,
    )
