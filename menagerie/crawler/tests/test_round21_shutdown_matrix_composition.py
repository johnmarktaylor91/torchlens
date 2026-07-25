"""Round-21 exhaustive real-prefix shutdown/admission/atomicity matrix."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import signal
from typing import Any, Mapping

import pytest

import menagerie.crawler.driver as driver_module
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.driver import SupervisedForwardLane
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    real_environment_registry,
)
from menagerie.crawler.tests.test_boundary_shutdown_composition import TinyAdapterAuthor
from menagerie.crawler.tests.test_slice_f_driver import FakeChecker, _driver, _paths, _snapshot


@dataclass(frozen=True)
class _ShutdownScenario:
    """One exact shutdown-matrix signal point and its expected durable partition."""

    cell_id: str
    signal_boundary: str
    admission_boundary: str
    model_count: int
    models_before_resume: int
    attempts_before_resume: int
    spawns_before_resume: int
    spawns_after_resume: int


_SCENARIOS = (
    _ShutdownScenario("S01", "pre-author", "author-admission", 1, 0, 0, 0, 1),
    _ShutdownScenario("S02", "pre-checker", "checker-admission", 1, 0, 0, 0, 1),
    _ShutdownScenario(
        "S03", "pre-environment-create", "environment-create-admission", 1, 0, 0, 0, 1
    ),
    _ShutdownScenario("S04", "pre-environment-use", "environment-use-admission", 1, 0, 0, 0, 1),
    _ShutdownScenario("S05", "pre-model", "model-admission", 1, 0, 0, 0, 1),
    _ShutdownScenario("S06", "pre-forward", "forward-admission", 1, 0, 0, 0, 1),
    _ShutdownScenario("S07", "worker-lease-started", "worker-supervision", 1, 0, 0, 1, 2),
    _ShutdownScenario("S08", "after-forward", "post-attempt-pre-award", 1, 0, 1, 1, 1),
    _ShutdownScenario("S09", "pre-publication", "pre-publication-admission", 1, 0, 1, 1, 1),
    _ShutdownScenario("S10", "pre-award-commit", "pre-award-commit", 1, 0, 1, 1, 1),
    _ShutdownScenario("S11", "award-commit-entered", "post-award-commit", 1, 1, 1, 1, 1),
    _ShutdownScenario("S12", "post-award-commit", "post-award-commit", 2, 1, 1, 1, 2),
    _ShutdownScenario("S13", "after-reduce", "model-admission", 3, 1, 1, 1, 3),
)

_ADMISSION_CELL_BY_EDGE = {
    "author": "S01",
    "checker": "S02",
    "environment-create": "S03",
    "environment-use": "S04",
    "model": "S05",
    "lease": "S06",
    "spawn": "S06",
    "run-model-assembly": "S08",
    "publication-admission": "S09",
    "publication": "S11",
    "terminal-publication": "S11",
    "model-append": "S11",
    "post-award-observation": "S12",
}
_AWARD_CELL_BY_EVENT = {
    "after-forward": "S08",
    "post-attempt-pre-award": "S08",
    "pre-publication": "S09",
    "pre-award-commit": "S10",
    "award-commit-entered": "S11",
    "post-award-commit": "S12",
}
_CHILD_CELL_BY_EVENT = {"worker-lease-started": "S07"}
_WAVE_CELL_BY_EVENT = {"after-reduce": "S13"}
_ROUND21_SHUTDOWN_MATRIX = tuple((scenario.cell_id, scenario) for scenario in _SCENARIOS)


def _files_below(path: Path) -> list[str]:
    """Return stable relative names for every materialized file below a root."""

    if not path.exists():
        return []
    return sorted(
        str(candidate.relative_to(path)) for candidate in path.rglob("*") if candidate.is_file()
    )


def _capture_state(root: Path, paths: Any) -> dict[str, Any]:
    """Capture the complete durable state needed to prove one matrix partition.

    Parameters
    ----------
    root:
        Isolated campaign root.
    paths:
        Production driver paths for the campaign.

    Returns
    -------
    dict[str, Any]
        Canonical ledgers, materialization inventories, and lease state.
    """

    return {
        "models": scan_jsonl(paths.ledgers.models),
        "attempts": scan_jsonl(paths.ledgers.attempts),
        "gates": scan_jsonl(paths.ledgers.gates),
        "artifact_events": scan_jsonl(paths.ledgers.artifacts),
        "operational": scan_jsonl(paths.operational_ledger),
        "repository_files": _files_below(root / "menagerie"),
        "public_mirror_files": _files_below(paths.runtime_root / "mirrors" / "public"),
        "worker_lease_exists": paths.worker_lease.exists(),
        "driver_state": json.loads(paths.driver_state.read_text(encoding="utf-8")),
    }


def _run_shutdown_scenario(
    root: str,
    fixture: RealEnvironmentFixture,
    scenario: _ShutdownScenario,
) -> None:
    """Run one real SIGTERM interruption and one exact production resume.

    Parameters
    ----------
    root:
        Isolated campaign root serialized for the forked process.
    fixture:
        Shared strict real-prefix fixture.
    scenario:
        Exact matrix signal and expected partition.
    """

    root_path = Path(root)
    snapshot = _snapshot(root_path, count=scenario.model_count)
    paths = _paths(root_path, snapshot)
    author = TinyAdapterAuthor()
    checker = FakeChecker()
    environments = RealEnvironmentLane(fixture)
    observed_boundaries: list[tuple[str, str]] = []
    signal_sent = False

    def signal_once(boundary: str, stable_id: str) -> None:
        """Send one real SIGTERM at the selected shipped boundary hook."""

        nonlocal signal_sent
        observed_boundaries.append((boundary, stable_id))
        if boundary == scenario.signal_boundary and not signal_sent:
            signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    interrupted = _driver(
        root_path,
        snapshot,
        author=author,
        checker=checker,
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=environments,
        boundary=signal_once,
        registry=real_environment_registry(fixture),
    ).run()
    before = _capture_state(root_path, paths)
    before.update(
        {
            "status": interrupted.status,
            "admission_boundary": (
                interrupted.shutdown_interruption.admission_boundary
                if interrupted.shutdown_interruption is not None
                else None
            ),
            "signal_sent": signal_sent,
            "observed_boundaries": observed_boundaries,
            "author_calls": sum(author.calls.values()),
            "checker_calls": checker.metadata_calls + checker.fidelity_calls,
            "environment_events": environments.events,
            "handlers_restored": all(
                signal.getsignal(signum) == previous_handlers[signum]
                for signum in (signal.SIGTERM, signal.SIGINT)
            ),
        }
    )

    resume_boundaries: list[tuple[str, str]] = []

    def observe_resume(boundary: str, stable_id: str) -> None:
        """Record every shipped boundary reached during the sole resume."""

        resume_boundaries.append((boundary, stable_id))

    resumed = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=observe_resume,
        registry=real_environment_registry(fixture),
    ).run()
    after = _capture_state(root_path, paths)
    after.update({"status": resumed.status, "observed_boundaries": resume_boundaries})
    (root_path / f"round21-{scenario.cell_id}-observation.json").write_text(
        json.dumps({"before": before, "after": after}, sort_keys=True),
        encoding="utf-8",
    )


def _worker_start_count(state: Mapping[str, Any]) -> int:
    """Return the number of durable real child-start lifecycle facts."""

    return sum(event.get("event_kind") == "worker-lease-started" for event in state["operational"])


def _shutdown_events(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return typed shutdown-interruption operational facts from a captured state."""

    return [
        event
        for event in state["operational"]
        if event.get("event_kind") == "worker-shutdown-interrupted"
    ]


def _assert_artifact_model_bijection(root: Path, state: Mapping[str, Any]) -> None:
    """Prove every final artifact and model refer to the same bytes and claims.

    Parameters
    ----------
    root:
        Campaign root containing canonical repository materializations.
    state:
        Captured canonical model and artifact ledgers.
    """

    models = state["models"]
    final_events = [
        event for event in state["artifact_events"] if event.get("event_kind") == "published"
    ]
    assert {model["stable_id"] for model in models} == {
        event["stable_id"] for event in final_events
    }
    events_by_id = {event["artifact_event_id"]: event for event in final_events}
    for model in models:
        authority = model["artifact_authority"]
        event = events_by_id[authority["committed_event_id"]]
        assert authority["state"] == "published"
        assert authority["transaction_id"] == event["transaction_id"]
        assert authority["authorization_id"] == event["authorization_id"]
        assert authority["reconstruction_sha256"] == event["reconstruction"]["sha256"]
        assert authority["claim_ids"] == sorted(claim["claim_id"] for claim in event["claims"])
        assert event["publication_inventory"]["lane"] == "public"
        assert all(claim["license_disposition"] == "public-compatible" for claim in event["claims"])
        objects = {obj["object_id"]: obj for obj in event["objects"]}
        for claim in event["claims"]:
            obj = objects[claim["object_id"]]
            materialized = root / claim["logical_path"]
            assert obj["mirror_class"] == "public"
            assert materialized.is_file()
            assert hash_bytes(materialized.read_bytes()) == obj["content_sha256"]
            assert materialized.stat().st_size == obj["byte_count"]


def _assert_registry_totality_and_order() -> None:
    """Prove every registered admission/event source derives exactly S01--S13."""

    assert set(_ADMISSION_CELL_BY_EDGE) == set(driver_module._SHUTDOWN_ADMISSION_REGISTRY)
    hook_registry = driver_module._SHUTDOWN_COMPOSITION_HOOK_REGISTRY
    expected_hooks = {
        "author": "pre-author",
        "checker": "pre-checker",
        "environment-create": "pre-environment-create",
        "environment-use": "pre-environment-use",
        "model": "pre-model",
        "lease|spawn": "pre-forward",
        "child-durable": "worker-lease-started",
        "attempt-durable": "after-forward",
        "run-model-assembly": "post-attempt-pre-award",
        "publication-admission": "pre-publication",
        "award-entry": "pre-award-commit",
        "award-atomic": "award-commit-entered",
        "post-award-observation": "post-award-commit",
        "wave-transition": "after-reduce",
    }
    assert hook_registry == expected_hooks
    assert supervisor_module._SHUTDOWN_CHILD_DURABILITY_EVENT_REGISTRY == {
        "worker-lease-started": ("child_pid", "child_start_token", "child_pgid")
    }
    derived_ids = {
        *_ADMISSION_CELL_BY_EDGE.values(),
        *_AWARD_CELL_BY_EVENT.values(),
        *_CHILD_CELL_BY_EVENT.values(),
        *_WAVE_CELL_BY_EVENT.values(),
    }
    assert derived_ids == {f"S{index:02d}" for index in range(1, 14)}
    assert tuple(cell_id for cell_id, _scenario in _ROUND21_SHUTDOWN_MATRIX) == tuple(
        f"S{index:02d}" for index in range(1, 14)
    )

    environment_source = inspect.getsource(driver_module.CrawlerDriver._run_environment_work)
    assert environment_source.index('boundary_hook("pre-environment-create"') < (
        environment_source.index('_check_shutdown("environment-create-admission"')
    )
    assert environment_source.index('boundary_hook("pre-environment-use"') < (
        environment_source.index('_check_shutdown("environment-use-admission"')
    )
    assert environment_source.index('boundary_hook("pre-model"') < environment_source.index(
        '_check_shutdown("model-admission"'
    )
    forward_source = inspect.getsource(driver_module.CrawlerDriver._forward_and_reduce)
    assert forward_source.index('boundary_hook("pre-forward"') < forward_source.index(
        '_check_shutdown(\n            "forward-admission"'
    )
    publication = forward_source.index("self._authorize_and_publish_artifact(")
    append = forward_source.index("reducer.append_model(")
    post_award_hook = forward_source.index('boundary_hook("post-award-commit"')
    post_award_guard = forward_source.index('_check_shutdown("post-award-commit"')
    assert publication < append < post_award_hook < post_award_guard
    assert "_check_shutdown" not in forward_source[publication:append]


def _assert_shutdown_partition(
    root: Path,
    scenario: _ShutdownScenario,
    observation: Mapping[str, Any],
) -> None:
    """Assert the exact interruption/resume partition for one real matrix cell.

    Parameters
    ----------
    root:
        Isolated campaign root.
    scenario:
        Expected cell partition.
    observation:
        Forked-process durable state before and after resume.
    """

    before = observation["before"]
    after = observation["after"]
    assert before["status"] == "interrupted:shutdown"
    assert before["admission_boundary"] == scenario.admission_boundary
    assert before["signal_sent"] is True
    assert any(row[0] == scenario.signal_boundary for row in before["observed_boundaries"])
    assert before["handlers_restored"] is True
    assert before["worker_lease_exists"] is False
    assert before["driver_state"] == {"status": "interrupted:shutdown"}
    assert len(before["models"]) == scenario.models_before_resume
    assert len(before["attempts"]) == scenario.attempts_before_resume
    assert all(attempt["result"] == "succeeded" for attempt in before["attempts"])
    assert _worker_start_count(before) == scenario.spawns_before_resume
    shutdown_events = _shutdown_events(before)
    assert len(shutdown_events) == 1
    assert shutdown_events[0]["details"]["admission_boundary"] == scenario.admission_boundary
    if scenario.cell_id == "S07":
        details = shutdown_events[0]["details"]
        assert details["child_pid"] is not None
        assert details["child_start_token"] is not None
        assert details["child_pgid"] == details["child_pid"]
        assert details["parent_observation"]["shutdown_requested"] is True

    if scenario.cell_id <= "S10":
        assert before["repository_files"] == []
        assert before["public_mirror_files"] == []
    _assert_artifact_model_bijection(root, before)

    assert after["status"] == "complete"
    assert after["worker_lease_exists"] is False
    assert len(after["models"]) == scenario.model_count
    assert len(after["attempts"]) == scenario.model_count
    assert len({model["stable_id"] for model in after["models"]}) == scenario.model_count
    assert _worker_start_count(after) == scenario.spawns_after_resume
    assert len(_shutdown_events(after)) == 1
    before_attempts = {
        attempt["attempt_id"]: attempt["payload_sha256"] for attempt in before["attempts"]
    }
    after_attempts = {
        attempt["attempt_id"]: attempt["payload_sha256"] for attempt in after["attempts"]
    }
    assert before_attempts.items() <= after_attempts.items()
    _assert_artifact_model_bijection(root, after)


@pytest.mark.parametrize(
    ("cell_id", "scenario"),
    _ROUND21_SHUTDOWN_MATRIX,
    ids=[cell_id for cell_id, _scenario in _ROUND21_SHUTDOWN_MATRIX],
)
def test_round21_shutdown_matrix(
    cell_id: str,
    scenario: _ShutdownScenario,
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """Execute every registry-derived S01--S13 cell as a real composition.

    Parameters
    ----------
    cell_id, scenario:
        Exact closed matrix identifier and signal/resume contract.
    tmp_path:
        Per-cell scratch root deleted immediately after the test.
    real_environment_fixture:
        Sole shared read-only strict real-prefix fixture.
    """

    _assert_registry_totality_and_order()
    assert cell_id == scenario.cell_id
    assert set(structural.ROUND21_VS5_PROOF_REGISTRY) == {
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P12",
        "P13",
        "P14",
        "P17",
        "P19",
        "T01",
        "T01-CI",
        "T02",
        "T03",
    }
    process = multiprocessing.get_context("fork").Process(
        target=_run_shutdown_scenario,
        args=(str(tmp_path), real_environment_fixture, scenario),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail(f"{cell_id} shutdown composition child did not exit")
    assert process.exitcode == 0
    observation = json.loads(
        (tmp_path / f"round21-{cell_id}-observation.json").read_text(encoding="utf-8")
    )
    _assert_shutdown_partition(tmp_path, scenario, observation)
