"""Round-17 graceful-shutdown award-boundary composition regression tests."""

from __future__ import annotations

import ast
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import signal
import textwrap
from typing import Any

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.driver import (
    AuthorArtifact,
    DriverConfig,
    SupervisedForwardLane,
    WorkItem,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    real_environment_registry,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
)


_TINY_ADAPTER = """from __future__ import annotations

import torch
import menagerie_round19_sentinel as round19_sentinel


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def build_model() -> object:
    assert round19_sentinel.INTERPRETER_SENTINEL == 'round19-selected-prefix'
    return Tiny()


def make_dummy_call(seed: int, device: str) -> tuple[tuple[object, ...], dict[str, object]]:
    del seed
    return ((torch.zeros(1, 2, device=device),), {})
"""

VS2_LANDING_MANIFEST: dict[str, Any] = {
    "findings": ("SOL-R16-03",),
    "production_symbols": {
        "driver": (
            "_SHUTDOWN_ADMISSION_REGISTRY",
            "CrawlerDriver._forward_and_reduce",
            "CrawlerDriver._terminalize",
        ),
    },
    "real_composition_nodes": (
        "test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes",
        "test_signal_at_admission_boundary_publishes_and_awards_nothing_then_resumes",
    ),
    "structural_nodes": (
        "test_shutdown_admission_registry_and_atomic_award_sections_are_complete",
        "test_vs2_landing_manifest_is_complete",
    ),
}


class TinyAdapterAuthor(FakeAuthor):
    """Author one real eval-only typed adapter for the shutdown composition."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Return a proposal whose accepted code can run in the real worker."""

        artifact = super().author(item, work_root, config, context)
        adapter_path = artifact.model_dir / "adapter.py"
        adapter_path.write_text(_TINY_ADAPTER, encoding="utf-8")
        adapter_digest = hash_bytes(adapter_path.read_bytes())
        code_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
        proposal = artifact.proposal
        facts = proposal["proposed_facts"]
        facts["implementation"].update(
            {
                "recipe_type": "typed-adapter",
                "code_path": "adapter.py",
                "code_sha256": adapter_digest,
                "builder_symbol": "build_model",
                "dummy_call_symbol": "make_dummy_call",
                "library_recipe": None,
                "code_manifest": code_manifest,
            }
        )
        facts["input_contract"]["args"][0]["shape"] = [1, 2]
        facts["modes"]["meaningful_modes"] = ["eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
        facts["evidence"]["excerpts"][0]["supports"] = sorted(
            set(facts["evidence"]["excerpts"][0]["supports"])
            | {
                "implementation.code_manifest[].path",
                "implementation.code_manifest[].sha256",
            }
        )
        proposal["verified_hashes"]["code"] = adapter_digest
        proposal["verified_hashes"]["code_manifest"] = stable_hash(code_manifest)
        _refresh_proposal_identities(
            proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        return _rebind_fake_author_result(artifact)


def _files_below(path: Path) -> list[str]:
    """Return stable relative names for every materialized file below ``path``."""

    if not path.exists():
        return []
    return sorted(
        str(candidate.relative_to(path)) for candidate in path.rglob("*") if candidate.is_file()
    )


def _called_symbols(function: Any) -> set[str]:
    """Return every direct name or attribute call in one production function."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            symbols.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            symbols.add(node.func.attr)
    return symbols


def _run_after_forward_shutdown(root: str, fixture: RealEnvironmentFixture) -> None:
    """Run the real driver under its production signal handler in a child process."""

    root_path = Path(root)
    snapshot = _snapshot(root_path, count=1)
    paths = _paths(root_path, snapshot)

    def signal_after_forward(boundary: str, stable_id: str) -> None:
        """Request graceful shutdown exactly after the real worker forward."""

        del stable_id
        if boundary == "after-forward":
            os.kill(os.getpid(), signal.SIGTERM)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    driver = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=signal_after_forward,
        registry=real_environment_registry(fixture),
    )
    interrupted = driver.run()

    artifact_events = scan_jsonl(paths.ledgers.artifacts)
    operational = scan_jsonl(paths.operational_ledger)
    attempts = scan_jsonl(paths.ledgers.attempts)
    observation: dict[str, Any] = {
        "status": interrupted.status,
        "admission_boundary": (
            interrupted.shutdown_interruption.admission_boundary
            if interrupted.shutdown_interruption is not None
            else None
        ),
        "models": len(scan_jsonl(paths.ledgers.models)),
        "attempt_results": [attempt["result"] for attempt in attempts],
        "artifact_event_kinds": [event["event_kind"] for event in artifact_events],
        "operational_event_kinds": [event["event_kind"] for event in operational],
        "public_mirror_files": _files_below(paths.runtime_root / "mirrors" / "public"),
        "repository_files": _files_below(root_path / "menagerie"),
        "worker_lease_exists": paths.worker_lease.exists(),
        "handlers_restored": all(
            signal.getsignal(signum) == previous_handlers[signum]
            for signum in (signal.SIGTERM, signal.SIGINT)
        ),
        "driver_state": json.loads(paths.driver_state.read_text(encoding="utf-8")),
    }

    def reject_second_forward(boundary: str, stable_id: str) -> None:
        """Prove resume consumes the durable real attempt without another worker."""

        del stable_id
        if boundary == "after-forward":
            raise AssertionError("resume unexpectedly admitted a second worker forward")

    resumed_driver = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=reject_second_forward,
        registry=real_environment_registry(fixture),
    )
    resumed = resumed_driver.run()
    observation.update(
        {
            "resume_status": resumed.status,
            "resume_models": len(scan_jsonl(paths.ledgers.models)),
            "resume_attempts": len(scan_jsonl(paths.ledgers.attempts)),
        }
    )
    (root_path / "shutdown-observation.json").write_text(
        json.dumps(observation, sort_keys=True), encoding="utf-8"
    )


def _run_admission_boundary_shutdown(
    root: str,
    fixture: RealEnvironmentFixture,
    target_boundary: str,
) -> None:
    """Signal at one admission event and record interruption plus real-prefix resume."""

    root_path = Path(root)
    snapshot = _snapshot(root_path, count=1)
    paths = _paths(root_path, snapshot)
    author = TinyAdapterAuthor()
    checker = FakeChecker()
    observed_boundaries: list[str] = []
    signal_sent = False

    def signal_at_boundary(boundary: str, stable_id: str) -> None:
        """Send one real SIGTERM at the selected supported lifecycle event."""

        nonlocal signal_sent
        del stable_id
        observed_boundaries.append(boundary)
        if boundary == target_boundary and not signal_sent:
            signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    driver = _driver(
        root_path,
        snapshot,
        author=author,
        checker=checker,
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=signal_at_boundary,
        registry=real_environment_registry(fixture),
    )
    interrupted = driver.run()

    attempts_before_resume = scan_jsonl(paths.ledgers.attempts)
    observation: dict[str, Any] = {
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
        "models": len(scan_jsonl(paths.ledgers.models)),
        "attempt_results": [attempt["result"] for attempt in attempts_before_resume],
        "artifact_event_kinds": [
            event["event_kind"] for event in scan_jsonl(paths.ledgers.artifacts)
        ],
        "gate_count": len(scan_jsonl(paths.ledgers.gates)),
        "operational_event_kinds": [
            event["event_kind"] for event in scan_jsonl(paths.operational_ledger)
        ],
        "public_mirror_files": _files_below(paths.runtime_root / "mirrors" / "public"),
        "repository_files": _files_below(root_path / "menagerie"),
        "worker_lease_exists": paths.worker_lease.exists(),
        "handlers_restored": all(
            signal.getsignal(signum) == previous_handlers[signum]
            for signum in (signal.SIGTERM, signal.SIGINT)
        ),
        "driver_state": json.loads(paths.driver_state.read_text(encoding="utf-8")),
    }

    def reject_repeated_forward(boundary: str, stable_id: str) -> None:
        """Reject a second worker only when the pre-publication attempt is durable."""

        del stable_id
        if target_boundary == "pre-publication" and boundary == "after-forward":
            raise AssertionError("resume unexpectedly admitted a second worker forward")

    resumed_driver = _driver(
        root_path,
        snapshot,
        author=TinyAdapterAuthor(),
        checker=FakeChecker(),
        forward=SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()),
        environments=RealEnvironmentLane(fixture),
        boundary=reject_repeated_forward,
        registry=real_environment_registry(fixture),
    )
    resumed = resumed_driver.run()
    resumed_attempts = scan_jsonl(paths.ledgers.attempts)
    observation.update(
        {
            "resume_status": resumed.status,
            "resume_models": len(scan_jsonl(paths.ledgers.models)),
            "resume_attempts": len(resumed_attempts),
            "resume_manifest_identities": sorted(
                {str(attempt["execution_read_manifest_identity"]) for attempt in resumed_attempts}
            ),
            "resume_environment_authority_ids": sorted(
                {
                    str(attempt["environment"]["environment_authority_id"])
                    for attempt in resumed_attempts
                }
            ),
            "resume_environment_ids": sorted(
                {str(attempt["environment"]["env_id"]) for attempt in resumed_attempts}
            ),
            "resume_environment_authority_epochs": sorted(
                {str(attempt["environment"]["authority_epoch"]) for attempt in resumed_attempts}
            ),
            "resume_selected_interpreters": sorted(
                {
                    str(attempt["environment"]["selected_interpreter_relative_path"])
                    for attempt in resumed_attempts
                }
            ),
        }
    )
    (root_path / f"shutdown-{target_boundary}-observation.json").write_text(
        json.dumps(observation, sort_keys=True), encoding="utf-8"
    )


def test_signal_after_real_forward_publishes_and_awards_nothing_then_resumes(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """SIGTERM after a real v3 forward must leave the slot unawarded and resumable."""

    process = multiprocessing.get_context("fork").Process(
        target=_run_after_forward_shutdown,
        args=(str(tmp_path), real_environment_fixture),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("shutdown composition child did not exit")
    assert process.exitcode == 0

    observation = json.loads((tmp_path / "shutdown-observation.json").read_text(encoding="utf-8"))
    assert observation["status"] == "interrupted:shutdown"
    assert observation["models"] == 0
    assert observation["admission_boundary"] == "post-attempt-pre-award"
    assert observation["attempt_results"] == ["succeeded"]
    assert observation["artifact_event_kinds"] == ["staged-private"]
    assert observation["operational_event_kinds"].count("worker-shutdown-interrupted") == 1
    assert observation["public_mirror_files"] == []
    assert observation["repository_files"] == []
    assert observation["worker_lease_exists"] is False
    assert observation["handlers_restored"] is True
    assert observation["driver_state"] == {"status": "interrupted:shutdown"}
    assert observation["resume_status"] == "complete"
    assert observation["resume_models"] == 1
    assert observation["resume_attempts"] == 1


@pytest.mark.parametrize(
    (
        "target_boundary",
        "expected_admission_boundary",
        "expected_author_calls",
        "expected_checker_calls",
        "expected_attempt_results",
        "expected_artifact_events",
        "expected_gate_count",
    ),
    (
        ("pre-author", "author-admission", 0, 0, [], [], 0),
        ("pre-checker", "checker-admission", 1, 0, [], ["staged-private"], 0),
        (
            "pre-publication",
            "pre-publication-admission",
            1,
            1,
            ["succeeded"],
            ["staged-private"],
            1,
        ),
    ),
)
def test_signal_at_admission_boundary_publishes_and_awards_nothing_then_resumes(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
    target_boundary: str,
    expected_admission_boundary: str,
    expected_author_calls: int,
    expected_checker_calls: int,
    expected_attempt_results: list[str],
    expected_artifact_events: list[str],
    expected_gate_count: int,
) -> None:
    """Each pre-admission SIGTERM must append no model or public artifact."""

    process = multiprocessing.get_context("fork").Process(
        target=_run_admission_boundary_shutdown,
        args=(str(tmp_path), real_environment_fixture, target_boundary),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail(f"{target_boundary} shutdown composition child did not exit")
    assert process.exitcode == 0

    observation = json.loads(
        (tmp_path / f"shutdown-{target_boundary}-observation.json").read_text(encoding="utf-8")
    )
    assert observation["status"] == "interrupted:shutdown"
    assert observation["admission_boundary"] == expected_admission_boundary
    assert observation["signal_sent"] is True
    assert observation["observed_boundaries"][-1] == target_boundary
    assert observation["author_calls"] == expected_author_calls
    assert observation["checker_calls"] == expected_checker_calls
    assert observation["models"] == 0
    assert observation["attempt_results"] == expected_attempt_results
    assert "failed" not in observation["attempt_results"]
    assert observation["artifact_event_kinds"] == expected_artifact_events
    assert observation["gate_count"] == expected_gate_count
    assert observation["operational_event_kinds"].count("worker-shutdown-interrupted") == 1
    assert observation["public_mirror_files"] == []
    assert observation["repository_files"] == []
    assert observation["worker_lease_exists"] is False
    assert observation["handlers_restored"] is True
    assert observation["driver_state"] == {"status": "interrupted:shutdown"}
    assert observation["resume_status"] == "complete"
    assert observation["resume_models"] == 1
    assert observation["resume_attempts"] == 1
    assert len(observation["resume_manifest_identities"]) == 1
    assert len(observation["resume_environment_authority_ids"]) == 1
    assert observation["resume_environment_ids"] == [str(real_environment_fixture.prefix)]
    assert observation["resume_environment_authority_epochs"] == [
        "menagerie.crawler.environment-authority.v1"
    ]
    assert observation["resume_selected_interpreters"] == ["bin/python"]


def test_shutdown_admission_registry_and_atomic_award_sections_are_complete() -> None:
    """All award/publication edges remain guarded or inside the atomic section."""

    registry = driver_module._SHUTDOWN_ADMISSION_REGISTRY
    assert registry == {
        "author": "guard:author-admission",
        "checker": "guard:checker-admission",
        "environment-create": "guard:environment-create-admission",
        "environment-use": "guard:environment-use-admission",
        "model": "guard:model-admission",
        "lease": "guard:forward-admission|pre-slot-resolution",
        "spawn": "guard:forward-admission|pre-slot-resolution",
        "run-model-assembly": "guard:post-attempt-pre-award",
        "publication-admission": "guard:pre-publication-admission",
        "publication": "atomic:award-commit",
        "terminal-publication": "atomic:award-commit",
        "model-append": "atomic:award-commit",
        "post-award-observation": "guard:post-award-commit",
    }
    run_calls = _called_symbols(driver_module.CrawlerDriver._forward_and_reduce)
    terminal_calls = _called_symbols(driver_module.CrawlerDriver._terminalize)
    assert {
        "_check_shutdown",
        "_assemble_run_model",
        "_authorize_and_publish_artifact",
        "append_model",
    } <= run_calls
    assert {"_check_shutdown", "_authorize_terminal_artifact", "append_model"} <= terminal_calls

    run_source = inspect.getsource(driver_module.CrawlerDriver._forward_and_reduce)
    post_attempt = run_source.index('"post-attempt-pre-award"')
    assembly = run_source.index("_assemble_run_model(")
    pre_publication = run_source.index('"pre-publication"')
    publication_admission = run_source.index('"pre-publication-admission"')
    pre_commit = run_source.index('"pre-award-commit"')
    publication = run_source.index("_authorize_and_publish_artifact(")
    append = run_source.index("reducer.append_model(")
    post_commit = run_source.index('"post-award-commit"')
    assert (
        post_attempt
        < assembly
        < pre_publication
        < publication_admission
        < pre_commit
        < publication
        < append
        < post_commit
    )
    assert "_check_shutdown" not in run_source[publication:append]

    terminal_source = inspect.getsource(driver_module.CrawlerDriver._terminalize)
    terminal_pre_publication = terminal_source.index('"pre-publication"')
    terminal_publication_admission = terminal_source.index('"pre-publication-admission"')
    terminal_pre_commit = terminal_source.index('"pre-award-commit"')
    terminal_publication = terminal_source.index("_authorize_terminal_artifact(")
    terminal_append = terminal_source.index("reducer.append_model(")
    terminal_post_commit = terminal_source.index('"post-award-commit"')
    assert (
        terminal_pre_publication
        < terminal_publication_admission
        < terminal_pre_commit
        < terminal_publication
        < terminal_append
        < terminal_post_commit
    )
    assert "_check_shutdown" not in terminal_source[terminal_publication:terminal_append]


def test_vs2_landing_manifest_is_complete() -> None:
    """The VS2 landing unit names its driver seams and collected regressions."""

    for symbol in VS2_LANDING_MANIFEST["production_symbols"]["driver"]:
        current: Any = driver_module
        for part in symbol.split("."):
            current = getattr(current, part)
    expected_nodes = {
        *VS2_LANDING_MANIFEST["real_composition_nodes"],
        *VS2_LANDING_MANIFEST["structural_nodes"],
    }
    assert expected_nodes <= globals().keys()
