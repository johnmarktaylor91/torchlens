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
from unittest.mock import patch

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
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    RealEnvironmentLane,
    real_environment_registry,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
)


_TINY_ADAPTER = """from __future__ import annotations

import torch


class Tiny(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def build_model() -> object:
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


def _claim_specific_license_decisions(
    driver: driver_module.CrawlerDriver,
    artifact: AuthorArtifact,
) -> dict[Any, LicenseDecision]:
    """Provide distinct legitimate dispositions for source and adapter claims."""

    del driver
    assert artifact.staged is not None
    objects = {obj.object_id: obj for obj in artifact.staged.objects}
    decisions: dict[Any, LicenseDecision] = {}
    for claim in artifact.staged.custody_claims:
        public = claim.logical_role == "code"
        decisions[claim.claim_id] = LicenseDecision(
            content_sha256=objects[claim.object_id].content_sha256,
            redistribution_class=(
                RedistributionClass.PUBLIC_OK if public else RedistributionClass.RESTRICTED_PRIVATE
            ),
            evidence_ids=("evidence-1",),
            rationale="test fixture binds claim-specific accepted license evidence",
        )
    return decisions


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
    with (
        patch.object(
            driver_module.CrawlerDriver,
            "_license_decisions",
            _claim_specific_license_decisions,
        ),
    ):
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
    with (
        patch.object(
            driver_module.CrawlerDriver,
            "_license_decisions",
            _claim_specific_license_decisions,
        ),
    ):
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


def test_shutdown_admission_registry_and_atomic_award_sections_are_complete() -> None:
    """All award/publication edges remain guarded or inside the atomic section."""

    registry = driver_module._SHUTDOWN_ADMISSION_REGISTRY
    assert registry == {
        "author": "guard:author-admission",
        "checker": "guard:external-call-admission",
        "environment-create": "guard:environment-create-admission",
        "environment-use": "guard:environment-use-admission",
        "model": "guard:model-admission",
        "lease": "guard:forward-admission|pre-slot-resolution",
        "spawn": "guard:forward-admission|pre-slot-resolution",
        "run-model-assembly": "guard:post-attempt-pre-award",
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
    pre_commit = run_source.index('"pre-award-commit"')
    publication = run_source.index("_authorize_and_publish_artifact(")
    append = run_source.index("reducer.append_model(")
    post_commit = run_source.index('"post-award-commit"')
    assert post_attempt < assembly < pre_commit < publication < append < post_commit
    assert "_check_shutdown" not in run_source[publication:append]

    terminal_source = inspect.getsource(driver_module.CrawlerDriver._terminalize)
    terminal_pre_commit = terminal_source.index('"pre-award-commit"')
    terminal_publication = terminal_source.index("_authorize_terminal_artifact(")
    terminal_append = terminal_source.index("reducer.append_model(")
    terminal_post_commit = terminal_source.index('"post-award-commit"')
    assert terminal_pre_commit < terminal_publication < terminal_append < terminal_post_commit
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
