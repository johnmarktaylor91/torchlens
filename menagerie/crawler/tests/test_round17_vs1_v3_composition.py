"""Round-17 real worker-result.v3 composition regression tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import menagerie.crawler.driver as driver_module
from menagerie.crawler.driver import SupervisedForwardLane
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.policy import compile_execution_read_manifest
from menagerie.crawler.proposal import model_code_manifest
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    _driver,
    _paths,
    _rebind_fake_author_result,
    _refresh_proposal_identities,
    _snapshot,
    _test_authority_context,
    _test_environment,
)
from menagerie.crawler.tests.conftest import make_worker_result_v3_mapping


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

VS1_LANDING_MANIFEST = {
    "findings": ("SOL-R16-01",),
    "production_symbols": {
        "driver": (
            "_verified_worker_result",
            "_receipt_envelope_error",
            "_attempts_from_supervised",
            "SupervisedForwardLane",
        ),
        "worker_supervisor": (
            "VerifiedWorkerResult",
            "verify_supervised_worker_result",
        ),
    },
    "real_composition_nodes": ("test_real_v3_worker_result_awards_through_driver_and_reducer",),
    "structural_nodes": (
        "test_driver_has_no_direct_supervised_worker_receipt_reads",
        "test_live_protocol_comparisons_stay_in_worker_supervisor",
        "test_vs1_landing_manifest_is_complete",
    ),
}


def test_real_v3_worker_result_awards_through_driver_and_reducer(tmp_path: Path) -> None:
    """A real supervised v3 worker success must earn one canonical ``runs`` revision."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    context = _test_authority_context(snapshot, driver.config)
    artifact = FakeAuthor().author(item, paths.work_root, driver.config, context)

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
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    artifact = _rebind_fake_author_result(artifact)

    environment_root = tmp_path / "current-interpreter-env"
    environment_root.mkdir()
    environment = _test_environment(environment_root)
    execution_identity = driver_module._execution_identity(artifact.proposal, environment)
    code_identity = stable_hash(code_manifest)
    manifest = compile_execution_read_manifest(
        stable_id=item.stable_id,
        work_id=str(artifact.proposal["work_id"]),
        execution_identity=execution_identity,
        code_manifest_identity=code_identity,
        code_members=((adapter_path, adapter_digest, "python-source"),),
        runtime_support=((Path.cwd() / "menagerie", "runtime-root"),),
    )
    with CanonicalReducer(paths.ledgers, context) as reducer:
        artifact = driver._stage_author_result(item, artifact, reducer)
        gate = FakeChecker().check_metadata([artifact], paths.work_root, driver.config).gate
        assert gate is not None
        reducer.append_gate(gate)
        # Keep this wrapper-consumption slice independent of exact-runtime-closure enforcement
        # while exercising the production lane, lease, supervisor, worker, and attempt sink.
        with patch.object(driver_module, "_compile_worker_read_manifest", return_value=manifest):
            attempts = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
                artifact,
                environment,
                1,
                paths.work_root,
                worker_lock_path=paths.worker_lock,
                worker_lease_path=paths.worker_lease,
                run_id=driver.config.run_id,
                attempt_sink=reducer.append_attempt,
            )

        result_path = (
            paths.work_root
            / item.stable_id
            / "forward"
            / "cold-1"
            / "eval"
            / "result"
            / "receipt.json"
        )
        wrapper = json.loads(result_path.read_text(encoding="utf-8"))
        assert wrapper["result_version"] == "menagerie.crawler.worker-result.v3"
        assert wrapper["raw_award_receipt"] is not None
        synthetic = make_worker_result_v3_mapping(
            wrapper["diagnostic"],
            raw_award_receipt=wrapper["raw_award_receipt"],
        )
        assert set(synthetic) == set(wrapper)
        assert set(synthetic["diagnostic"]) == set(wrapper["diagnostic"])
        assert synthetic["result_version"] == wrapper["result_version"]
        assert all(attempt["result"] == "succeeded" for attempt in attempts), [
            (attempt["result"], attempt["error"]) for attempt in attempts
        ]
        assert driver_module._attempt_policy_satisfied(attempts, artifact.proposal, 1)

        persisted = scan_jsonl(paths.ledgers.attempts)
        model = driver_module._assemble_run_model(
            item,
            artifact,
            persisted,
            [gate],
            driver.config,
        )
        appended = reducer.append_model(reducer.prepare_model(model))

    assert appended.record["status"]["code"] == "runs"
    assert len(scan_jsonl(paths.ledgers.models)) == 1


def test_driver_has_no_direct_supervised_worker_receipt_reads() -> None:
    """Every driver semantic consumer must use the central typed projection."""

    tree = ast.parse(Path(driver_module.__file__).read_text(encoding="utf-8"))
    direct_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "worker_receipt"
    ]
    assert direct_reads == []


def test_live_protocol_comparisons_stay_in_worker_supervisor() -> None:
    """Driver code must not branch on nested or outer live protocol literals."""

    versions = {
        "menagerie.crawler.worker-result.v3",
        "menagerie.crawler.worker-receipt.v1",
        "menagerie.crawler.raw-award-receipt.v3",
    }
    tree = ast.parse(Path(driver_module.__file__).read_text(encoding="utf-8"))
    compared_literals = {
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        for value in (node.left, *node.comparators)
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    assert compared_literals.isdisjoint(versions)


def test_vs1_landing_manifest_is_complete() -> None:
    """The VS1 landing unit names every production seam and collected regression."""

    import menagerie.crawler.worker_supervisor as supervisor_module

    modules = {
        "driver": driver_module,
        "worker_supervisor": supervisor_module,
    }
    for module_name, symbols in VS1_LANDING_MANIFEST["production_symbols"].items():
        module = modules[module_name]
        assert all(hasattr(module, symbol) for symbol in symbols)
    expected_nodes = {
        *VS1_LANDING_MANIFEST["real_composition_nodes"],
        *VS1_LANDING_MANIFEST["structural_nodes"],
    }
    assert expected_nodes <= globals().keys()
