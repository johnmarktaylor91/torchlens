"""Round-17 real worker-result.v3 composition regression tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.authority import derive_mode_summary
from menagerie.crawler.driver import SupervisedForwardLane
from menagerie.crawler.identity import hash_bytes, stable_hash
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
)
from menagerie.crawler.tests.conftest import (
    RealEnvironmentFixture,
    make_worker_result_v3_mapping,
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

_MODE_ADAPTER = _TINY_ADAPTER.replace(
    "return value + 1",
    "return value + (1 if self.training else 2)",
)

VS1_LANDING_MANIFEST: dict[str, Any] = {
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


@pytest.mark.parametrize(
    ("adapter_source", "expected_divergence"),
    ((_TINY_ADAPTER, "none"), (_MODE_ADAPTER, "statistical")),
)
def test_real_v3_worker_result_awards_through_driver_and_reducer(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
    adapter_source: str,
    expected_divergence: str,
) -> None:
    """Real persisted value digests must reach none/statistical mode authority.

    Parameters
    ----------
    tmp_path:
        Isolated campaign root.
    adapter_source, expected_divergence:
        Real adapter bytes and their authenticated mode classification.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    context = _test_authority_context(snapshot, driver.config)
    artifact = FakeAuthor().author(item, paths.work_root, driver.config, context)

    adapter_path = artifact.model_dir / "adapter.py"
    adapter_path.write_text(adapter_source, encoding="utf-8")
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
    facts["modes"]["meaningful_modes"] = ["train", "eval"]
    facts["modes"]["train_eval_divergence"] = expected_divergence
    facts["external_metadata"]["modes"]["meaningful_modes"] = ["train", "eval"]
    facts["external_metadata"]["modes"]["train_eval_divergence"] = expected_divergence
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

    environment = real_environment_fixture.binding
    context = replace(
        context,
        environment_generations={
            **context.environment_generations,
            environment.family: environment.env_generation,
        },
    )

    def append_attempt(attempt: Mapping[str, Any]) -> None:
        """Append one attempt while satisfying the forward-lane sink contract."""

        reducer.append_attempt(attempt)

    with CanonicalReducer(paths.ledgers, context) as reducer:
        artifact = driver._stage_author_result(item, artifact, reducer)
        gate = FakeChecker().check_metadata([artifact], paths.work_root, driver.config).gate
        assert gate is not None
        reducer.append_gate(gate)
        attempts = SupervisedForwardLane(timeout_seconds=20, cwd=Path.cwd()).forward(
            artifact,
            environment,
            1,
            paths.work_root,
            worker_lock_path=paths.worker_lock,
            worker_lease_path=paths.worker_lease,
            run_id=driver.config.run_id,
            attempt_sink=append_attempt,
        )

        for mode in ("train", "eval"):
            result_path = (
                paths.work_root
                / item.stable_id
                / "forward"
                / "cold-1"
                / mode
                / "result"
                / "receipt.json"
            )
            wrapper = json.loads(result_path.read_text(encoding="utf-8"))
            assert wrapper["result_version"] == "menagerie.crawler.worker-result.v3"
            assert wrapper["raw_award_receipt"] is not None, wrapper["diagnostic"]
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
        closure = driver_module._collect_worker_executable_closure(artifact, environment)
        execution_identity = driver_module._execution_identity(
            artifact,
            environment,
            closure_identity=closure.identity,
        )
        recomputed_manifest = driver_module._compile_worker_read_manifest(
            artifact,
            environment,
            execution_identity,
            closure=closure,
        )
        assert {str(attempt["execution_read_manifest_identity"]) for attempt in attempts} == {
            recomputed_manifest.manifest_id
        }
        for attempt in attempts:
            argv = list(attempt["invocation"]["argv"])
            interpreter_index = argv.index(str(environment.python_executable))
            assert argv[interpreter_index : interpreter_index + 4] == [
                str(environment.python_executable),
                "-B",
                "-m",
                "menagerie.crawler.worker",
            ]
        assert all(
            attempt["environment"]["environment_authority_id"]
            == environment.environment_authority_id
            for attempt in attempts
        )

        persisted = scan_jsonl(paths.ledgers.attempts)
        by_mode = {str(attempt["mode"]): attempt for attempt in persisted}
        summary = derive_mode_summary(by_mode["train"], by_mode["eval"])
        assert summary.comparison_state == "verified"
        assert summary.classification == expected_divergence
        assert summary.compared_fields == ("output_signature", "output_value_sha256")
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
    assert driver_module._current_run_is_fresh(
        appended.record,
        artifact,
        environment,
        [gate],
    )
    prior_execution_identity = str(appended.record["execution"]["execution_identity"])
    adapter_path.write_text(f"{adapter_source}\nMUTATED_MEMBER = True\n", encoding="utf-8")
    mutated_digest = hash_bytes(adapter_path.read_bytes())
    mutated_manifest = [dict(row) for row in model_code_manifest(adapter_path, artifact.model_dir)]
    artifact.proposal["proposed_facts"]["implementation"]["code_sha256"] = mutated_digest
    artifact.proposal["proposed_facts"]["implementation"]["code_manifest"] = mutated_manifest
    artifact.proposal["verified_hashes"]["code"] = mutated_digest
    artifact.proposal["verified_hashes"]["code_manifest"] = stable_hash(mutated_manifest)
    _refresh_proposal_identities(
        artifact.proposal,
        checker_model=driver.config.checker_model,
        checker_version=driver.config.checker_version,
    )
    mutated_artifact = _rebind_fake_author_result(artifact)
    assert driver_module._execution_identity(mutated_artifact, environment) != (
        prior_execution_identity
    )
    assert not driver_module._current_run_is_fresh(
        appended.record,
        mutated_artifact,
        environment,
        [gate],
    )


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
