"""Focused tests for the round-14 replayable authority kernel."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.authority import (
    AuthorityContext,
    AuthorityDerivationError,
    DependencyState,
    ExecutionReadManifestV2,
    RuntimeLookupDirectory,
    RuntimeMember,
    ShutdownInterruptionFact,
    authenticate_accepted_attempts,
    compile_execution_read_manifest_v2,
    derive_attempt_projection,
    derive_dependency_vector,
    derive_execution_identity,
    derive_family_authority,
    derive_mode_summary,
    derive_parent_attestation,
    derive_per_mode_run,
    derive_terminal_observation,
    derive_terminal_proof,
    derive_award_closure_identity,
    derive_runner_identity,
    family_authority_projection,
    load_current_attempt_proof,
    load_current_gate_proof,
    raw_award_receipt_sha256,
    resolve_exact_gate_item_membership,
    validate_currency,
    verify_execution_read_manifest_v2,
    completion_line_for_raw_award_receipt,
)
from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointProjection,
    ArtifactTransactionProjection,
)
from menagerie.crawler.constants import InvocationOrigin
from menagerie.crawler.driver import DriverResult, DriverShutdown
from menagerie.crawler.identity import hash_bytes, payload_hash, stable_hash

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64
STARTED = "2026-07-16T12:00:00Z"
FINISHED = "2026-07-16T12:00:01Z"


def test_phase_zero_shared_contract_shapes_are_exact() -> None:
    """Freeze the cross-workstream shapes before any producer wires them."""

    assert tuple(field.name for field in fields(RuntimeMember)) == (
        "path",
        "sha256",
        "kind",
        "provenance",
    )
    assert tuple(field.name for field in fields(RuntimeLookupDirectory)) == (
        "path",
        "provenance",
    )
    assert tuple(field.name for field in fields(ExecutionReadManifestV2)) == (
        "manifest_version",
        "manifest_id",
        "stable_id",
        "work_id",
        "execution_identity",
        "code_manifest_identity",
        "environment_generation",
        "installed_package_inventory_sha256",
        "code_members",
        "runtime_members",
        "standard_input_asset",
        "lookup_directories",
    )
    assert tuple(field.name for field in fields(ShutdownInterruptionFact)) == (
        "invocation_id",
        "admission_boundary",
        "stable_id",
        "work_id",
        "execution_identity",
        "request_identity",
        "lease_id",
        "child_pid",
        "child_start_token",
        "child_pgid",
        "signal",
        "parent_observation",
        "partial_receipt",
    )
    assert tuple(field.name for field in fields(DriverResult)) == (
        "status",
        "terminal_models",
        "models_reduced",
        "paused_reason",
        "shutdown_interruption",
    )
    assert issubclass(DriverShutdown, BaseException)
    assert not issubclass(DriverShutdown, Exception)
    assert {origin.value for origin in InvocationOrigin} == {
        "ordinary-run",
        "manual-resume",
        "wake-callback",
    }
    assert tuple(field.name for field in fields(ArtifactTransactionProjection)) == (
        "stable_id",
        "work_id",
        "transaction_id",
        "final_event_id",
        "final_event_kind",
        "authorization_id",
        "accepted_gate_id",
        "reconstruction_path",
        "reconstruction_sha256",
        "reconstruction_inputs",
        "objects",
        "claims",
    )
    assert tuple(field.name for field in fields(ArtifactCheckpointProjection)) == (
        "transactions",
        "objects",
        "claims",
    )


def _observation(mode: str, signature: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build one closed successful raw worker observation.

    Parameters
    ----------
    mode:
        Requested runtime mode.
    signature:
        Optional output signature override.

    Returns
    -------
    dict[str, Any]
        Raw observation matching the frozen attempt-v3 shape.
    """

    return {
        "present": True,
        "receipt_sha256": None,
        "observed_recipe_revision": HASH_B,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": HASH_C,
        "observed_input_asset_sha256": None,
        "constructor_seconds": 0.1,
        "forward_seconds": 0.2,
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "forward_started": True,
        "forward_completed": True,
        "mode": mode,
        "input_signature": {"args": []},
        "output_signature": dict(signature or {"tree": {"leaf": 0}, "leaves": []}),
        "input_kind": "random-fallback",
        "input_asset": None,
        "input_note": "deterministic fixture",
        "parameter_count_total": 2,
        "parameter_count_trainable": 2,
        "native_framework": "torch",
        "delegated_method": "forward",
    }


def _raw_receipt(
    mode: str = "eval", signature: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Build one closed v3 raw award receipt.

    Parameters
    ----------
    mode:
        Requested runtime mode.
    signature:
        Optional output signature override.

    Returns
    -------
    dict[str, Any]
        Closed raw receipt.
    """

    return {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": f"nonce-{mode}",
        "request_sha256": HASH_A if mode == "eval" else HASH_D,
        "stable_id": "m_example",
        "work_id": "work-1",
        "execution_identity": HASH_A,
        "recipe_revision": HASH_B,
        "code_manifest_identity": HASH_C,
        "input_identity": HASH_D,
        "requested_mode": mode,
        "observation": _observation(mode, signature),
    }


def _supervisor(completion_line: str) -> dict[str, Any]:
    """Build parent-owned supervisor facts.

    Parameters
    ----------
    completion_line:
        Exact child completion marker.

    Returns
    -------
    dict[str, Any]
        Parent observation used by the attestation and attempt projection.
    """

    return {
        "exit_code": 0,
        "signal": None,
        "wall_seconds": 1.0,
        "cpu_seconds": 0.5,
        "peak_rss_bytes": 100,
        "timed_out": False,
        "rss_exceeded": False,
        "stdout_sha256": HASH_A,
        "stdout_bytes": 10,
        "stdout_tail": "",
        "stdout_completion_line": completion_line,
        "stderr_sha256": HASH_B,
        "stderr_bytes": 0,
        "stderr_tail": "",
        "full_log_local_path": ".crawl-local/logs/worker.log",
        "full_log_retention": "campaign",
    }


def _attempt(mode: str = "eval", attempt_no: int = 1) -> dict[str, Any]:
    """Build one fully associated v3 success attempt.

    Parameters
    ----------
    mode:
        Requested runtime mode.
    attempt_no:
        Deterministic attempt ordinal.

    Returns
    -------
    dict[str, Any]
        Candidate admission attempt.
    """

    raw = _raw_receipt(mode)
    line = completion_line_for_raw_award_receipt(raw)
    supervisor = _supervisor(line)
    parent = derive_parent_attestation(
        raw,
        line,
        supervisor,
        started_at=STARTED,
        finished_at=FINISHED,
    )
    policy = {
        "network_attempted": False,
        "socket_targets": [],
        "checkpoint_or_weight_read_attempted": False,
        "checkpoint_paths": [],
        "write_outside_scratch_attempted": False,
        "write_paths": [],
        "credentials_present": False,
        "torchlens_import_attempted": False,
        "cache_read_attempted": False,
    }
    return {
        "schema_version": "menagerie.crawler.attempt.v3",
        "attempt_id": f"attempt-{mode}-{attempt_no}",
        "attempt_no": attempt_no,
        "ledger_seq": attempt_no,
        "stable_id": "m_example",
        "work_id": "work-1",
        "stage": "forward",
        "mode": mode,
        "result": "succeeded",
        "started_at": STARTED,
        "finished_at": FINISHED,
        "identities": {"execution": HASH_A, "recipe": HASH_B},
        "invocation": {"mode": mode},
        "worker_receipt": raw["observation"],
        "supervisor_observation": supervisor,
        "policy_observation": policy,
        "error": None,
        "raw_award_receipt": raw,
        "raw_award_receipt_sha256": raw_award_receipt_sha256(raw),
        "parent_attestation": parent,
        "unattested_partial": None,
    }


def _failed_attempt(stage: str, reason: str, attempt_no: int = 1) -> dict[str, Any]:
    """Build one canonical same-work failed attempt.

    Parameters
    ----------
    stage, reason:
        Closed failure stage and reason.
    attempt_no:
        Attempt ordering ordinal.

    Returns
    -------
    dict[str, Any]
        Failure proof fact.
    """

    return {
        "attempt_id": f"failed-{stage}-{attempt_no}",
        "attempt_no": attempt_no,
        "ledger_seq": attempt_no,
        "stable_id": "m_example",
        "work_id": "work-1",
        "stage": stage,
        "mode": None,
        "result": "failed",
        "worker_receipt": {"present": False},
        "supervisor_observation": {"peak_rss_bytes": 7},
        "error": {
            "stage": stage,
            "reason_code": reason,
            "root_cause_fingerprint": stable_hash([stage, reason, attempt_no]),
        },
    }


def _terminal_gate(predicate: str) -> dict[str, Any]:
    """Build an accepted exact terminal-disposition gate.

    Parameters
    ----------
    predicate:
        Closed terminal predicate.

    Returns
    -------
    dict[str, Any]
        Minimal canonical gate facts consumed by the resolver.
    """

    return {
        "gate_id": f"gate-{predicate}",
        "gate_kind": "terminal_disposition",
        "gate_round": 1,
        "ledger_seq": 1,
        "items": [
            {
                "stable_id": "m_example",
                "work_id": "work-1",
                "verdict": "accurate",
                "integrity": {"verdict": "accurate"},
                "rung_check": {"selected_rung": "R5_SKIP", "verdict": "accurate"},
                "terminal_disposition": {
                    "predicate": predicate,
                    "handoff_proposal_id": None,
                    "handoff_sha256": None,
                    "verdict": "accepted",
                    "source_manifest_identity": HASH_A,
                    "source_ids": ["source-1"],
                    "evidence_identity": HASH_B,
                    "evidence_ids": ["evidence-1"],
                    "license_identity": HASH_C,
                },
            }
        ],
    }


def _persisted_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Add the canonical ledger self-hash to one attempt fixture.

    Parameters
    ----------
    attempt:
        Attempt fixture with a positive ledger sequence.

    Returns
    -------
    dict[str, Any]
        Persisted current-proof fixture.
    """

    persisted = deepcopy(dict(attempt))
    persisted["payload_sha256"] = payload_hash(persisted)
    return persisted


def _current_gate(
    item: Mapping[str, Any], *, gate_id: str = "gate-current", ledger_seq: int = 1
) -> dict[str, Any]:
    """Build a persisted current-v3 gate proof around one exact item.

    Parameters
    ----------
    item:
        Exact checker item.
    gate_id, ledger_seq:
        Immutable gate identity and ledger position.

    Returns
    -------
    dict[str, Any]
        Minimal proof-complete gate consumed by authority producers.
    """

    gate: dict[str, Any] = {
        "schema_version": "menagerie.crawler.gate.v3",
        "gate_id": gate_id,
        "ledger_seq": ledger_seq,
        "gate_kind": "metadata_batch",
        "batch_size": 1,
        "gate_round": 1,
        "gate_identity": HASH_A,
        "checker": {
            "provider": "openai",
            "model": "checker",
            "version": "v1",
            "prompt_sha256": HASH_B,
        },
        "items": [deepcopy(dict(item))],
        "author_result_schema_identity": HASH_C,
        "dispatcher_identity": HASH_D,
    }
    gate["result_envelope_sha256"] = stable_hash(
        {key: value for key, value in gate.items() if key != "ledger_seq"}
    )
    gate["payload_sha256"] = payload_hash(gate)
    return gate


def test_manifest_v2_compiler_inventories_files_without_semantic_root_grants(
    tmp_path: Path,
) -> None:
    """Exact members compile while lookup scaffolding grants no descendant bytes."""

    helper_path = tmp_path / "helper.py"
    helper_path.write_text("VALUE = 1\n", encoding="utf-8")
    code_path = tmp_path / "adapter.py"
    code_path.write_text("import helper\nVALUE = helper.VALUE\n", encoding="utf-8")
    interpreter_path = tmp_path / "python-runtime"
    interpreter_path.write_bytes(b"interpreter")
    metadata_path = tmp_path / "RECORD"
    metadata_path.write_text("adapter.py,sha256=fixture\n", encoding="utf-8")
    manifest = compile_execution_read_manifest_v2(
        stable_id="m_example",
        work_id="work-1",
        execution_identity=HASH_A,
        code_manifest_identity=HASH_B,
        environment_generation=HASH_C,
        installed_package_inventory_sha256=HASH_D,
        code_members=(
            RuntimeMember(
                code_path,
                hash_bytes(code_path.read_bytes()),
                "python-source",
                "accepted-model-code",
            ),
            RuntimeMember(
                helper_path,
                hash_bytes(helper_path.read_bytes()),
                "python-source",
                "static-model-import",
            ),
        ),
        runtime_members=(
            RuntimeMember(
                interpreter_path,
                hash_bytes(interpreter_path.read_bytes()),
                "interpreter",
                "environment-interpreter",
            ),
            RuntimeMember(
                metadata_path,
                hash_bytes(metadata_path.read_bytes()),
                "import-metadata",
                "installed-record",
            ),
        ),
        lookup_directories=(RuntimeLookupDirectory(tmp_path, "lookup-only"),),
    )
    assert manifest.manifest_version == "menagerie.crawler.execution-read-manifest.v2"
    assert {member.path for member in manifest.runtime_members} == {
        interpreter_path,
        metadata_path,
    }

    unrelated = tmp_path / "undeclared.txt"
    unrelated.write_text("not authority", encoding="utf-8")
    verify_execution_read_manifest_v2(manifest)


def test_manifest_v2_rejects_semantic_roots_and_detects_member_mutation(
    tmp_path: Path,
) -> None:
    """A directory-shaped grant rejects and an inventoried byte change stales proof."""

    helper_path = tmp_path / "undeclared_helper.py"
    helper_path.write_text("VALUE = 1\n", encoding="utf-8")
    importer_path = tmp_path / "importer.py"
    importer_path.write_text("import undeclared_helper\n", encoding="utf-8")
    with pytest.raises(AuthorityDerivationError, match="outside the executable member inventory"):
        compile_execution_read_manifest_v2(
            stable_id="m_example",
            work_id="work-1",
            execution_identity=HASH_A,
            code_manifest_identity=HASH_B,
            environment_generation=HASH_C,
            installed_package_inventory_sha256=HASH_D,
            code_members=(
                RuntimeMember(
                    importer_path,
                    hash_bytes(importer_path.read_bytes()),
                    "python-source",
                    "accepted-model-code",
                ),
            ),
            runtime_members=(),
            lookup_directories=(RuntimeLookupDirectory(tmp_path, "lookup-only"),),
        )

    with pytest.raises(AuthorityDerivationError, match="non-file or unknown kind"):
        compile_execution_read_manifest_v2(
            stable_id="m_example",
            work_id="work-1",
            execution_identity=HASH_A,
            code_manifest_identity=HASH_B,
            environment_generation=HASH_C,
            installed_package_inventory_sha256=HASH_D,
            code_members=(),
            runtime_members=(RuntimeMember(tmp_path, HASH_A, "runtime-root", "repository-root"),),
        )

    runtime_path = tmp_path / "worker.py"
    runtime_path.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = compile_execution_read_manifest_v2(
        stable_id="m_example",
        work_id="work-1",
        execution_identity=HASH_A,
        code_manifest_identity=HASH_B,
        environment_generation=HASH_C,
        installed_package_inventory_sha256=HASH_D,
        code_members=(),
        runtime_members=(
            RuntimeMember(
                runtime_path,
                hash_bytes(runtime_path.read_bytes()),
                "python-source",
                "worker-bootstrap",
            ),
        ),
    )
    runtime_path.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(AuthorityDerivationError, match="digest changed"):
        verify_execution_read_manifest_v2(manifest)


def test_current_proof_loaders_reject_legacy_attempts_and_gates() -> None:
    """Readable v2 facts cannot enter any current attempt or gate authority seam."""

    attempt = _persisted_attempt(_attempt())
    attempt["schema_version"] = "menagerie.crawler.attempt.v2"
    attempt["payload_sha256"] = payload_hash(attempt)
    with pytest.raises(AuthorityDerivationError, match="legacy rows lack v3 proof"):
        load_current_attempt_proof(attempt)

    item = _terminal_gate("needs-cuda")["items"][0]
    gate = _current_gate(item)
    gate["schema_version"] = "menagerie.crawler.gate.v2"
    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256"}
        }
    )
    gate["payload_sha256"] = payload_hash(gate)
    with pytest.raises(AuthorityDerivationError, match="legacy rows lack v3 proof"):
        load_current_gate_proof(gate)


@pytest.mark.parametrize(
    "mutation",
    ("legacy", "missing-raw", "missing-parent", "mutated-raw-digest"),
)
def test_every_award_counted_attempt_independently_replays_v3_proof(mutation: str) -> None:
    """A non-decisive confirmation slot cannot borrow decisive-slot authority."""

    decisive = _persisted_attempt(_attempt("eval", 1))
    confirmation = _persisted_attempt(_attempt("train", 2))
    control = authenticate_accepted_attempts(
        [decisive["attempt_id"], confirmation["attempt_id"]],
        [decisive, confirmation],
        stable_id="m_example",
        work_id="work-1",
        execution_identity=HASH_A,
    )
    assert tuple(proof.attempt_id for proof in control) == (
        decisive["attempt_id"],
        confirmation["attempt_id"],
    )
    if mutation == "legacy":
        confirmation["schema_version"] = "menagerie.crawler.attempt.v2"
    elif mutation == "missing-raw":
        confirmation["raw_award_receipt"] = None
    elif mutation == "missing-parent":
        confirmation["parent_attestation"] = None
    else:
        confirmation["raw_award_receipt_sha256"] = HASH_D
    confirmation["payload_sha256"] = payload_hash(confirmation)

    with pytest.raises(AuthorityDerivationError):
        authenticate_accepted_attempts(
            [decisive["attempt_id"], confirmation["attempt_id"]],
            [decisive, confirmation],
            stable_id="m_example",
            work_id="work-1",
            execution_identity=HASH_A,
        )


def test_gate_item_membership_is_exact_unique_and_ledger_owned() -> None:
    """Caller gate IDs and duplicate item memberships cannot select publication authority."""

    item = deepcopy(_terminal_gate("needs-cuda")["items"][0])
    digest = stable_hash(item)
    gate = _current_gate(item)
    owning_gate, owning_item = resolve_exact_gate_item_membership(
        [gate],
        accepted_gate_item=item,
        accepted_gate_item_sha256=digest,
    )
    assert owning_gate["gate_id"] == "gate-current"
    assert owning_item == item

    injected = deepcopy(item)
    injected["gate_id"] = "gate-foreign"
    with pytest.raises(AuthorityDerivationError, match="zero or multiple"):
        resolve_exact_gate_item_membership(
            [gate],
            accepted_gate_item=injected,
            accepted_gate_item_sha256=stable_hash(injected),
        )

    duplicate_gate = _current_gate(item, gate_id="gate-duplicate", ledger_seq=2)
    with pytest.raises(AuthorityDerivationError, match="zero or multiple"):
        resolve_exact_gate_item_membership(
            [gate, duplicate_gate],
            accepted_gate_item=item,
            accepted_gate_item_sha256=digest,
        )


def test_attempt_projection_replays_raw_parent_and_every_consumed_projection() -> None:
    """Raw bytes, completion, parent facts, and candidate projections are inseparable."""

    attempt = _attempt()
    authority = derive_attempt_projection(
        attempt["raw_award_receipt"],
        attempt["parent_attestation"],
        candidate_attempt=attempt,
    )
    assert authority.attempt_id == attempt["attempt_id"]
    assert authority.raw_award_receipt_sha256 == attempt["raw_award_receipt_sha256"]

    mutations = (
        ("stable_id", "m_other"),
        ("work_id", "work-other"),
        ("mode", "train"),
        ("result", "observed"),
    )
    for field, value in mutations:
        altered = deepcopy(attempt)
        altered[field] = value
        with pytest.raises(AuthorityDerivationError, match="attempt projection"):
            derive_attempt_projection(
                altered["raw_award_receipt"],
                altered["parent_attestation"],
                candidate_attempt=altered,
            )


def test_attempt_projection_rejects_raw_mutation_and_completion_line_reuse() -> None:
    """Altered raw bytes and a cross-mode completion witness cannot authenticate."""

    attempt = _attempt()
    altered_raw = deepcopy(attempt["raw_award_receipt"])
    altered_raw["input_identity"] = HASH_A
    with pytest.raises(AuthorityDerivationError, match="raw request/receipt"):
        derive_attempt_projection(
            altered_raw,
            attempt["parent_attestation"],
            candidate_attempt=attempt,
        )

    train = _attempt("train")
    train["supervisor_observation"]["stdout_completion_line"] = attempt["supervisor_observation"][
        "stdout_completion_line"
    ]
    with pytest.raises(AuthorityDerivationError):
        derive_attempt_projection(
            train["raw_award_receipt"],
            train["parent_attestation"],
            candidate_attempt=train,
        )


def test_mode_summary_is_structural_or_unverifiable_from_authenticated_receipts() -> None:
    """Structure is decidable, while missing frozen value-digest evidence fails closed."""

    train = _attempt("train")
    evaluation = _attempt("eval")
    summary = derive_mode_summary(train, evaluation)
    assert summary.comparison_state == "unverifiable"
    assert summary.classification == "unverifiable"
    assert summary.reason == "matching output signatures lack stable output value digests"

    evaluation = _attempt("eval")
    evaluation["raw_award_receipt"]["observation"]["output_signature"] = {
        "tree": {"tuple": []},
        "leaves": [],
    }
    evaluation["worker_receipt"] = evaluation["raw_award_receipt"]["observation"]
    raw = evaluation["raw_award_receipt"]
    line = completion_line_for_raw_award_receipt(raw)
    supervisor = _supervisor(line)
    evaluation["supervisor_observation"] = supervisor
    evaluation["parent_attestation"] = derive_parent_attestation(
        raw, line, supervisor, started_at=STARTED, finished_at=FINISHED
    )
    evaluation["raw_award_receipt_sha256"] = raw_award_receipt_sha256(raw)
    structural = derive_mode_summary(train, evaluation)
    assert structural.comparison_state == "verified"
    assert structural.classification == "structural"


def test_terminal_failure_uses_exact_decisive_stage_and_complete_mode_map() -> None:
    """An unrelated failure cannot prove a terminal and highest mode attempts win."""

    train_first = _attempt("train", 1)
    train_last = _attempt("train", 4)
    evaluation = _attempt("eval", 2)
    failure = _failed_attempt("source", "missing-mandatory-link", 5)
    proof = derive_terminal_proof(
        "m_example",
        "work-1",
        "failed:source",
        attempts=[train_first, train_last, evaluation, failure],
        proof_rule_identity=HASH_A,
    )
    assert proof.decisive_attempt_ids == (failure["attempt_id"],)
    assert proof.per_mode_attempt_ids == (
        ("train", train_last["attempt_id"]),
        ("eval", evaluation["attempt_id"]),
    )
    assert derive_per_mode_run(
        [train_first, train_last, evaluation, failure],
        stable_id="m_example",
        work_id="work-1",
    ) == {
        "train": {"attempt_id": train_last["attempt_id"], "status": "succeeded"},
        "eval": {"attempt_id": evaluation["attempt_id"], "status": "succeeded"},
    }
    assert proof.reason_code == "missing-mandatory-link"

    with pytest.raises(AuthorityDerivationError, match="same-stage"):
        derive_terminal_proof(
            "m_example",
            "work-1",
            "failed:source",
            attempts=[_failed_attempt("runner", "protocol-violation")],
            proof_rule_identity=HASH_A,
        )


def test_terminal_deferral_resolves_gate_source_and_literal_support() -> None:
    """A nonempty ID list alone does not prove a platform deferral."""

    sources = [{"source_id": "source-1", "url": "https://example.test/source"}]
    evidence = [
        {
            "evidence_id": "evidence-1",
            "source_id": "source-1",
            "supports": ["platform.needs-cuda"],
        }
    ]
    proof = derive_terminal_proof(
        "m_example",
        "work-1",
        "deferred:needs-cuda",
        attempts=[],
        gates=[_terminal_gate("needs-cuda")],
        source_manifest=sources,
        evidence_excerpts=evidence,
        source_manifest_identity=HASH_A,
        evidence_identity=HASH_B,
        license_identity=HASH_C,
        proof_rule_identity=HASH_A,
    )
    assert proof.platform_claim == "needs-cuda"
    assert proof.source_ids == ("source-1",)

    unsupported = deepcopy(evidence)
    unsupported[0]["supports"] = ["taxonomy.family"]
    with pytest.raises(AuthorityDerivationError, match="does not support"):
        derive_terminal_proof(
            "m_example",
            "work-1",
            "deferred:needs-cuda",
            attempts=[],
            gates=[_terminal_gate("needs-cuda")],
            source_manifest=sources,
            evidence_excerpts=unsupported,
            source_manifest_identity=HASH_A,
            evidence_identity=HASH_B,
            license_identity=HASH_C,
            proof_rule_identity=HASH_A,
        )


def test_terminal_skip_rechecks_r5_search_and_typed_evidence_predicate() -> None:
    """An accepted gate still cannot turn an unproved R5 recommendation into a skip."""

    sources = [{"source_id": "source-1", "url": "https://example.test/paper"}]
    evidence = [
        {
            "evidence_id": "evidence-1",
            "source_id": "source-1",
            "supports": ["source_resolution.sufficiency_gap"],
            "disposition": "insufficient-for-faithful-reimpl",
        }
    ]
    resolution: dict[str, Any] = {
        "rung": "R5_SKIP",
        "sufficiency_gap": "The source omits all hidden dimensions.",
        "search_report": {"conclusion": "No implementation or sufficient specification found."},
    }
    proof = derive_terminal_proof(
        "m_example",
        "work-1",
        "skipped:insufficient-description",
        attempts=[],
        gates=[_terminal_gate("insufficient-description")],
        source_manifest=sources,
        evidence_excerpts=evidence,
        source_resolution=resolution,
        source_manifest_identity=HASH_A,
        evidence_identity=HASH_B,
        license_identity=HASH_C,
        proof_rule_identity=HASH_A,
    )
    assert proof.gate_id == "gate-insufficient-description"

    invalid_resolution = deepcopy(resolution)
    invalid_resolution["sufficiency_gap"] = None
    with pytest.raises(AuthorityDerivationError, match="sufficiency gap"):
        derive_terminal_proof(
            "m_example",
            "work-1",
            "skipped:insufficient-description",
            attempts=[],
            gates=[_terminal_gate("insufficient-description")],
            source_manifest=sources,
            evidence_excerpts=evidence,
            source_resolution=invalid_resolution,
            source_manifest_identity=HASH_A,
            evidence_identity=HASH_B,
            license_identity=HASH_C,
            proof_rule_identity=HASH_A,
        )


def _context(*, variant: bool = False) -> AuthorityContext:
    """Build one mandatory active authority context.

    Parameters
    ----------
    variant:
        Whether the example stable ID is bound to a representative.

    Returns
    -------
    AuthorityContext
        Exact trust roots for dependency tests.
    """

    intake = {"stable_id": "m_example", "variant": "Large"}
    bindings = (
        {
            "m_example": {
                "representative_stable_id": "m_representative",
                "variant_token": "Large",
            }
        }
        if variant
        else {}
    )
    return AuthorityContext(
        active_intake_snapshot_id="snapshot-1",
        active_intake_snapshot_sha256=HASH_A,
        intake_by_stable_id={
            "m_example": intake,
            "m_representative": {"stable_id": "m_representative"},
        },
        family_bindings=bindings,
        author_prompt_identity=HASH_A,
        author_model_identity="author-model",
        author_schema_identity=HASH_B,
        author_dispatcher_identity=HASH_C,
        checker_prompt_identity=HASH_D,
        checker_model_identity="checker-model",
        checker_schema_identity=HASH_A,
        environment_generations={"core": HASH_B},
        reducer_policy_identity=HASH_C,
        runner_policy_identity=HASH_D,
        terminal_policy_identity=HASH_A,
        publication_policy_identity=HASH_B,
    )


def test_family_authority_is_intake_derived_on_terminal_lanes() -> None:
    """Trusted variant binding cannot disappear when terminal record prose is absent."""

    context = _context(variant=True)
    representative = {
        "stable_id": "m_representative",
        "record_revision": HASH_C,
        "accuracy_gate": {"gate_id": "gate-representative"},
        "dependency_vector": {"proposal_identity": HASH_D},
    }
    authority = derive_family_authority(context, "m_example", representative_record=representative)
    assert authority.representative_stable_id == "m_representative"
    assert authority.template_source_revision == HASH_C

    with pytest.raises(AuthorityDerivationError, match="representative"):
        derive_family_authority(context, "m_example")


def test_dependency_vector_and_currency_bind_active_context_for_non_runs() -> None:
    """A source terminal binds active roots without unrelated runner/environment axes."""

    context = _context()
    failure = _failed_attempt("source", "identity-unresolved")
    proof = derive_terminal_proof(
        "m_example",
        "work-1",
        "failed:source",
        attempts=[failure],
        proof_rule_identity=context.terminal_policy_identity,
    )
    vector = derive_dependency_vector(
        context,
        stable_id="m_example",
        terminal_proof=proof,
        source_manifest_identity=HASH_D,
    )
    assert vector.runner_identity == DependencyState.NOT_APPLICABLE
    record = {
        "schema_version": "menagerie.crawler.model.v3",
        "stable_id": "m_example",
        "status": {"code": "failed:source"},
        "dependency_vector": vector,
        "family_authority": family_authority_projection(
            derive_family_authority(context, "m_example")
        ),
    }
    assert validate_currency(context, record, terminal_proof=proof) is None

    stale = deepcopy(record)
    stale_vector = deepcopy(vector.__dict__)
    stale_vector["terminal_rule_identity"] = HASH_D
    stale["dependency_vector"] = stale_vector
    assert validate_currency(context, stale, terminal_proof=proof) == (
        "dependency-vector: stale terminal_rule_identity"
    )


def test_terminal_observation_is_deterministic_and_attempt_complete() -> None:
    """Terminal observed facts include every same-work attempt in canonical order."""

    later = _failed_attempt("runner", "protocol-violation", 2)
    earlier = _failed_attempt("source", "identity-unresolved", 1)
    observed = derive_terminal_observation(
        [later, earlier], stable_id="m_example", work_id="work-1"
    )
    assert observed["measurement_attempt_ids"] == [earlier["attempt_id"], later["attempt_id"]]
    assert observed["snippet_sha256"] == stable_hash(observed["snippet"])


def test_pure_closure_and_execution_identities_bind_every_resolved_component() -> None:
    """The integrator can replace driver-local identity code without dynamic imports or I/O."""

    runner = derive_runner_identity(
        {"worker.main": HASH_A}, platform_name="linux", selected_asset_identity=HASH_B
    )
    award = derive_award_closure_identity({"reducer.append_model": HASH_C}, {"model.v3": HASH_D})
    execution = derive_execution_identity(
        stable_id="m_example",
        recipe_revision=HASH_A,
        environment_generation=HASH_B,
        runner_identity=runner,
        target="linux-x86_64",
        machine_class="x86_64",
        input_seed=0,
        framework="torch",
        recipe_type="typed-adapter",
        award_closure_identity=award,
        runtime_dependencies_identity=HASH_C,
        device="cpu",
    )
    changed = derive_execution_identity(
        stable_id="m_example",
        recipe_revision=HASH_A,
        environment_generation=HASH_B,
        runner_identity=runner,
        target="linux-x86_64",
        machine_class="x86_64",
        input_seed=1,
        framework="torch",
        recipe_type="typed-adapter",
        award_closure_identity=award,
        runtime_dependencies_identity=HASH_C,
        device="cpu",
    )
    assert execution != changed
