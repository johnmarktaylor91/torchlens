"""Full-contract synthetic fixtures for crawler Slice A tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.author_dispatch import AuthorResultBinding, ProposedAuthorResult
from menagerie.crawler.authority import (
    AuthorityContext,
    build_authority_context,
    completion_line_for_raw_award_receipt,
    derive_parent_attestation,
    raw_award_receipt_sha256,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3 as ATTEMPT_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3 as AUTHOR_PROPOSAL_SCHEMA_VERSION,
    AUTHOR_PROMPT_NAME,
    CHECKER_PROMPT_NAME,
    GATE_SCHEMA_VERSION_V3 as GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3 as MODEL_SCHEMA_VERSION,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    SourceRung,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicensedArtifact,
    RedistributionClass,
    classify_redistribution,
    recompute_license_decision,
)
from menagerie.crawler.metadata import (
    authored_fact_leaves,
    recompute_accepted_identities,
)
from menagerie.crawler.proposal import ProposalValidationReport
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
    RetentionClass,
)
from menagerie.crawler.standard_inputs import ASSET_ROOT

HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64
NOW = "2026-07-14T12:00:00Z"

_FACT_KEYS = (
    "identity",
    "taxonomy",
    "external_metadata",
    "website",
    "people_and_origin",
    "dates",
    "citation",
    "licenses",
    "source_resolution",
    "evidence",
    "implementation",
    "input_contract",
    "modes",
    "fidelity",
)


def make_licensed_artifact_fixture(
    mirrors: MirrorStore,
    content: bytes,
    *,
    staged_path: Path,
    origin: ArtifactOrigin,
    evidence: tuple[LicenseEvidence, ...],
    media_type: str = "application/octet-stream",
) -> LicensedArtifact:
    """Materialize explicit legacy mirror data for license-sweep unit fixtures.

    This test-only constructor carries no production authorization meaning. Tests of
    canonical publication use artifact transactions and reducer authorization directly.

    Parameters
    ----------
    mirrors, content, staged_path, origin, evidence, media_type:
        Explicit fixture storage and evidence inputs.

    Returns
    -------
    LicensedArtifact
        Data-only legacy manifest row used by checkpoint/license unit tests.
    """

    redistribution = classify_redistribution(evidence)
    public = redistribution is RedistributionClass.PUBLIC_OK
    manifest = mirrors.put(
        content,
        mirror_class=MirrorClass.PUBLIC if public else MirrorClass.PRIVATE,
        retention_class=(
            RetentionClass.DURABLE_PUBLIC if public else RetentionClass.RESTRICTED_PRIVATE
        ),
        origin=origin,
        media_type=media_type,
    )
    return LicensedArtifact(
        staged_path=staged_path,
        manifest=manifest,
        decision=recompute_license_decision(manifest.content_sha256, evidence),
    )


def make_authority_context(
    stable_ids: Any,
    *,
    snapshot_id: str = "snapshot-test",
    snapshot_sha256: str = HASH,
) -> AuthorityContext:
    """Build the mandatory production-shaped authority context for tests.

    Parameters
    ----------
    stable_ids:
        Iterable of stable model identifiers admitted by the synthetic intake.
    snapshot_id, snapshot_sha256:
        Exact synthetic or materialized intake snapshot identity.

    Returns
    -------
    AuthorityContext
        Context derived from exact shipped contract bytes and synthetic intake rows.
    """

    rows = tuple({"stable_id": str(stable_id)} for stable_id in stable_ids)
    return build_authority_context(
        active_intake_snapshot_id=snapshot_id,
        active_intake_snapshot_sha256=snapshot_sha256,
        intake_rows=rows,
        author_model="claude",
        author_version="test",
        checker_model="codex",
        checker_version="test",
        environment_generations={"env-test": HASH},
    )


def _checker_prompt_hash() -> str:
    """Return the exact frozen checker prompt byte hash used by fixtures."""

    path = Path(__file__).parents[1] / "prompts" / f"{CHECKER_PROMPT_NAME}.txt"
    return hash_bytes(path.read_bytes())


def _author_prompt_hash() -> str:
    """Return the exact frozen author prompt byte hash used by fixtures."""

    path = Path(__file__).parents[1] / "prompts" / f"{AUTHOR_PROMPT_NAME}.txt"
    return hash_bytes(path.read_bytes())


def _model_facts(model: dict[str, Any]) -> dict[str, Any]:
    """Extract proposal fact roots from a synthetic canonical model."""

    return {key: model[key] for key in _FACT_KEYS}


def _bind_model_identities(model: dict[str, Any]) -> None:
    """Populate synthetic model identity claims from its exact accepted facts."""

    identities = recompute_accepted_identities(
        _model_facts(model),
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    model["evidence"]["evidence_identity"] = identities.evidence
    model["implementation"]["recipe_revision"] = identities.recipe
    # Recipe includes evidence/implementation fields, so recompute once after embedding
    # the first-pass derived values.
    identities = recompute_accepted_identities(
        _model_facts(model),
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    model["implementation"]["recipe_revision"] = identities.recipe
    model["accuracy_gate"]["vet_identity"] = identities.vet
    model["accuracy_gate"]["prompt_sha256"] = _checker_prompt_hash()


def make_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str = "attempt-1",
    execution_identity: str = HASH,
    mode: Optional[str] = "eval",
) -> dict[str, Any]:
    """Build a complete valid forward attempt.

    Parameters
    ----------
    stable_id:
        Attempt model ID.
    attempt_id:
        Immutable attempt ID.
    execution_identity:
        Execution identity bound to the receipt.
    mode:
        Forward mode.

    Returns
    -------
    dict[str, Any]
        Complete attempt.v2 payload.
    """

    model = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "ledger_seq": 1,
        "payload_sha256": HASH,
        "work_id": f"work-{stable_id}",
        "stable_id": stable_id,
        "attempt_no": 1,
        "parent_attempt_id": None,
        "actor": "worker",
        "stage": "forward",
        "mode": mode,
        "started_at": NOW,
        "finished_at": NOW,
        "result": "succeeded",
        "attempted_rungs": ["R1_LIBRARY"],
        "retries": {
            "stage_attempt": 1,
            "root_cause_repeat": 0,
            "author_round": 0,
            "gate_round": 0,
        },
        "identities": {
            "source": HASH,
            "evidence": HASH,
            "recipe": HASH,
            "environment": HASH,
            "execution": execution_identity,
            "runner": HASH,
            "author_prompt": HASH,
            "checker_prompt": HASH,
        },
        "environment": {
            "family": "core",
            "target": "test",
            "env_id": "env-test",
            "lock_sha256": HASH,
            "resolved_export_sha256": HASH,
            "python": "3.11",
            "packages_manifest_sha256": HASH,
            "compiler_identity": "test-compiler",
            "sdk_identity": "test-sdk",
        },
        "host": {
            "machine_id": "machine-test",
            "os": "linux",
            "os_build": "test",
            "architecture": "x86_64",
            "cpu": "test-cpu",
            "ram_bytes": 1024,
            "accelerator": None,
            "accelerator_runtime": None,
        },
        "invocation": {
            "argv": ["python", "worker.py"],
            "cwd": "/scratch",
            "safe_env": {"OFFLINE": "1"},
            "seed": 0,
            "device": "cpu",
            "mode": mode,
            "network_policy": "offline",
            "timeout_seconds": 300,
            "rss_limit_bytes": 1024,
            "scratch_limit_bytes": 1024,
        },
        "worker_receipt": {
            "present": True,
            "receipt_sha256": HASH,
            "observed_recipe_revision": HASH,
            "observed_adapter_sha256": None,
            "observed_code_manifest_sha256": HASH,
            "observed_input_asset_sha256": hash_bytes((ASSET_ROOT / "image.ppm").read_bytes()),
            "constructor_started": True,
            "constructor_completed": True,
            "input_completed": True,
            "forward_started": True,
            "forward_completed": True,
            "mode": mode,
            "input_signature": {
                "tree": {
                    "args": {"tuple": [{"leaf": 0}]},
                    "kwargs": {},
                },
                "leaves": [
                    {
                        "path": "input.args[0]",
                        "kind": "tensor",
                        "shape": [1, 3, 8, 8],
                        "dtype": "float32",
                        "device": "cpu",
                        "python_type": "torch.Tensor",
                    }
                ],
            },
            "output_signature": {
                "tree": {"leaf": 0},
                "leaves": [
                    {
                        "path": "output",
                        "kind": "tensor",
                        "shape": [1, 2],
                        "dtype": "float32",
                        "device": "cpu",
                        "python_type": "torch.Tensor",
                    }
                ],
            },
            "input_kind": "standard-image",
            "input_asset": (
                f"standard:image.ppm:{hash_bytes((ASSET_ROOT / 'image.ppm').read_bytes())}"
            ),
            "input_note": "canonical test image",
            "parameter_count_total": 2,
            "parameter_count_trainable": 2,
            "native_framework": "pytorch",
            "delegated_method": "forward",
            "constructor_seconds": 0.1,
            "forward_seconds": 0.1,
        },
        "supervisor_observation": {
            "exit_code": 0,
            "signal": None,
            "wall_seconds": 0.1,
            "cpu_seconds": 0.1,
            "peak_rss_bytes": 128,
            "stdout_sha256": HASH,
            "stdout_bytes": 0,
            "stdout_tail": "",
            "stdout_completion_line": None,
            "stderr_sha256": HASH,
            "stderr_bytes": 0,
            "stderr_tail": "",
            "full_log_local_path": "/logs/test",
            "full_log_retention": "campaign",
        },
        "policy_observation": {
            "network_attempted": False,
            "socket_targets": [],
            "checkpoint_or_weight_read_attempted": False,
            "checkpoint_paths": [],
            "write_outside_scratch_attempted": False,
            "write_paths": [],
            "credentials_present": False,
            "torchlens_import_attempted": False,
            "cache_read_attempted": False,
        },
        "error": None,
        "defer_evidence": None,
    }
    reference = make_model(stable_id, accepted=True)
    facts = _model_facts(reference)
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="test",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    model["identities"].update(
        {
            "source": identities.source,
            "evidence": identities.evidence,
            "recipe": identities.recipe,
            "checker_prompt": _checker_prompt_hash(),
        }
    )
    receipt = model["worker_receipt"]
    receipt["receipt_sha256"] = None
    receipt["observed_recipe_revision"] = identities.recipe
    raw_receipt = {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": f"nonce-{attempt_id}",
        "request_sha256": HASH,
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "execution_identity": execution_identity,
        "recipe_revision": identities.recipe,
        "code_manifest_identity": HASH,
        "input_identity": hash_bytes((ASSET_ROOT / "image.ppm").read_bytes()),
        "requested_mode": mode,
        "observation": deepcopy(receipt),
    }
    completion_line = completion_line_for_raw_award_receipt(raw_receipt)
    completion_bytes = (completion_line + "\n").encode("utf-8")
    observation = model["supervisor_observation"]
    observation["stdout_sha256"] = hash_bytes(completion_bytes)
    observation["stdout_bytes"] = len(completion_bytes)
    # The public record keeps only the parent-attested TorchLens marker; arbitrary
    # worker stdout belongs in the gitignored local diagnostic sidecar.
    observation["stdout_completion_line"] = completion_line
    model.update(
        {
            "execution_read_manifest_identity": HASH,
            "raw_award_receipt": raw_receipt,
            "raw_award_receipt_sha256": raw_award_receipt_sha256(raw_receipt),
            "parent_attestation": derive_parent_attestation(
                raw_receipt,
                completion_line,
                observation,
                started_at=NOW,
                finished_at=NOW,
            ),
            "unattested_partial": None,
        }
    )
    return model


def make_failed_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str = "attempt-1",
    stage: str = "source",
    reason_code: str = "identity-unresolved",
) -> dict[str, Any]:
    """Build one reducer-valid failed attempt for terminal-evidence tests.

    Parameters
    ----------
    stable_id, attempt_id:
        Exact model and immutable attempt identities.
    stage, reason_code:
        Closed failure stage and reason.

    Returns
    -------
    dict[str, Any]
        Complete failed attempt payload with redacted diagnostics.
    """

    attempt = make_attempt(stable_id, attempt_id=attempt_id)
    diagnostic = {
        "redaction": "externally-controlled-text-v1",
        "content_sha256": HASH,
        "local_path": f".crawl-local/diagnostics/{attempt_id}.json",
        "diagnostic_key": "$.error.message",
    }
    mode = "eval" if stage == "forward" else None
    attempt.update(
        {
            "actor": "driver",
            "stage": stage,
            "mode": mode,
            "result": "failed",
            "environment": None,
            "error": {
                "stage": stage,
                "reason_code": reason_code,
                "exception_type": "builtins.RuntimeError",
                "message": diagnostic,
                "traceback": None,
                "no_traceback_reason": "synthetic failure has no traceback",
                "native_crash": False,
                "root_cause_fingerprint": HASH,
                "details": {},
            },
        }
    )
    attempt["identities"]["environment"] = None
    attempt["identities"]["execution"] = None
    attempt["invocation"].update({"mode": mode, "network_policy": "not-invoked"})
    attempt["worker_receipt"].update(
        {
            "present": False,
            "receipt_sha256": None,
            "observed_recipe_revision": None,
            "observed_adapter_sha256": None,
            "observed_code_manifest_sha256": None,
            "observed_input_asset_sha256": None,
            "constructor_started": False,
            "constructor_completed": False,
            "input_completed": False,
            "forward_started": False,
            "forward_completed": False,
            "mode": mode,
            "input_signature": None,
            "output_signature": None,
            "input_kind": None,
            "input_asset": None,
            "input_note": "worker was not invoked for this synthetic failure",
            "parameter_count_total": None,
            "parameter_count_trainable": None,
            "native_framework": None,
            "delegated_method": None,
        }
    )
    attempt["supervisor_observation"].update(
        {
            "exit_code": None,
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
            "stdout_sha256": None,
            "stdout_bytes": 0,
            "stdout_completion_line": None,
            "stderr_sha256": None,
            "stderr_bytes": 0,
            "full_log_local_path": "driver-observed",
        }
    )
    attempt["raw_award_receipt"] = None
    attempt["raw_award_receipt_sha256"] = None
    parent_attestation = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": f"driver-{attempt_id}",
        "request_sha256": HASH,
        "completion_line_sha256": None,
        "named_raw_award_receipt_sha256": None,
        "exit_code": None,
        "signal": None,
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": 0,
        "stdout_sha256": hash_bytes(b""),
        "stderr_sha256": hash_bytes(b""),
        "started_at": NOW,
        "finished_at": NOW,
    }
    parent_attestation["attestation_sha256"] = stable_hash(parent_attestation)
    attempt["parent_attestation"] = parent_attestation
    attempt["unattested_partial"] = None
    return attempt


def rebind_attempt_raw_proof(attempt: dict[str, Any]) -> dict[str, Any]:
    """Rebuild exact raw receipt and parent proof after a test mutates an attempt."""

    receipt = attempt["worker_receipt"]
    raw_receipt = {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": f"nonce-{attempt['attempt_id']}",
        "request_sha256": HASH,
        "stable_id": attempt["stable_id"],
        "work_id": attempt["work_id"],
        "execution_identity": attempt["identities"]["execution"],
        "recipe_revision": attempt["identities"]["recipe"],
        "code_manifest_identity": receipt["observed_code_manifest_sha256"],
        "input_identity": receipt["observed_input_asset_sha256"] or HASH,
        "requested_mode": attempt["mode"],
        "observation": deepcopy(receipt),
    }
    line = completion_line_for_raw_award_receipt(raw_receipt)
    completion_bytes = (line + "\n").encode("utf-8")
    supervisor = attempt["supervisor_observation"]
    supervisor["stdout_completion_line"] = line
    supervisor["stdout_sha256"] = hash_bytes(completion_bytes)
    supervisor["stdout_bytes"] = len(completion_bytes)
    attempt["raw_award_receipt"] = raw_receipt
    attempt["raw_award_receipt_sha256"] = raw_award_receipt_sha256(raw_receipt)
    attempt["parent_attestation"] = derive_parent_attestation(
        raw_receipt,
        line,
        supervisor,
        started_at=str(attempt["started_at"]),
        finished_at=str(attempt["finished_at"]),
    )
    return attempt


def rebind_nonaward_parent_proof(attempt: dict[str, Any]) -> dict[str, Any]:
    """Rebuild exact parent-only proof after a fixture becomes non-awarding.

    Parameters
    ----------
    attempt:
        Current-v3 failed or observed attempt fixture.

    Returns
    -------
    dict[str, Any]
        The same fixture with closed empty-stream parent attestation.
    """

    parent = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": f"driver-{attempt['attempt_id']}",
        "request_sha256": HASH,
        "completion_line_sha256": None,
        "named_raw_award_receipt_sha256": None,
        "exit_code": attempt["supervisor_observation"].get("exit_code"),
        "signal": attempt["supervisor_observation"].get("signal"),
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": attempt["supervisor_observation"].get("peak_rss_bytes"),
        "stdout_sha256": attempt["supervisor_observation"].get("stdout_sha256") or hash_bytes(b""),
        "stderr_sha256": attempt["supervisor_observation"].get("stderr_sha256") or hash_bytes(b""),
        "started_at": attempt["started_at"],
        "finished_at": attempt["finished_at"],
    }
    parent["attestation_sha256"] = stable_hash(parent)
    attempt["parent_attestation"] = parent
    return attempt


def bind_terminal_attempts(model: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind a terminal model fixture to exact canonical attempt observations.

    Parameters
    ----------
    model:
        Non-run model fixture to update in place.
    attempts:
        Ordered canonical attempts supporting its terminal status.

    Returns
    -------
    dict[str, Any]
        The updated model fixture.
    """

    from menagerie.crawler.authority import derive_terminal_observation

    attempt_ids = [str(attempt["attempt_id"]) for attempt in attempts]
    model["status"]["attempt_ids"] = attempt_ids
    model["execution"]["accepted_attempt_ids"] = []
    model["observed"] = derive_terminal_observation(
        attempts,
        stable_id=str(model["stable_id"]),
        work_id=str(attempts[0]["work_id"]) if attempts else "not-applicable",
    )
    model["modes"]["per_mode_run"] = {
        str(attempt["mode"]): {
            "attempt_id": attempt["attempt_id"],
            "status": attempt["result"],
        }
        for attempt in attempts
        if attempt.get("mode") in model["modes"]["meaningful_modes"]
    }
    return model


def make_gate(
    stable_ids: Optional[list[str]] = None,
    *,
    gate_id: str = "gate-1",
    gate_kind: str = "metadata_batch",
    vet_identity: Optional[str] = None,
    fidelity_identity: Optional[str] = None,
) -> dict[str, Any]:
    """Build a complete valid metadata-batch or fidelity gate.

    Parameters
    ----------
    stable_ids:
        Item stable IDs. Metadata defaults to ten IDs; fidelity uses one.
    gate_id:
        Immutable gate ID.
    gate_kind:
        ``metadata_batch`` or ``fidelity``.
    vet_identity, fidelity_identity:
        Item identities.

    Returns
    -------
    dict[str, Any]
        Complete gate.v2 payload.
    """

    if stable_ids is None:
        stable_ids = [f"m_{index}" for index in range(10)]
    fidelity_required = gate_kind == "fidelity"
    items: list[dict[str, Any]] = []
    for stable_id in stable_ids:
        model = make_model(stable_id, accepted=True)
        item_vet_identity = str(vet_identity or model["accuracy_gate"]["vet_identity"])
        items.append(
            {
                "work_id": f"work-{stable_id}",
                "campaign_root_work_id": f"work-{stable_id}",
                "stable_id": stable_id,
                "family_representative_id": stable_id,
                "fidelity_identity": fidelity_identity if fidelity_required else None,
                "vet_identity": item_vet_identity,
                "verified_hashes": {
                    "proposal": HASH,
                    "source_manifest": HASH,
                    "evidence": HASH,
                    "code": None,
                    "source_to_code_map": HASH,
                    "family_template": None,
                },
                "integrity": {
                    "verdict": "accurate",
                    "hash_mismatches": [],
                    "excerpt_discrepancies": [],
                    "locator_failures": [],
                },
                "verdict": "accurate",
                "field_checks": [
                    {
                        "field": field,
                        "verdict": "accurate",
                        "evidence_ids": ["evidence-1"],
                        "checked_source_ids": ["source-1"],
                        "reason": "supported",
                        "required_repair": None,
                    }
                    for field in authored_fact_leaves(
                        _model_facts(model), schema_version=MODEL_SCHEMA_VERSION
                    )
                ],
                "fidelity": {
                    "required": fidelity_required,
                    "verdict": "match" if fidelity_required else "not-applicable",
                    "material_checks": [],
                    "unsupported_choices": [],
                    "contradictions": [],
                    "omissions": [],
                    "permanent_scar": False,
                },
                "rung_check": {
                    "selected_rung": "R1_LIBRARY",
                    "highest_applicable": "R1_LIBRARY",
                    "verdict": "accurate",
                    "findings": [],
                },
                "unsupported_claims": [],
                "required_repairs": [],
                "confidence": "high",
                "terminal_disposition": None,
            }
        )
    proposal = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_id": gate_id,
        "ledger_seq": 1,
        "payload_sha256": HASH,
        "gate_kind": gate_kind,
        "batch_size": len(items),
        "gate_round": 1,
        "gate_identity": HASH,
        "checker": {
            "provider": "openai",
            "model": "codex",
            "version": "test",
            "prompt_sha256": _checker_prompt_hash(),
            "started_at": NOW,
            "finished_at": NOW,
        },
        "items": items,
        "result_envelope_sha256": HASH,
        "author_result_schema_identity": HASH,
        "dispatcher_identity": HASH,
    }
    proposal["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in proposal.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return proposal


def _citation() -> dict[str, Any]:
    """Return complete cited-work metadata.

    Returns
    -------
    dict[str, Any]
        Citation block.
    """

    return {
        "status": "present",
        "title": "Example Model",
        "authors": ["A. Author"],
        "year": 2020,
        "venue": "TestConf",
        "arxiv_id": None,
        "doi": None,
        "openreview_id": None,
        "url": "https://example.com/paper",
        "bibtex": None,
        "source_evidence_ids": ["evidence-1"],
    }


def make_model(
    stable_id: str = "m_example",
    *,
    accepted: bool = False,
    status_code: str = "runs",
    attempt_id: str = "attempt-1",
) -> dict[str, Any]:
    """Build a complete valid model revision.

    Parameters
    ----------
    stable_id:
        Model stable ID.
    accepted:
        Whether to populate every gated source-read block.
    status_code:
        Closed terminal status code.
    attempt_id:
        Per-mode accepted forward attempt.

    Returns
    -------
    dict[str, Any]
        Complete model.v3 payload with syntactically valid reducer-owned authority fields.
    """

    source = {
        "source_id": "source-1",
        "role": "implementation",
        "kind": "repository",
        "url": "https://example.com/model",
        "revision_kind": "commit",
        "revision": "abc123",
        "locator": "model.py",
        "content_sha256": HASH,
        "byte_count": 100,
        "media_type": "text/x-python",
        "retrieved_at": NOW,
        "fetch_recipe": "https-get",
        "mirror_class": "public",
        "mirror_digest": HASH,
    }
    metadata_blocks: dict[str, Any]
    if accepted:
        citation = _citation()
        metadata_blocks = {
            "taxonomy": {
                "family": "ExampleNet",
                "domains": ["vision"],
                "tasks": ["classification"],
                "modalities": ["vision"],
                "era": "modern",
                "architecture_tags": ["CNN"],
                "novel_ops": [],
            },
            "external_metadata": {
                "modality": ["vision"],
                "architecture_class": ["CNN"],
                "domain": ["computer vision"],
                "task": ["classification"],
                "field": "machine learning",
                "subfield": "computer vision",
                "paradigm": ["supervised"],
                "lineage": [],
                "predecessors": [],
                "tags": ["example"],
                "keywords": ["cnn"],
                "venue": "TestConf",
                "family": "ExampleNet",
                "era": "modern",
                "year": 2020,
                "country": "US",
                "authors": ["A. Author"],
                "institution": ["Example Lab"],
                "citation": citation,
                "license": "Apache-2.0",
                "key_contribution": "A grounded example.",
                "description": "A small source-grounded example network.",
                "original_framework": "pytorch",
                "run_framework": "pytorch",
                "modes": {
                    "meaningful_modes": ["eval"],
                    "train_eval_divergence": "none",
                },
            },
            "website": {
                "kind": "family-representative",
                "tagline": "A compact example network",
                "description": "A source-grounded example. It is used for integrity tests.",
                "key_contribution": "A grounded example.",
                "voice_version": "v1",
                "family_grounding_id": "grounding-1",
                "template_source_model_id": None,
                "variant_parameter_input_line": None,
                "template_hash": None,
            },
            "people_and_origin": {
                "authors": ["A. Author"],
                "labs": ["Example Lab"],
                "institutions": ["Example Institute"],
                "origin_countries": ["US"],
                "country_basis": "institution affiliation",
                "country_confidence": "high",
                "country_note": "Grounded in the paper.",
            },
            "dates": {
                "year": 2020,
                "year_basis": "paper publication",
                "first_public_date": "2020-01-01",
                "first_public_date_basis": "repository release",
            },
            "citation": citation,
            "licenses": {
                "code": {
                    "spdx": "Apache-2.0",
                    "status": "declared",
                    "source_id": "source-1",
                    "locator": "LICENSE",
                    "evidence_ids": ["evidence-1"],
                },
                "paper_text": {"status": "linked-not-redistributed", "source_id": "source-1"},
                "weights": {"status": "not-used"},
                "data": {
                    "spdx": None,
                    "status": "not-applicable",
                    "source_id": None,
                    "evidence_ids": [],
                },
                "redistribution_class": "public-compatible",
            },
        }
    else:
        metadata_blocks = {
            "taxonomy": None,
            "external_metadata": None,
            "website": None,
            "people_and_origin": None,
            "dates": None,
            "citation": None,
            "licenses": None,
        }
    kind = status_code.split(":", 1)[0]
    stage = status_code.split(":", 1)[1] if kind == "failed" else None
    reason = "identity-unresolved" if stage == "source" else None
    model = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "record_seq": 1,
        "record_revision": HASH,
        "parent_revision": None,
        "created_at": NOW,
        "revised_by": {"actor": "driver"},
        "authored_metadata_state": "accepted" if accepted else "pending",
        "family_variant_derivation": None,
        "intake": {
            "snapshot_id": "snapshot-1",
            "snapshot_sha256": HASH,
            "legacy_row_sha256": None,
            "legacy_recipe_sha256": None,
            "legacy_module_sha256": None,
            "legacy_claims_untrusted": True,
            "preserved_legacy_flags": [],
            "discovery_sources": ["master_catalog"],
        },
        "identity": {
            "canonical_name": "ExampleNet",
            "aliases": [],
            "acronym": None,
            "variant": "base",
            "variant_scope": "family",
            "family_representative_id": stable_id,
            "duplicate_of": None,
            "alias_of": None,
        },
        **metadata_blocks,
        "source_resolution": {
            "rung": "R1_LIBRARY",
            "decision": "official implementation",
            "rung_evidence": "source-1",
            "sufficiency_gap": None,
            "searched_at": NOW,
            "attempted_rungs": [
                {
                    "rung": "R1_LIBRARY",
                    "result": "selected",
                    "reason_code": "available",
                    "evidence_ids": ["evidence-1"],
                }
            ],
            "search_report": {
                "queries": ["ExampleNet implementation"],
                "places_checked": ["web"],
                "links_checked": ["https://example.com/model"],
                "languages_checked": ["en"],
                "archives_checked": [],
                "started_at": NOW,
                "finished_at": NOW,
                "conclusion": "Official implementation found.",
            },
            "mandatory_link_status": "ok",
            "primary_source_id": "source-1",
            "sources": [source],
        },
        "evidence": {
            "excerpts": [
                {
                    "evidence_id": "evidence-1",
                    "source_id": "source-1",
                    "locator": "README:1",
                    "text": "ExampleNet is a small convolutional network.",
                    "text_sha256": hash_bytes(b"ExampleNet is a small convolutional network."),
                    "supports": ["identity.canonical_name"],
                    "family_level": True,
                    "disposition": "supporting",
                    "license_disposition": "short-excerpt-committed",
                }
            ],
            "coverage": {
                "all_agent_fields_have_support": accepted,
                "missing_support": [] if accepted else ["authored_metadata"],
                "family_grounding_complete": accepted,
            },
            "evidence_identity": HASH,
            "family_grounding_path": None,
        },
        "implementation": {
            "original_framework": "pytorch",
            "run_framework": "pytorch",
            "native_object_type": "torch.nn.Module",
            "native_call_method": "forward",
            "transparent_forward_adapter": True,
            "recipe_type": "declarative-library",
            "code_path": None,
            "code_sha256": None,
            "builder_symbol": None,
            "dummy_call_symbol": None,
            "library_recipe": {
                "distribution": "example",
                "version": "1.0",
                "artifact_sha256": HASH,
                "module": "example",
                "symbol": "ExampleNet",
                "kwargs": {"weights": None},
                "pretrained_disable_fields": ["weights"],
            },
            "upstream_files": [],
            "patches": [],
            "source_to_code_map": [],
            "declared_choices": [],
            "initialization": {
                "policy": "random",
                "pretrained_disabled": True,
                "source_specified_choices": [],
            },
            "mode": "eval",
            "device_policy": "cpu",
            "required_construct_asset": None,
            "recipe_revision": HASH,
            "torchlens_import_static_check": "passed",
        },
        "input_contract": {
            "builder_symbol": "make_dummy_call",
            "seed": 0,
            "semantic_description": "One small RGB image.",
            "source_basis": ["evidence-1"],
            "smallest_valid_probe_rationale": "Smallest valid spatial extent.",
            "args": [
                {
                    "path": "args[0]",
                    "kind": "tensor",
                    "semantic_role": "image",
                    "shape": [1, 3, 8, 8],
                    "dtype": "float32",
                    "device_policy": "cpu",
                    "distribution": "normal",
                    "constraints": [],
                    "source_evidence_ids": ["evidence-1"],
                }
            ],
            "kwargs": [],
            "non_tensor_values": [],
            "masks_state_and_control": [],
            "expected_output_semantics": "class scores",
        },
        "observed": {
            "parameter_count_total": 2,
            "parameter_count_trainable": 2,
            "native_framework": "pytorch",
            "delegated_method": "forward",
            "output_signature": {
                "tree": {"leaf": 0},
                "leaves": [
                    {
                        "path": "output",
                        "kind": "tensor",
                        "shape": [1, 2],
                        "dtype": "float32",
                        "device": "cpu",
                        "python_type": "torch.Tensor",
                    }
                ],
            },
            "input_kind": "standard-image",
            "input_asset": (
                f"standard:image.ppm:{hash_bytes((ASSET_ROOT / 'image.ppm').read_bytes())}"
            ),
            "input_note": "canonical test image",
            "constructor_seconds": 0.1,
            "forward_seconds": 0.1,
            "peak_rss_bytes": 128,
            "measurement_attempt_ids": [attempt_id],
            "snippet": "driver-owned isolated forward",
            "snippet_sha256": stable_hash("driver-owned isolated forward"),
        },
        "modes": {
            "meaningful_modes": ["eval"],
            "per_mode_run": {"eval": {"attempt_id": attempt_id, "status": "succeeded"}},
            "train_eval_divergence": "none",
            "divergence_evidence": "single meaningful mode",
        },
        "fidelity": {
            "required": False,
            "reason": "R1 official library",
            "verdict": None,
            "fidelity_identity": None,
            "gate_id": None,
            "current": kind == "runs",
            "permanent_scar": False,
            "deviations": [],
        },
        "accuracy_gate": {
            "required": True,
            "vet_identity": HASH if accepted else None,
            "gate_id": "gate-1" if accepted else None,
            "verdict": "accurate" if accepted else None,
            "current": accepted,
            "checker_model": "codex",
            "checker_version": "test",
            "prompt_sha256": HASH,
        },
        "execution": {
            "execution_identity": HASH,
            "environment_id": "env-test",
            "env_generation": HASH,
            "accepted_attempt_ids": [attempt_id],
            "confirmation_policy": "single-mechanical",
            "network_attempted": False,
            "checkpoint_accessed": False,
            "last_verified_at": NOW,
            "current": kind == "runs",
        },
        "status": {
            "kind": kind,
            "code": status_code,
            "stage": stage,
            "reason_code": reason,
            # Failed diagnostics must be absent or sidecar-redacted at the reducer boundary.
            "detail": None,
            "traceback": None,
            "no_traceback_reason": "no Python exception" if kind == "failed" else None,
            "attempted_rungs": ["R1_LIBRARY"],
            "retries": {
                "source": 0,
                "fetch": 0,
                "evidence": 0,
                "author": 0,
                "gate": 0,
                "environment": 0,
                "import": 0,
                "constructor": 0,
                "input": 0,
                "forward": 0,
                "fidelity": 0,
            },
            "environment": "env-test",
            "timestamp": NOW,
            "attempt_ids": [attempt_id],
            "root_cause_fingerprint": HASH if kind == "failed" else None,
            "supersedes_revision": None,
            "human_review": {
                "required": False,
                "reason": None,
                "queue": None,
                "requested_at": None,
            },
        },
        "provenance": {
            "author_model": "claude",
            "author_version": "test",
            "author_prompt_sha256": HASH,
            "checker_model": "codex",
            "checker_version": "test",
            "producer_run_id": "run-test",
            "machine_id": "machine-test",
        },
        "budget": {
            "author_sessions_used": 0,
            "author_sessions_max": 3,
            "gate_rounds_used": 0,
            "run_revisions_used": 1,
            "explicit_grants": [],
        },
        "flags": [],
        "notes": "",
        "scar_history": [],
        "completeness": {
            "schema_valid": True,
            "mandatory_source_present": True,
            "source_read_fields_complete": accepted,
            "evidence_coverage_complete": accepted,
            "accuracy_gate_current": accepted,
            "required_fidelity_current": True,
            "execution_current": kind == "runs",
            "family_template_valid": True,
            "release_eligible": accepted and kind == "runs",
            "issues": (
                []
                if accepted and kind == "runs"
                else ["authored-metadata-pending"]
                if kind == "runs"
                else [status_code]
            ),
        },
        "dependency_vector": {
            "intake_snapshot_id": "snapshot-1",
            "intake_snapshot_sha256": HASH,
            "intake_item_sha256": HASH,
            "author_result_schema_identity": "pending-untrusted",
            "author_dispatcher_identity": "pending-untrusted",
            "author_prompt_identity": "pending-untrusted",
            "checker_prompt_identity": "pending-untrusted",
            "terminal_rule_identity": "pending-untrusted",
            "status_proof_identity": "pending-untrusted",
            "source_manifest_identity": "pending-untrusted",
            "proposal_identity": "pending-untrusted",
            "author_result_identity": "pending-untrusted",
            "checker_gate_identity": "pending-untrusted",
            "recipe_revision": HASH,
            "runner_identity": "pending-untrusted",
            "award_closure_identity": "pending-untrusted",
            "environment_generation": HASH,
            "artifact_transaction_id": "not-applicable",
            "representative_revision": "not-applicable",
            "publication_policy_identity": "pending-untrusted",
            "accepted_attempt_ids": [attempt_id],
            "artifact_claim_ids": [],
        },
        "artifact_authority": {
            "state": "not-applicable",
            "transaction_id": "not-applicable",
            "committed_event_id": "not-applicable",
            "authorization_id": "not-applicable",
            "reconstruction_sha256": "not-applicable",
            "claim_ids": [],
        },
        "family_authority": {
            "binding_state": "ordinary",
            "representative_stable_id": stable_id,
            "representative_revision": "not-applicable",
            "representative_gate_id": "not-applicable",
            "representative_proposal_id": "not-applicable",
            "variant_token": "not-applicable",
            "template_source_revision": "not-applicable",
            "derivation_rule_identity": "not-applicable",
        },
    }
    if accepted:
        model["evidence"]["excerpts"][0]["supports"] = list(
            authored_fact_leaves(_model_facts(model), schema_version=MODEL_SCHEMA_VERSION)
        )
        _bind_model_identities(model)
    return model


def make_author_proposal(stable_id: str = "m_example") -> dict[str, Any]:
    """Build a complete staged author proposal.

    Parameters
    ----------
    stable_id:
        Proposed model ID.

    Returns
    -------
    dict[str, Any]
        Complete author-proposal.v3 payload.
    """

    model = make_model(stable_id, accepted=True)
    fact_keys = (
        "identity",
        "taxonomy",
        "external_metadata",
        "website",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
        "implementation",
        "input_contract",
        "modes",
        "fidelity",
    )
    proposal = {
        "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": "proposal-1",
        "proposal_sha256": HASH,
        "work_id": f"work-{stable_id}",
        "campaign_id": f"campaign-{stable_id}",
        "stable_id": stable_id,
        "intake_snapshot_id": "snapshot-test",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": stable_hash({"stable_id": stable_id}),
        "source_manifest_identity": HASH,
        "dispatcher_identity": HASH,
        "created_at": NOW,
        "author": {
            "provider": "anthropic",
            "model": "claude",
            "version": "test",
            "prompt_sha256": _author_prompt_hash(),
        },
        "source_identity": HASH,
        "evidence_identity": HASH,
        "recipe_revision": HASH,
        "fidelity_identity": None,
        "vet_identity": HASH,
        "verified_hashes": {
            "source_manifest": HASH,
            "evidence": HASH,
            "code": None,
            "source_to_code_map": HASH,
            "family_template": None,
        },
        "proposed_facts": {key: deepcopy(model[key]) for key in fact_keys},
    }
    identities = recompute_accepted_identities(
        proposal["proposed_facts"],
        checker_prompt_hash=_checker_prompt_hash(),
        checker_model="codex",
        checker_version="current",
        schema_version=MODEL_SCHEMA_VERSION,
    )
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "fidelity_identity": identities.fidelity,
            "vet_identity": identities.vet,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    return proposal


def make_proposed_artifact(
    proposal: dict[str, Any], source_manifest: dict[str, Any], model_dir: Path
) -> Any:
    """Wrap an injected proposal in the mandatory typed author-result arm."""

    from menagerie.crawler.driver import AuthorArtifact  # noqa: PLC0415

    raw_result = {
        "result_id": f"result-{proposal['stable_id']}",
        "result_sha256": HASH,
        "stable_id": proposal["stable_id"],
        "work_id": proposal["work_id"],
        "campaign_id": proposal.get("campaign_id", proposal["work_id"]),
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": source_manifest.get("manifest_sha256", HASH),
        "intake_snapshot_id": proposal.get("intake_snapshot_id", "snapshot-test"),
        "intake_snapshot_sha256": proposal.get("intake_snapshot_sha256", HASH),
        "intake_item_sha256": proposal.get("intake_item_sha256", HASH),
        "created_at": proposal.get("created_at", NOW),
    }
    binding = AuthorResultBinding(raw_result=raw_result, **raw_result)
    report = ProposalValidationReport(
        stable_id=str(proposal["stable_id"]),
        rung=SourceRung(str(proposal["proposed_facts"]["source_resolution"]["rung"])),
        code_path=None,
        supported_claims=frozenset(),
    )
    result = ProposedAuthorResult(binding=binding, proposal=proposal, validation_report=report)
    return AuthorArtifact(result, source_manifest, model_dir)


def make_operational_event() -> dict[str, Any]:
    """Build a complete usage-pause operational event.

    Returns
    -------
    dict[str, Any]
        Complete operational-event.v1 payload.
    """

    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": "event-1",
        "ledger_seq": 1,
        "payload_sha256": HASH,
        "created_at": NOW,
        "event_kind": "usage-pause",
        "status": "paused:usage-limit",
        "provider": "anthropic",
        "observed_response": "limit reached",
        "reset_at": "2026-07-14T13:00:00Z",
        "queued_work_counts": {"author": 2},
        "current_environment": "env-test",
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": {"wakeup": "scheduled"},
    }


def make_shutdown_interruption_event() -> dict[str, Any]:
    """Build the frozen operational-only shutdown interruption fixture.

    Returns
    -------
    dict[str, Any]
        Complete pre-spawn worker-shutdown-interrupted event.
    """

    event = make_operational_event()
    event.update(
        {
            "event_kind": "worker-shutdown-interrupted",
            "status": "interrupted:shutdown",
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "current_environment": None,
            "details": {
                "invocation_id": "invocation-1",
                "admission_boundary": "pre-spawn",
                "stable_id": "m_example",
                "work_id": "work-m_example",
                "execution_identity": HASH,
                "request_identity": None,
                "lease_id": None,
                "child_pid": None,
                "child_start_token": None,
                "child_pgid": None,
                "signal": None,
                "parent_observation": None,
                "partial_receipt": None,
            },
        }
    )
    return event


@pytest.fixture
def valid_model() -> dict[str, Any]:
    """Return a valid accepted model payload.

    Returns
    -------
    dict[str, Any]
        Accepted model payload.
    """

    return make_model(accepted=True)
