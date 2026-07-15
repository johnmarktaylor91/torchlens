"""Full-contract synthetic fixtures for crawler Slice A tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.metadata import (
    authored_fact_leaves,
    recompute_accepted_identities,
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


def _checker_prompt_hash() -> str:
    """Return the exact frozen checker prompt byte hash used by fixtures."""

    path = Path(__file__).parents[1] / "prompts" / "codex_accuracy_checker_v2.txt"
    return hash_bytes(path.read_bytes())


def _author_prompt_hash() -> str:
    """Return the exact frozen author prompt byte hash used by fixtures."""

    path = Path(__file__).parents[1] / "prompts" / "claude_crawler_author_v2.txt"
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
            "observed_code_manifest_sha256": None,
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
    )
    model["identities"].update(
        {
            "source": identities.source,
            "evidence": identities.evidence,
            "recipe": identities.recipe,
            "checker_prompt": _checker_prompt_hash(),
        }
    )
    model["worker_receipt"]["observed_recipe_revision"] = identities.recipe
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
                    for field in authored_fact_leaves(_model_facts(model))
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
    }
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
        Complete model.v2 payload.
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
                    "text_sha256": HASH,
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
            "code_path": None,
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
            "input_asset": "asset:test",
            "input_note": "canonical test image",
            "constructor_seconds": 0.1,
            "forward_seconds": 0.1,
            "peak_rss_bytes": 128,
            "measurement_attempt_ids": [attempt_id],
            "snippet": "ExampleNet()",
            "snippet_sha256": HASH,
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
            "current": True,
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
            "current": True,
        },
        "status": {
            "kind": kind,
            "code": status_code,
            "stage": stage,
            "reason_code": reason,
            "detail": "synthetic failure" if kind == "failed" else None,
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
            "execution_current": True,
            "family_template_valid": accepted,
            "release_eligible": accepted,
            "issues": [] if accepted else ["authored-metadata-pending"],
        },
    }
    if accepted:
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
        Complete author-proposal.v2 payload.
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
        "stable_id": stable_id,
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


@pytest.fixture
def valid_model() -> dict[str, Any]:
    """Return a valid accepted model payload.

    Returns
    -------
    dict[str, Any]
        Accepted model payload.
    """

    return make_model(accepted=True)
