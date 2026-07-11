"""Import and construction tripwires for runnable contract scaffolding."""

from __future__ import annotations

from dataclasses import fields

from torchlens.errors import ReattachError, RunnableTLSPECError, StateBindingError
from torchlens.runnable import (
    CANONICAL_INITIALIZER_BY_ROLE,
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    InitializerPolicy,
    RunReport,
    ReadinessReport,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSlotRole,
    WitnessCompleteness,
)


def test_frozen_runnable_schema_values() -> None:
    """Keep the Stage 0 schema version and key enum values frozen."""

    assert RUNNABLE_TLSPEC_SCHEMA_VERSION == "sparse_recorded_taken_path_v1"
    assert WitnessCompleteness.COMPLETE.value == "complete"
    assert RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY.value == "unsupported_backend_replay"
    assert RunnableErrorCode.NUMERIC_ATTESTATION_FAILED.value == "numeric_attestation_failed"
    assert CANONICAL_INITIALIZER_BY_ROLE[StateSlotRole.WEIGHT] is InitializerPolicy.KAIMING_NORMAL


def test_runnable_errors_import_and_instantiate() -> None:
    """Ensure public runnable errors retain the shared TorchLens payload style."""

    error = ReattachError(registry_id="callable:1")
    assert isinstance(error, RunnableTLSPECError)
    assert error.fields == {"registry_id": "callable:1"}
    assert isinstance(StateBindingError("bad state"), ValueError)


def test_authoritative_descriptor_and_report_field_names() -> None:
    """Keep persisted descriptor and report shapes synchronized with the frozen document."""

    assert tuple(field.name for field in fields(SparseRunDescriptor)) == (
        "capability",
        "backend",
        "call_recipe",
        "callable_ref_schema",
        "state_binding",
        "input_binding",
        "control_witness",
        "initializer_policy_version",
        "payload_layers",
        "callable_registry",
        "calls",
        "tensor_slots",
        "control_witnesses",
        "witness_completeness",
        "compatibility",
        "preflight",
        "unsupported_sites",
    )
    assert tuple(field.name for field in fields(RunReport)) == (
        "readiness",
        "state_source",
        "initializer_policy_version",
        "seed",
        "random_filled_slot_ids",
        "contract_checks",
        "path_faithfulness",
        "first_mismatch",
        "numeric_attestation",
        "poisoned",
    )
    assert tuple(field.name for field in fields(ReadinessReport)) == (
        "status",
        "provider",
        "backend",
        "capability",
        "resolver_records",
        "state_sources_available",
        "witness_completeness",
        "diagnostics",
    )
