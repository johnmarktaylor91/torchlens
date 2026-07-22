"""Import and construction tripwires for runnable contract scaffolding."""

from __future__ import annotations

from dataclasses import fields

from torchlens.errors import ReattachError, RunnableTLSPECError, StateBindingError
from torchlens.runnable import (
    CANONICAL_INITIALIZER_BY_ROLE,
    LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS,
    RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
    RUNNABLE_CALL_RECIPE_VERSION,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    ActivationPayloadLayerDescriptor,
    AmbientExecutionContext,
    CallExecutionContext,
    InitializerPolicy,
    InputAttestationFingerprint,
    RunReport,
    ReadinessReport,
    RunnableCallDescriptor,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSlotRole,
    WitnessCompleteness,
)


def test_frozen_runnable_schema_values() -> None:
    """Keep the v2 schema versions and key enum values frozen."""

    assert RUNNABLE_TLSPEC_SCHEMA_VERSION == "sparse_recorded_taken_path_v2"
    assert RUNNABLE_CALL_RECIPE_VERSION == "non_tensor_args_tensor_slots_context_and_obligations_v3"
    assert RUNNABLE_INITIALIZER_POLICY_VERSION == "torchlens_role_init_v2"
    assert RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION == "selected_activation_v2"
    assert LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS == frozenset({"sparse_recorded_taken_path_v1"})
    assert WitnessCompleteness.COMPLETE.value == "complete"
    assert RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY.value == "unsupported_backend_replay"
    assert RunnableErrorCode.NUMERIC_ATTESTATION_FAILED.value == "numeric_attestation_failed"
    assert (
        RunnableErrorCode.INPUT_ALIAS_TOPOLOGY_UNRESOLVED.value == "input_alias_topology_unresolved"
    )
    assert RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE.value == "execution_context_unavailable"
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
        # r71 A: the REQUIRED witness-free input-boundary record (site/arity +
        # metadata-envelope domain authority).
        "input_boundary",
        "control_witnesses",
        # r71 A: the explicit typed coverage-gap ledger the completeness floor
        # derives from.
        "coverage_gaps",
        # r69 A: the descriptor-native presence ledger, DEMOTED r71 to a redundant
        # discharge mirror (contract section 4).
        "required_witness_inventory",
        "witness_completeness",
        "rng_profile",
        "ambient_context",
        "compatibility",
        "preflight",
        "unsupported_sites",
    )
    assert tuple(field.name for field in fields(RunnableCallDescriptor)) == (
        "call_id",
        "op_labels",
        "registry_id",
        "dispatch_kind",
        "argument_names",
        "num_positional_args",
        "num_keyword_args",
        "tensor_arguments",
        "literal_arguments",
        "output_slot_ids",
        "parent_call_ids",
        "is_inplace",
        "runtime_fingerprint",
        "execution_context",
        # r71 A2 (call-recipe v3): REQUIRED owner-record obligations.
        "control_obligations",
        "control_dependencies",
    )
    assert tuple(field.name for field in fields(CallExecutionContext)) == (
        "autocast",
        "grad_enabled",
        "inference_mode",
    )
    assert tuple(field.name for field in fields(AmbientExecutionContext)) == (
        "default_dtype",
        "default_device",
        "float32_matmul_precision",
        "deterministic_algorithms",
        "deterministic_algorithms_warn_only",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_enabled",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "grad_enabled",
        "inference_mode",
        "fill_uninitialized_memory",
        "attestation_ineligible_context",
    )
    assert tuple(field.name for field in fields(ActivationPayloadLayerDescriptor)) == (
        "present",
        "schema",
        "members",
        "original_input_digests",
        "capture_state_digests",
        "input_fingerprints",
    )
    assert tuple(field.name for field in fields(InputAttestationFingerprint)) == (
        "slot_id",
        "byte_digest",
        "device_type",
        "device_index",
        "layout",
        "sizes",
        "strides",
        "storage_offset",
        "is_contiguous",
        "is_channels_last",
        "is_channels_last_3d",
        "is_conj",
        "is_neg",
        "tensor_class",
        "requires_grad",
        "is_inference",
        "alignment_class",
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
        "nondeterministic_sources",
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
