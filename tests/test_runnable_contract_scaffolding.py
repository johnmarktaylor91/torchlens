"""Import and construction tripwires for runnable contract scaffolding."""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

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
    ReadinessReport,
    RunnableCallDescriptor,
    RunnableErrorCode,
    RunnableTraceProtocol,
    RunReport,
    SparseRunDescriptor,
    StateSlotRole,
    WitnessCompleteness,
)


@pytest.mark.smoke
def test_frozen_runnable_schema_values() -> None:
    """Keep the v2 schema versions and key enum values frozen."""

    assert RUNNABLE_TLSPEC_SCHEMA_VERSION == "sparse_recorded_taken_path_v2"
    assert RUNNABLE_CALL_RECIPE_VERSION == "non_tensor_args_tensor_slots_context_and_obligations_v3"
    assert RUNNABLE_INITIALIZER_POLICY_VERSION == "torchlens_role_init_v2"
    assert RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION == "selected_activation_v2"
    assert frozenset({"sparse_recorded_taken_path_v1"}) == LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS
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
        "state_carried",
        "truncation",
        "truncated",
        "stopped_at",
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


# The S1 seam contract (docs/reference/runnable_model.md "Extension points"):
# every extension point resolves on its contract module, the protocol stays
# the stable typed minimum the concrete door satisfies, and the run-door
# conflict matrix is bound by CODE LIST, never by count.

_S1_EXTENSION_POINTS: dict[str, tuple[str, ...]] = {
    "torchlens.runnable": (
        "WITNESS_FAMILY_REGISTRY",
        "WITNESS_GAP_REGISTRY",
        "CANONICAL_INITIALIZER_BY_ROLE",
        "SparseRunDescriptor",
        "RunnableErrorCode",
        "RunProvider",
        "RunnableTraceProtocol",
        "RunResult",
        "mark_trace_path_status",
        "refuse_poisoned_trace",
    ),
    "torchlens._runnable_seam": (
        "RUNNABLE_TRACE_PUBLIC_MEMBERS",
        "RunnableCoordinator",
        "RunnableTraceState",
        "runnable_trace_state",
    ),
    "torchlens._runnable_execution": (
        "run_loaded_sparse_trace",
        "run_live_trace",
        "_finalize_provider_run",
    ),
    "torchlens._runnable_state": ("load_trace_state_dict",),
    # Coordinator verbs reached through the RunnableCoordinator boundary keep
    # their transport-side homes (contract E2).
    "torchlens._io.runnable": ("build_sparse_run_descriptor",),
    "torchlens._io.runnable_load": (
        "parse_sparse_run_descriptor",
        "attach_sparse_run_readiness",
    ),
    "torchlens._fast_run": ("run_fast_loaded_trace", "run_fast_live_trace"),
}

# Trace.run keyword-conflict matrix, bound as the CODE LIST (S1 contract:
# totality binds to codes, not counts). Every new run keyword lands with its
# conflict-matrix row here in the same change.
RUN_DOOR_CONFLICT_CODES: frozenset[str] = frozenset(
    {
        "run_legacy_arguments_conflict",
        "run_legacy_options_conflict",
        "run_fast_divergence_policy_invalid",
        "run_fast_requires_inputs",
        "run_source_model_collected",
        "run_capability_unavailable",
    }
)


def test_s1_extension_points_resolve_on_contract_modules() -> None:
    """Every S1 extension point resolves on its declared contract module."""

    import importlib

    for module_name, names in _S1_EXTENSION_POINTS.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name} lost S1 extension point {name}"


def test_trace_run_satisfies_protocol_minimum() -> None:
    """The concrete run door structurally satisfies the frozen protocol minimum.

    ``RunnableTraceProtocol.run`` promises the stable typed minimum (``inputs``,
    ``seed``, ``fast``, ``on_divergence``). The concrete ``Trace.run`` may add
    keywords only as defaulted (keyword-only preferred) parameters, so every
    existing typed consumer keeps compiling while unstable keywords incubate.
    """

    from torchlens.data_classes.trace import Trace

    protocol_params = inspect.signature(RunnableTraceProtocol.run).parameters
    concrete = inspect.signature(Trace.run)
    for name in ("inputs", "seed", "fast", "on_divergence"):
        assert name in concrete.parameters, f"Trace.run lost protocol parameter {name}"
        proto_param = protocol_params[name]
        conc_param = concrete.parameters[name]
        if proto_param.default is not inspect.Parameter.empty:
            assert conc_param.default == proto_param.default, (
                f"Trace.run default for {name} drifted from the protocol minimum"
            )
    # Structural satisfiability: every concrete parameter beyond the protocol
    # minimum must be optional (defaulted or VAR kinds), so a protocol-typed
    # call never breaks.
    for name, param in concrete.parameters.items():
        if name in {"self", "inputs", "seed", "fast", "on_divergence"}:
            continue
        assert param.default is not inspect.Parameter.empty or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ), f"Trace.run parameter {name} must carry a default (protocol satisfaction)"


def test_run_door_conflict_matrix_codes_present_in_door_source() -> None:
    """The conflict-matrix code list stays raised on the run-door path.

    Literal codes must appear in the door source itself; enum-spelled codes
    (``run_capability_unavailable``) must stay declared as ``RunnableErrorCode``
    members whose member name is referenced by runnable machinery the door
    delegates to.
    """

    from pathlib import Path

    package_root = Path(__file__).resolve().parent.parent / "torchlens"
    door_source = (package_root / "data_classes" / "_trace_validation.py").read_text(
        encoding="utf-8"
    )
    enum_values = {member.value for member in RunnableErrorCode}
    for code in sorted(RUN_DOOR_CONFLICT_CODES):
        if f'"{code}"' in door_source:
            continue
        assert code in enum_values, (
            f"run-door conflict code {code} is neither a door literal nor an enum member"
        )
        member_name = RunnableErrorCode(code).name
        referenced = any(
            member_name in path.read_text(encoding="utf-8")
            for path in package_root.glob("_runnable_*.py")
        )
        assert referenced, (
            f"run-door conflict code {code} has no enum-member raise site in runnable machinery"
        )
