"""Frozen type contracts for sparse runnable ``.tlspec`` artifacts.

This module contains schema and API shapes only. Runtime state binding lives in
``torchlens._runnable_state``; this module intentionally performs no capture,
serialization, callable resolution, state binding, or execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, TypeAlias

from .intervention.types import FunctionRegistryKey

RUNNABLE_TLSPEC_SCHEMA_VERSION: Final = "sparse_recorded_taken_path_v1"
"""Frozen sparse runnable descriptor capability/version string."""

RUNNABLE_CALL_RECIPE_VERSION: Final = "non_tensor_args_and_tensor_slots_v1"
"""Frozen sparse call-recipe version string."""

RUNNABLE_CALLABLE_REF_SCHEMA_VERSION: Final = 1
"""Frozen ``FunctionRegistryKey`` schema version accepted by rung 1."""

RUNNABLE_INITIALIZER_POLICY_VERSION: Final = "torchlens_role_init_v1"
"""Frozen canonical role-based random-initializer policy version."""


class TensorSlotRole(str, Enum):
    """Semantic roles available to tensor slots in the sparse core."""

    MODEL_INPUT = "model_input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    INTERMEDIATE = "intermediate"
    CONSTANT_LIKE_TENSOR = "constant_like_tensor"
    RNG_SOURCE = "rng_source"
    OUTPUT = "output"


class StateSlotRole(str, Enum):
    """Canonical roles used to bind and initialize parameter and buffer slots."""

    WEIGHT = "weight"
    BIAS = "bias"
    NORM_SCALE = "norm_scale"
    NORM_OFFSET = "norm_offset"
    RUNNING_MEAN = "running_mean"
    RUNNING_VAR = "running_var"
    COUNTER = "counter"
    GENERIC_BUFFER = "generic_buffer"


class InitializerPolicy(str, Enum):
    """Fill policies in the frozen N1-a initializer table."""

    KAIMING_NORMAL = "kaiming_normal"
    ZEROS = "zeros"
    ONES = "ones"


CANONICAL_INITIALIZER_BY_ROLE: Mapping[StateSlotRole, InitializerPolicy] = MappingProxyType(
    {
        StateSlotRole.WEIGHT: InitializerPolicy.KAIMING_NORMAL,
        StateSlotRole.BIAS: InitializerPolicy.ZEROS,
        StateSlotRole.NORM_SCALE: InitializerPolicy.ONES,
        StateSlotRole.NORM_OFFSET: InitializerPolicy.ZEROS,
        StateSlotRole.RUNNING_MEAN: InitializerPolicy.ZEROS,
        StateSlotRole.RUNNING_VAR: InitializerPolicy.ONES,
        StateSlotRole.COUNTER: InitializerPolicy.ZEROS,
        StateSlotRole.GENERIC_BUFFER: InitializerPolicy.ZEROS,
    }
)
"""Frozen N1-a role-to-policy mapping for ``torchlens_role_init_v1``."""


class WitnessCompleteness(str, Enum):
    """Coverage classifications for a runnable control-witness set."""

    COMPLETE = "complete"
    INCOMPLETE_SCALAR_ESCAPE = "incomplete_scalar_escape"
    INCOMPLETE_OPAQUE_SIDE_EFFECT = "incomplete_opaque_side_effect"
    INCOMPLETE_UNOBSERVED_PREDICATE = "incomplete_unobserved_predicate"


class PathFaithfulness(str, Enum):
    """Honesty states reported by a completed sparse run."""

    VERIFIED = "verified"
    DIVERGED = "diverged"
    UNVERIFIABLE = "unverifiable"


class DivergencePolicy(str, Enum):
    """Allowed responses to a detected path divergence."""

    RAISE = "raise"
    RETURN_DIVERGED = "return_diverged"


class ReadinessStatus(str, Enum):
    """Overall non-executing runnable readiness states."""

    READY = "ready"
    UNAVAILABLE = "unavailable"


class RunProvider(str, Enum):
    """Providers selected by the unified ``Trace.run`` contract."""

    LIVE = "live"
    LOADED_SPARSE = "loaded_sparse"
    LOADED_ANALYSIS = "loaded_analysis"


class StateSource(str, Enum):
    """Possible sources of state used by a run."""

    LIVE_MODEL_STATE = "live_model_state"
    EMBEDDED_CAPTURE_STATE = "embedded_capture_state"
    USER_STATE_DICT = "user_state_dict"
    RANDOM_INITIALIZATION = "random_initialization"
    NOT_APPLICABLE = "not_applicable"


class NumericAttestationStatus(str, Enum):
    """Result states for the optional saved-activation oracle."""

    ATTESTED = "attested"
    NUMERIC_ATTESTATION_FAILED = "numeric_attestation_failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_PRESENT = "not_present"


class ResolverStatus(str, Enum):
    """Per-reference callable reattachment outcomes."""

    RESOLVED_EXACT = "resolved_exact"
    RESOLVED_ALIAS = "resolved_alias"
    UNAVAILABLE = "unavailable"


class RunnableErrorCode(str, Enum):
    """Frozen machine-readable error and divergence taxonomy."""

    RUN_CAPABILITY_UNAVAILABLE = "run_capability_unavailable"
    SPARSE_PREFLIGHT_FAILED = "sparse_preflight_failed"
    SPARSE_CORE_TENSOR_PAYLOAD = "sparse_core_tensor_payload"
    UNSUPPORTED_LITERAL = "unsupported_literal"
    UNSUPPORTED_TENSOR_CONSTANT = "unsupported_tensor_constant"
    MISSING_TENSOR_SLOT = "missing_tensor_slot"
    MISSING_INPUT_CONTAINER_CONTRACT = "missing_input_container_contract"
    MISSING_OUTPUT_CONTAINER_CONTRACT = "missing_output_container_contract"
    MISSING_CONTROL_CLASSIFICATION = "missing_control_classification"
    MISSING_CALLABLE_REF = "missing_callable_ref"
    UNSUPPORTED_BACKEND_REPLAY = "unsupported_backend_replay"
    UNSUPPORTED_REF_SCHEMA = "unsupported_ref_schema"
    UNRESOLVED_QUALNAME = "unresolved_qualname"
    AMBIGUOUS_QUALNAME = "ambiguous_qualname"
    CALLABLE_MOVED_OR_RENAMED = "callable_moved_or_renamed"
    CALLABLE_REMOVED = "callable_removed"
    PRIVATE_API_UNAVAILABLE = "private_api_unavailable"
    SIGNATURE_DRIFT = "signature_drift"
    RUNTIME_SIGNATURE_DRIFT = "runtime_signature_drift"
    SEMANTIC_DRIFT = "semantic_drift"
    WRAPPER_SHADOWED = "wrapper_shadowed"
    UNTRUSTED_CUSTOM_IMPORT = "untrusted_custom_import"
    STATE_MISSING_KEY = "state_missing_key"
    STATE_UNEXPECTED_KEY = "state_unexpected_key"
    STATE_SHAPE_MISMATCH = "state_shape_mismatch"
    STATE_DTYPE_MISMATCH = "state_dtype_mismatch"
    STATE_ROLE_MISMATCH = "state_role_mismatch"
    STATE_MODULE_PATH_MISMATCH = "state_module_path_mismatch"
    STATE_ALIAS_CONFLICT = "state_alias_conflict"
    INPUT_TREE_MISMATCH = "input_tree_mismatch"
    INPUT_SHAPE_MISMATCH = "input_shape_mismatch"
    INPUT_DTYPE_MISMATCH = "input_dtype_mismatch"
    CALL_ARITY_MISMATCH = "call_arity_mismatch"
    CALL_STRUCTURE_MISMATCH = "call_structure_mismatch"
    OUTPUT_STRUCTURE_MISMATCH = "output_structure_mismatch"
    OUTPUT_SHAPE_MISMATCH = "output_shape_mismatch"
    OUTPUT_DTYPE_MISMATCH = "output_dtype_mismatch"
    SLOT_PRODUCTION_MISMATCH = "slot_production_mismatch"
    MUTATION_VERSION_MISMATCH = "mutation_version_mismatch"
    SCALAR_BOOL_DIVERGENCE = "scalar_bool_divergence"
    CONDITIONAL_ARM_DIVERGENCE = "conditional_arm_divergence"
    LOOP_PREDICATE_DIVERGENCE = "loop_predicate_divergence"
    NUMERIC_ATTESTATION_FAILED = "numeric_attestation_failed"
    POISONED_RUN_REFUSED = "poisoned_run_refused"


class LiteralAtomKind(str, Enum):
    """Scalar kinds in the safe non-tensor literal grammar."""

    NONE = "none"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"


class LiteralSequenceKind(str, Enum):
    """Sequence kinds preserved by the safe non-tensor literal grammar."""

    LIST = "list"
    TUPLE = "tuple"


@dataclass(frozen=True, slots=True)
class LiteralAtom:
    """One explicitly tagged JSON scalar in a sparse call recipe."""

    kind: LiteralAtomKind
    value: None | bool | int | float | str


@dataclass(frozen=True, slots=True)
class LiteralTorchSymbol:
    """One allowlisted non-callable torch symbolic constant."""

    qualname: str


@dataclass(frozen=True, slots=True)
class LiteralTupleKey:
    """One recursively safe tuple used as a literal mapping key."""

    items: tuple[LiteralAtom | "LiteralTupleKey", ...]


LiteralDictKey: TypeAlias = LiteralAtom | LiteralTupleKey


@dataclass(frozen=True, slots=True)
class LiteralSequence:
    """One list or tuple node in a sparse non-tensor argument tree."""

    kind: LiteralSequenceKind
    items: tuple["NonTensorLiteral", ...]


@dataclass(frozen=True, slots=True)
class LiteralMappingEntry:
    """One ordered key/value entry in a sparse literal mapping."""

    key: LiteralDictKey
    value: "NonTensorLiteral"


@dataclass(frozen=True, slots=True)
class LiteralMapping:
    """One ordered mapping node in a sparse non-tensor argument tree."""

    entries: tuple[LiteralMappingEntry, ...]


NonTensorLiteral: TypeAlias = LiteralAtom | LiteralTorchSymbol | LiteralSequence | LiteralMapping
ArgumentPath: TypeAlias = tuple[str | int, ...]
ContainerPath: TypeAlias = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class TensorUseSite:
    """One argument or output path at which a tensor slot is used."""

    call_id: str
    argument_path: ArgumentPath


@dataclass(frozen=True, slots=True)
class InputSlotBinding:
    """Stable model-boundary binding for an input tensor slot."""

    io_role: Literal["model_input"]
    model_ref: str
    model_site_position: str | int | tuple[str | int, ...]
    container_record_id: int
    container_path: ContainerPath


@dataclass(frozen=True, slots=True)
class StateSlotBinding:
    """Name, ownership, role, and alias metadata for one state slot."""

    module_path: str
    state_dict_name: str
    semantic_role: StateSlotRole
    trainable: bool
    persistent: bool
    alias_group: str | None


@dataclass(frozen=True, slots=True)
class TensorSlotDescriptor:
    """Value-free allocation and binding contract for one tensor slot."""

    slot_id: str
    role: TensorSlotRole
    use_sites: tuple[TensorUseSite, ...]
    shape: tuple[int, ...]
    dtype: str
    rank: int
    device_type: str
    device_index: int | None
    mutable: bool
    version_of: str | None
    producer_slot_id: str | None
    output_path: ContainerPath | None
    input_binding: InputSlotBinding | None
    state_binding: StateSlotBinding | None


@dataclass(frozen=True, slots=True)
class TensorArgumentRef:
    """One call-tree leaf populated from a runtime tensor slot."""

    argument_path: ArgumentPath
    slot_id: str


@dataclass(frozen=True, slots=True)
class LiteralArgumentRef:
    """One call-tree leaf populated from the safe literal grammar."""

    argument_path: ArgumentPath
    value: NonTensorLiteral


@dataclass(frozen=True, slots=True)
class CallableRegistryEntry:
    """Deduplicated callable reference used by runnable call descriptors."""

    registry_id: str
    key: FunctionRegistryKey


@dataclass(frozen=True, slots=True)
class RunnableCallDescriptor:
    """Value-free recipe for one grouped computational call."""

    call_id: str
    op_labels: tuple[str, ...]
    registry_id: str
    dispatch_kind: Literal["function", "method", "dunder", "namespace_alias"]
    argument_names: tuple[str, ...]
    num_positional_args: int
    num_keyword_args: int
    tensor_arguments: tuple[TensorArgumentRef, ...]
    literal_arguments: tuple[LiteralArgumentRef, ...]
    output_slot_ids: tuple[str, ...]
    parent_call_ids: tuple[str, ...]
    is_inplace: bool
    runtime_fingerprint: str


class ControlWitnessKind(str, Enum):
    """Observable control facts persisted in a sparse descriptor."""

    SCALAR_BOOL = "scalar_bool"
    CONDITIONAL_ARM_ENTRY = "conditional_arm_entry"
    LOOP_PREDICATE = "loop_predicate"
    SHAPE_STRUCTURE_FACT = "shape_structure_fact"


@dataclass(frozen=True, slots=True)
class ControlWitness:
    """One ordered, non-tensor control-flow witness."""

    witness_id: str
    kind: ControlWitnessKind
    order: int
    call_id: str | None
    site_label: str
    observed_value: NonTensorLiteral


@dataclass(frozen=True, slots=True)
class PayloadLayerDescriptor:
    """Presence and schema declaration for one optional payload layer."""

    present: bool
    schema: str


@dataclass(frozen=True, slots=True)
class ActivationPayloadMember:
    """One archived capture-selected activation payload."""

    blob_id: str
    slot_id: str
    call_id: str | None
    op_label: str
    field: Literal["out", "transformed_out"]
    byte_digest: str


@dataclass(frozen=True, slots=True)
class SlotByteDigest:
    """Capture-time byte digest used to recognize one runtime tensor slot."""

    slot_id: str
    byte_digest: str


@dataclass(frozen=True, slots=True)
class StateByteDigest:
    """Capture-time byte digest used to recognize equivalent real state."""

    state_dict_name: str
    byte_digest: str


@dataclass(frozen=True, slots=True)
class ActivationPayloadLayerDescriptor:
    """Presence, membership, and eligibility metadata for activation payloads."""

    present: bool
    schema: Literal["selected_activation_v1"]
    members: tuple[ActivationPayloadMember, ...]
    original_input_digests: tuple[SlotByteDigest, ...]
    capture_state_digests: tuple[StateByteDigest, ...]


@dataclass(frozen=True, slots=True)
class ArchivedActivation:
    """One loaded activation payload exposed for offline inspection."""

    slot_id: str
    call_id: str | None
    op_label: str
    field: Literal["out", "transformed_out"]
    byte_digest: str
    value: Any


@dataclass(frozen=True, slots=True)
class PayloadLayersDescriptor:
    """Independent optional payload-layer declarations."""

    weights: PayloadLayerDescriptor
    nonpersistent_buffers: PayloadLayerDescriptor
    activations: PayloadLayerDescriptor | ActivationPayloadLayerDescriptor


@dataclass(frozen=True, slots=True)
class RunnableRngProfile:
    """Host (Python ``random`` / NumPy) RNG reproducibility metadata.

    The sparse recorded path is a single taken branch. When the traced forward
    consumed Python or NumPy RNG (e.g. ``if random.random() > 0.5:``), that
    branch is invisible to tensor-level control witnesses, so a run under a
    different seed than capture cannot be blessed as VERIFIED/ATTESTED -- the
    recorded path may not be the branch a fresh seeded call would take. Only a
    run that reproduces the captured seed/state is an honest VERIFIED replay.
    """

    host_rng_consumed: bool
    capture_seed: int | None


@dataclass(frozen=True, slots=True)
class RunnableCompatibility:
    """Producer/runtime versions recorded for readiness diagnostics."""

    torchlens_version: str
    python_version: str
    backend_version: str
    descriptor_version: str
    call_recipe_version: str
    callable_ref_schema_version: int
    initializer_policy_version: str


@dataclass(frozen=True, slots=True)
class RunnableDiagnostic:
    """Structured diagnostic shared by preflight, readiness, and run reports."""

    code: RunnableErrorCode
    message: str
    registry_id: str | None
    affected_op_labels: tuple[str, ...]
    recorded_runtime: str | None
    current_runtime: str | None
    detection_stage: str
    resolver_provenance: str | None
    analysis_load_available: bool
    details: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ProducerPreflight:
    """Whole-graph runnable producer preflight result."""

    passed: bool
    diagnostics: tuple[RunnableDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class SparseRunDescriptor:
    """Authoritative manifest descriptor for sparse runnable rung 1."""

    capability: Literal["sparse_recorded_taken_path_v1"]
    backend: str
    call_recipe: Literal["non_tensor_args_and_tensor_slots_v1"]
    callable_ref_schema: Literal[1]
    state_binding: Literal["module_path_role_v1"]
    input_binding: Literal["model_site_io_role_v1"]
    control_witness: Literal["scalar_bool_and_arm_entry_v1"]
    initializer_policy_version: Literal["torchlens_role_init_v1"]
    payload_layers: PayloadLayersDescriptor
    callable_registry: tuple[CallableRegistryEntry, ...]
    calls: tuple[RunnableCallDescriptor, ...]
    tensor_slots: tuple[TensorSlotDescriptor, ...]
    control_witnesses: tuple[ControlWitness, ...]
    witness_completeness: WitnessCompleteness
    rng_profile: RunnableRngProfile
    compatibility: RunnableCompatibility
    preflight: ProducerPreflight
    unsupported_sites: tuple[RunnableDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ResolverRecord:
    """Resolution provenance for one unique callable registry entry."""

    registry_id: str
    status: ResolverStatus
    recorded_key: FunctionRegistryKey
    resolved_qualname: str | None
    provenance: str
    diagnostics: tuple[RunnableDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Complete non-executing readiness report for a Trace provider."""

    status: ReadinessStatus
    provider: RunProvider
    backend: str
    capability: str | None
    resolver_records: tuple[ResolverRecord, ...]
    state_sources_available: tuple[StateSource, ...]
    witness_completeness: WitnessCompleteness | None
    diagnostics: tuple[RunnableDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ContractCheck:
    """One contract check recorded in a run report."""

    name: str
    passed: bool
    diagnostic: RunnableDiagnostic | None


@dataclass(frozen=True, slots=True)
class RunReport:
    """Honesty, state, resolution, and contract metadata for one run."""

    readiness: ReadinessReport
    state_source: StateSource
    initializer_policy_version: str | None
    seed: int | None
    random_filled_slot_ids: tuple[str, ...]
    contract_checks: tuple[ContractCheck, ...]
    path_faithfulness: PathFaithfulness
    first_mismatch: RunnableDiagnostic | None
    numeric_attestation: NumericAttestationStatus
    poisoned: bool


@dataclass(frozen=True, slots=True)
class RunResult:
    """Uniform result returned by both live and loaded ``Trace.run`` providers."""

    output: Any
    trace: Any
    report: RunReport


class RunnableTraceProtocol(Protocol):
    """Frozen type contract for ``Trace`` runnable methods across implementation stages."""

    def run(
        self,
        inputs: Any,
        *,
        seed: int | None = None,
        on_divergence: DivergencePolicy = DivergencePolicy.RAISE,
    ) -> RunResult:
        """Execute through the provider selected by the Trace.

        Parameters
        ----------
        inputs:
            Structured runtime inputs.
        seed:
            Optional isolated state and runtime RNG seed.
        on_divergence:
            Frozen strict or poison-return divergence policy.

        Returns
        -------
        RunResult
            Uniform transactional run result.
        """
        ...

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        """Atomically validate and stage a strict state mapping.

        Parameters
        ----------
        sd:
            Canonically named parameter and buffer mapping.
        """
        ...


def mark_trace_path_status(
    trace: Any,
    status: PathFaithfulness,
    first_mismatch: RunnableDiagnostic | None,
) -> tuple[PathFaithfulness, RunnableDiagnostic | None]:
    """Apply a monotonic non-faithful path mark to a Trace.

    Parameters
    ----------
    trace:
        Trace receiving the run-status mark.
    status:
        Faithfulness status observed by the current sparse run.
    first_mismatch:
        First structured contradiction, when one was observed.

    Returns
    -------
    tuple[PathFaithfulness, RunnableDiagnostic | None]
        Effective monotonic status and retained first mismatch.
    """

    previous = trace.__dict__.get("_runnable_path_faithfulness")
    previous_mismatch = trace.__dict__.get("_runnable_first_mismatch")
    if previous is PathFaithfulness.DIVERGED:
        effective = PathFaithfulness.DIVERGED
    elif status is PathFaithfulness.DIVERGED:
        effective = PathFaithfulness.DIVERGED
    elif previous is PathFaithfulness.UNVERIFIABLE or status is PathFaithfulness.UNVERIFIABLE:
        effective = PathFaithfulness.UNVERIFIABLE
    else:
        effective = PathFaithfulness.VERIFIED
    retained = (
        previous_mismatch if isinstance(previous_mismatch, RunnableDiagnostic) else first_mismatch
    )
    trace.__dict__["_runnable_path_faithfulness"] = effective
    trace.__dict__["_runnable_first_mismatch"] = retained
    trace.__dict__["_runnable_poisoned"] = effective is not PathFaithfulness.VERIFIED
    return effective, retained


def refuse_poisoned_trace(trace: Any, operation: str) -> None:
    """Refuse a downstream operation that assumes model-faithful path identity.

    Parameters
    ----------
    trace:
        Trace being consumed.
    operation:
        Human-readable faithful consumer name.

    Raises
    ------
    PoisonedRunError
        If the Trace carries a monotonic non-faithful sparse-run mark.
    """

    if not bool(trace.__dict__.get("_runnable_poisoned", False)):
        return
    from .errors import PoisonedRunError

    status = trace.__dict__.get("_runnable_path_faithfulness")
    mismatch = trace.__dict__.get("_runnable_first_mismatch")
    raise PoisonedRunError(
        f"{operation} refused a poison-marked sparse run Trace.",
        code=RunnableErrorCode.POISONED_RUN_REFUSED.value,
        operation=operation,
        path_faithfulness=status,
        first_mismatch=mismatch,
    )


__all__ = [
    "ActivationPayloadLayerDescriptor",
    "ActivationPayloadMember",
    "ArchivedActivation",
    "CANONICAL_INITIALIZER_BY_ROLE",
    "RUNNABLE_CALLABLE_REF_SCHEMA_VERSION",
    "RUNNABLE_CALL_RECIPE_VERSION",
    "RUNNABLE_INITIALIZER_POLICY_VERSION",
    "RUNNABLE_TLSPEC_SCHEMA_VERSION",
    "CallableRegistryEntry",
    "ContractCheck",
    "ControlWitness",
    "ControlWitnessKind",
    "DivergencePolicy",
    "InitializerPolicy",
    "InputSlotBinding",
    "LiteralArgumentRef",
    "LiteralAtom",
    "LiteralAtomKind",
    "LiteralMapping",
    "LiteralMappingEntry",
    "LiteralSequence",
    "LiteralSequenceKind",
    "LiteralTorchSymbol",
    "LiteralTupleKey",
    "NumericAttestationStatus",
    "PathFaithfulness",
    "PayloadLayerDescriptor",
    "PayloadLayersDescriptor",
    "ProducerPreflight",
    "ReadinessReport",
    "ReadinessStatus",
    "ResolverRecord",
    "ResolverStatus",
    "RunProvider",
    "RunReport",
    "RunResult",
    "RunnableCallDescriptor",
    "RunnableCompatibility",
    "RunnableDiagnostic",
    "RunnableErrorCode",
    "RunnableTraceProtocol",
    "SparseRunDescriptor",
    "StateSlotBinding",
    "StateSlotRole",
    "StateByteDigest",
    "StateSource",
    "TensorArgumentRef",
    "TensorSlotDescriptor",
    "TensorSlotRole",
    "TensorUseSite",
    "SlotByteDigest",
    "WitnessCompleteness",
    "mark_trace_path_status",
    "refuse_poisoned_trace",
]
