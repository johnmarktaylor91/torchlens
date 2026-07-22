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

RUNNABLE_TLSPEC_SCHEMA_VERSION: Final = "sparse_recorded_taken_path_v2"
"""Frozen sparse runnable descriptor capability/version string.

v2 requires EXPLICIT execution-context records (per-call ``execution_context``
plus the capture-scoped ``ambient_context``); the parser REJECTS their absence,
so an absent context can only ever mean a legacy v1 artifact, which is
analysis-only (fail-closed), never "assume disabled".
"""

RUNNABLE_CALL_RECIPE_VERSION: Final = "non_tensor_args_tensor_slots_context_and_obligations_v3"
"""Frozen sparse call-recipe version string.

v3 (r71 A) adds REQUIRED per-call obligation fields to every call record:
``control_obligations`` (owner-record scalar-bool/loop-predicate obligations) and
``control_dependencies`` (owner-record conditional arm-entry edges on the CHILD
call). Absence of either field is a parse failure -- the explicit-schema-decision
doctrine: obligation fields are never optional or defaulted.
"""

RUNNABLE_CALLABLE_REF_SCHEMA_VERSION: Final = 1
"""Frozen ``FunctionRegistryKey`` schema version accepted by rung 1."""

RUNNABLE_INITIALIZER_POLICY_VERSION: Final = "torchlens_role_init_v2"
"""Frozen canonical role-based random-initializer policy version.

v2 semantics are the v1 role table plus degenerate totality: a legal
``numel() == 0`` slot allocates and returns without sampling or consuming any
generator state, and every nonempty Kaiming slot requires finite positive
``fan_in`` (producer preflight and the runtime validator share the check).
"""

RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION: Final = "selected_activation_v2"
"""Frozen selected-activation payload schema (v2: physical input fingerprints)."""

LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {"sparse_recorded_taken_path_v1"}
)
"""Known legacy capability strings: analysis-only load, typed run refusal."""

WITNESS_FAMILY_REGISTRY_VERSION: Final = "witness_family_registry_v2"
"""Closed registry/version discriminator persisted in every required-witness inventory.

A parser only accepts an inventory authored against ITS OWN closed registry; any other
discriminator refuses at parse (``context_field_invalid``), so a future registry change
is an explicit schema decision, never a silent vocabulary drift. v2 (r71 A) extends the
registry from the seven ``SHAPE_STRUCTURE_FACT`` prefixes to EVERY verdict-steering
witness family -- the four direct control kinds included -- plus the two explicit
claim-only families, each row a typed :class:`FamilySpec` naming its independent
replay-structural anchor. A v1 inventory refuses at parse (analysis-only load); there
is no legacy fallback and no silent compatibility default.

``WITNESS_FAMILY_REGISTRY`` itself is defined after :class:`ControlWitnessKind` below
(the typed rows reference the kind enum).
"""


def encode_input_site_position(position: Any) -> str:
    """Encode one normalized root model-input site position as a canonical member ID.

    ``("arg", 0) -> "arg:0"`` / ``("kwarg", name) -> "kwarg:<name>"``. Raises
    ``ValueError`` for anything outside the two-element root-position grammar.
    """

    if (
        isinstance(position, (list, tuple))
        and len(position) == 2
        and position[0] in {"arg", "kwarg"}
        and isinstance(position[1], (str, int))
        and not isinstance(position[1], bool)
    ):
        kind, key = position
        if kind == "arg" and not isinstance(key, int):
            raise ValueError(f"Positional site index must be an int, got {key!r}.")
        if kind == "kwarg" and not isinstance(key, str):
            raise ValueError(f"Keyword site name must be a str, got {key!r}.")
        return f"{kind}:{key}"
    raise ValueError(f"Model-input site position {position!r} is outside the root grammar.")


def decode_input_site_position(member: str) -> "tuple[str, str | int]":
    """Decode one canonical site member ID back to its ``(kind, key)`` position."""

    kind, separator, key = member.partition(":")
    if not separator or kind not in {"arg", "kwarg"}:
        raise ValueError(f"Malformed input-site member ID {member!r}.")
    if kind == "arg":
        if not key.isdigit():
            raise ValueError(f"Malformed positional-site member ID {member!r}.")
        return ("arg", int(key))
    return ("kwarg", key)


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
    STATE_METADATA_MISMATCH = "state_metadata_mismatch"
    INPUT_TREE_MISMATCH = "input_tree_mismatch"
    INPUT_SHAPE_MISMATCH = "input_shape_mismatch"
    INPUT_DTYPE_MISMATCH = "input_dtype_mismatch"
    CALL_ARITY_MISMATCH = "call_arity_mismatch"
    INPUT_ARITY_EXTRA = "input_arity_extra"
    CALL_STRUCTURE_MISMATCH = "call_structure_mismatch"
    OUTPUT_STRUCTURE_MISMATCH = "output_structure_mismatch"
    OUTPUT_SHAPE_MISMATCH = "output_shape_mismatch"
    OUTPUT_DTYPE_MISMATCH = "output_dtype_mismatch"
    SLOT_PRODUCTION_MISMATCH = "slot_production_mismatch"
    MUTATION_VERSION_MISMATCH = "mutation_version_mismatch"
    SCALAR_BOOL_DIVERGENCE = "scalar_bool_divergence"
    CONDITIONAL_ARM_DIVERGENCE = "conditional_arm_divergence"
    LOOP_PREDICATE_DIVERGENCE = "loop_predicate_divergence"
    INPUT_ALIAS_TOPOLOGY_UNRESOLVED = "input_alias_topology_unresolved"
    STATE_ALIAS_TOPOLOGY_UNSUPPORTED = "state_alias_topology_unsupported"
    EXECUTION_CONTEXT_UNAVAILABLE = "execution_context_unavailable"
    CONTEXT_FIELD_INVALID = "context_field_invalid"
    NUMERIC_ATTESTATION_FAILED = "numeric_attestation_failed"
    POISONED_RUN_REFUSED = "poisoned_run_refused"


class LiteralAtomKind(str, Enum):
    """Scalar kinds in the safe non-tensor literal grammar."""

    NONE = "none"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    NONFINITE_FLOAT = "nonfinite_float"
    STR = "str"
    ELLIPSIS = "ellipsis"


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
class LiteralSlice:
    """One ``slice(start, stop, step)`` literal in a sparse call recipe.

    Each component is an already-encoded ``LiteralAtom`` restricted to the
    ``NONE`` or ``INT`` kinds -- exactly what a Python ``slice`` object may
    hold. Reconstruction is the inert builtin ``slice(start, stop, step)``:
    no callables, no imports, no security surface.
    """

    start: LiteralAtom
    stop: LiteralAtom
    step: LiteralAtom


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


NonTensorLiteral: TypeAlias = (
    LiteralAtom | LiteralSlice | LiteralTorchSymbol | LiteralSequence | LiteralMapping
)
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
    """Name, ownership, role, alias, and declared-fact metadata for one state slot.

    r71 A owner-record obligations (REQUIRED, never defaulted):

    * ``captured_requires_grad`` / ``captured_grad_fn`` -- the TOTALIZED declared
      state-metadata facts (r65 F-1, E1-settled): the capture-time autograd trainable
      bit recorded for EVERY declared state name (staging applies it -- capture truth
      always wins) and grad_fn presence (``True`` refuses at save: no staged leaf can
      carry a grad_fn, so every admitted artifact records ``False``).
    * ``host_escape_disposition`` -- the per-slot host-escape claim consumed by strict
      state preparation and ``_unbound_state_escape_stale``: ``"escaped"`` demands an
      exact ``unbound_state_escape`` witness (SHA staleness guard) XOR a typed gap;
      ``"inert"`` is the explicit no-witness claim (reserved vocabulary -- the current
      producer cannot prove inertness and always claims ``"escaped"``); ``None`` for a
      graph-bound, non-escaping slot. Parse-time domain totality requires a non-None
      disposition on every slot of an unbound state name.
    """

    module_path: str
    state_dict_name: str
    semantic_role: StateSlotRole
    trainable: bool
    persistent: bool
    alias_group: str | None
    captured_requires_grad: bool
    captured_grad_fn: bool
    host_escape_disposition: Literal["escaped", "inert"] | None


@dataclass(frozen=True, slots=True)
class TensorSlotDescriptor:
    """Value-free allocation and binding contract for one tensor slot.

    r71 A owner-record obligations (REQUIRED, never defaulted):

    * ``host_escape`` -- this slot's value escaped to the Python host (or was baked
      verbatim into a downstream literal): an exact ``tensor_derived_scalar_literal``
      digest witness is REQUIRED for the slot XOR a typed :class:`WitnessCoverageGap`.
      Consumed by ``_tensor_derived_scalar_stale`` as its required domain.
    * ``inert_sink`` -- the explicit claim discharging terminal-slot totality for a
      genuinely dead call-produced slot (e.g. the unused half of a multi-output op).
      Mutually exclusive with ``host_escape``; parse-time terminal-slot accounting
      requires every terminal non-output call-produced slot to be claimed by EXACTLY
      one of {scalar_bool, loop_predicate, tensor_derived_scalar_literal, inert_sink,
      typed gap}.
    """

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
    host_escape: bool
    inert_sink: bool


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
    execution_context: "CallExecutionContext"
    control_obligations: tuple[CallControlObligation, ...]
    control_dependencies: tuple[ControlDependencyEdge, ...]


@dataclass(frozen=True, slots=True)
class AutocastDeviceContext:
    """Explicit autocast state for one device class at a resolved call.

    Disabled state is written affirmatively (``enabled=False``); absence of a
    device record is never interpreted as disabled.
    """

    device_type: str
    enabled: bool
    dtype: str | None


@dataclass(frozen=True, slots=True)
class CallExecutionContext:
    """Required per-call execution context recorded at capture (v2).

    Replay enters exactly this context tightly around the resolved call:
    explicit ``enabled=False`` autocast contexts are actively entered so a
    caller's ambient autocast cannot contaminate a disabled capture, and the
    recorded grad/inference mode is restored. Context entry never touches RNG.
    """

    autocast: tuple[AutocastDeviceContext, ...]
    grad_enabled: bool
    inference_mode: bool


@dataclass(frozen=True, slots=True)
class AmbientExecutionContext:
    """Capture-scoped ambient backend context (v2, required, explicit).

    ``None`` values mean the producing runtime did not expose that control
    (feature-detected through ``utils/_torch_compat.py``); every exposed
    control is recorded affirmatively, including its disabled state. Replay
    restores the recorded values transactionally around the whole run; a
    recorded value the runtime cannot restore is a typed refusal, never a
    silent ambient passthrough. ``attestation_ineligible_context`` is the
    positive capture-time marking for a nondeterministic context (for example
    ``cudnn.benchmark=True``): such a capture can replay and verify its path
    but never claims byte-exact numeric attestation.

    ``grad_enabled`` and ``inference_mode`` (r53 hon_1) record the capture-scoped
    GLOBAL autograd/inference mode as REQUIRED strict booleans: a Python branch
    on ``torch.is_grad_enabled()`` / ``torch.is_inference_mode_enabled()`` is
    steered by this global, so replay must re-enter the recorded mode (as scoped
    contexts around the whole sparse run) for a fresh instance to take the SAME
    branch. A v2 descriptor missing either field refuses at parse and loads
    analysis-only: there is no honest default -- a defaulted grad mode could
    bless a different-context comparison as ``verified``.
    ``fill_uninitialized_memory`` is feature-detected (``None`` when the runtime
    does not expose ``torch.utils.deterministic.fill_uninitialized_memory``) and
    governs the hon_2 deterministic-fill refinement for the ``empty`` op family.
    """

    default_dtype: str
    default_device: str
    float32_matmul_precision: str | None
    deterministic_algorithms: bool | None
    deterministic_algorithms_warn_only: bool | None
    cuda_matmul_allow_tf32: bool | None
    cudnn_allow_tf32: bool | None
    cudnn_deterministic: bool | None
    cudnn_benchmark: bool | None
    cudnn_enabled: bool | None
    flash_sdp_enabled: bool | None
    mem_efficient_sdp_enabled: bool | None
    math_sdp_enabled: bool | None
    grad_enabled: bool
    inference_mode: bool
    fill_uninitialized_memory: bool | None
    attestation_ineligible_context: bool


@dataclass(frozen=True, slots=True)
class InputAttestationFingerprint:
    """Physical identity of one capture-time model-input slot (v2).

    Recorded from the live capture-time input value (never from an archived
    payload, whose serialization contiguifies strides). At run time the value
    that actually seeds execution is fingerprinted on the same basis; any
    physical mismatch makes the run changed-input-for-attestation only:
    numeric attestation is ``not_applicable`` while path faithfulness stays
    governed by the ordinary input/control contracts.
    """

    slot_id: str
    byte_digest: str
    device_type: str
    device_index: int | None
    layout: str
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    storage_offset: int
    is_contiguous: bool
    is_channels_last: bool
    is_channels_last_3d: bool
    is_conj: bool
    is_neg: bool
    tensor_class: str
    requires_grad: bool
    is_inference: bool
    alignment_class: int


class ControlWitnessKind(str, Enum):
    """Observable control facts persisted in a sparse descriptor."""

    SCALAR_BOOL = "scalar_bool"
    CONDITIONAL_ARM_ENTRY = "conditional_arm_entry"
    LOOP_PREDICATE = "loop_predicate"
    SHAPE_STRUCTURE_FACT = "shape_structure_fact"
    TENSOR_DERIVED_SCALAR_LITERAL = "tensor_derived_scalar_literal"


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """One closed registry-v2 row: the typed presence policy for one witness family (r71 A).

    Every verdict-steering witness family declares IN CODE (never prose):

    * ``witness_kind`` -- the direct :class:`ControlWitnessKind` the family's witnesses
      carry (``SHAPE_STRUCTURE_FACT`` for prefix families); ``None`` for a claim-only
      family that never owns witnesses.
    * ``site_prefix`` -- the ``site_label`` prefix for ``SHAPE_STRUCTURE_FACT`` families,
      ``None`` for direct-kind and claim-only families.
    * ``disposition`` -- ``inventory_indexed`` (exact member equality with the persisted
      mirror), ``independent_ceiling`` (anchored by a bidirectional structural
      cross-anchor, no members), or ``claim_only`` (members are structural claims, no
      witnesses exist).
    * ``anchor`` -- the NAMED independent replay-structural domain the parser re-derives
      the family's required members from. Never the witness stream, the inventory, or
      the completeness summary: those are never authority for their own coverage.
    * ``discharge`` -- ``exact_witness`` (every required member must own an exact
      witness), ``witness_or_gap`` (an explicit source-linked
      :class:`WitnessCoverageGap` may discharge instead, ceiling the derived
      completeness floor), or ``claim_only``.
    * ``runtime_consumer`` -- the named runtime consumer proving the anchor is
      replay-consumed structure, never a second decorative ledger.
    """

    family: str
    witness_kind: ControlWitnessKind | None
    site_prefix: str | None
    disposition: str
    anchor: str
    discharge: str
    runtime_consumer: str


WITNESS_FAMILY_REGISTRY: Final[Mapping[str, FamilySpec]] = MappingProxyType(
    {
        # ---- direct control kinds (the r70 corr2recover-H1 perimeter closure) ----
        "scalar_bool": FamilySpec(
            family="scalar_bool",
            witness_kind=ControlWitnessKind.SCALAR_BOOL,
            site_prefix=None,
            disposition="inventory_indexed",
            anchor="call_control_obligations+terminal_slot_totality",
            discharge="witness_or_gap",
            runtime_consumer="_call_witness_checks",
        ),
        "loop_predicate": FamilySpec(
            family="loop_predicate",
            witness_kind=ControlWitnessKind.LOOP_PREDICATE,
            site_prefix=None,
            disposition="inventory_indexed",
            anchor="call_control_obligations+terminal_slot_totality",
            discharge="witness_or_gap",
            runtime_consumer="_call_witness_checks",
        ),
        "conditional_arm_entry": FamilySpec(
            family="conditional_arm_entry",
            witness_kind=ControlWitnessKind.CONDITIONAL_ARM_ENTRY,
            site_prefix=None,
            disposition="inventory_indexed",
            anchor="child_call_control_dependencies+op_label_referential_integrity"
            "+conditional_predicate_pairing",
            discharge="witness_or_gap",
            runtime_consumer="_conditional_arm_check",
        ),
        "tensor_derived_scalar_literal": FamilySpec(
            family="tensor_derived_scalar_literal",
            witness_kind=ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL,
            site_prefix=None,
            disposition="inventory_indexed",
            anchor="tensor_slot_host_escape_obligations+terminal_slot_totality",
            discharge="witness_or_gap",
            runtime_consumer="_tensor_derived_scalar_stale",
        ),
        # ---- SHAPE_STRUCTURE_FACT prefix families (r69 A rows, re-expressed) ----
        "input_structure": FamilySpec(
            family="input_structure",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="input_structure:",
            disposition="inventory_indexed",
            anchor="input_boundary_sites+model_input_bindings+dense_positional_roots",
            discharge="exact_witness",
            runtime_consumer="_input_structure_witness_check",
        ),
        "model_input_literal": FamilySpec(
            family="model_input_literal",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="model_input_literal:",
            disposition="independent_ceiling",
            anchor="bidirectional_structure_leaf_cross_anchor",
            discharge="exact_witness",
            runtime_consumer="_input_literal_contract_checks",
        ),
        "model_input_metadata": FamilySpec(
            family="model_input_metadata",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="model_input_metadata:",
            disposition="inventory_indexed",
            anchor="input_boundary_tensor_sites_totalized_envelope",
            discharge="exact_witness",
            runtime_consumer="_input_metadata_contract_checks",
        ),
        "module_training_mode": FamilySpec(
            family="module_training_mode",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="module_training_mode:",
            disposition="inventory_indexed",
            anchor="mode_sensitive_call_domain",
            discharge="exact_witness",
            runtime_consumer="_mode_sensitive_op_unwitnessed",
        ),
        "state_metadata": FamilySpec(
            family="state_metadata",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="state_metadata:",
            disposition="inventory_indexed",
            anchor="declared_state_name_totality+state_binding_fact_mirror",
            discharge="exact_witness",
            runtime_consumer="_apply_state_metadata_facts",
        ),
        "container": FamilySpec(
            family="container",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="container:",
            disposition="inventory_indexed",
            anchor="container_record_snapshot_totality",
            discharge="exact_witness",
            runtime_consumer="_structure_witness_check",
        ),
        "unbound_state_escape": FamilySpec(
            family="unbound_state_escape",
            witness_kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            site_prefix="unbound_state_escape:",
            disposition="inventory_indexed",
            anchor="state_binding_host_escape_disposition_totality",
            discharge="witness_or_gap",
            runtime_consumer="_unbound_state_escape_stale",
        ),
        # ---- explicit claim-only families (terminal-slot / unbound-name totality) ----
        "inert_sink": FamilySpec(
            family="inert_sink",
            witness_kind=None,
            site_prefix=None,
            disposition="claim_only",
            anchor="terminal_slot_totality",
            discharge="claim_only",
            runtime_consumer="terminal_slot_accounting",
        ),
        "unbound_state_inert": FamilySpec(
            family="unbound_state_inert",
            witness_kind=None,
            site_prefix=None,
            disposition="claim_only",
            anchor="state_binding_host_escape_disposition_totality",
            discharge="claim_only",
            runtime_consumer="strict_state_preparation",
        ),
    }
)
"""THE closed replay-critical witness-family registry v2 (r71 A).

One typed :class:`FamilySpec` row per verdict-steering witness family -- the four
direct control kinds (``scalar_bool`` / ``loop_predicate`` / ``conditional_arm_entry``
/ ``tensor_derived_scalar_literal``) PLUS the seven ``SHAPE_STRUCTURE_FACT`` prefix
families -- and the two explicit claim-only families (``inert_sink`` /
``unbound_state_inert``) that discharge terminal-slot / unbound-state-name domain
totality without witnesses. This registry is the ONLY dispatch authority for producer
authoring, parser validation, and the generated mutation matrix.

The agreed r71 invariant (deletion-closure): every replay-structure item that can
affect a faithfulness verdict creates a typed, independently derived OBLIGATION,
anchored to structure the replay itself consumes. Each obligation is discharged
exactly once, by an exact witness XOR an explicit, source-linked
:class:`WitnessCoverageGap`. The witness stream, the required-witness inventory, and
the ``witness_completeness`` summary are NEVER authority for their own required
coverage; the inventory is a redundant mirror/diagnostic. Consequence: no combination
of record DELETIONS can improve a verdict -- any partial strip leaves a surviving
anchor contradiction (parse-refuse ``context_field_invalid``, analysis-only) or an
UNVERIFIABLE floor, never VERIFIED. The single out-of-scope boundary is coherent
reauthoring (see the threat-model subsection of
``docs/reference/runnable_tlspec_contract.md``).
"""

VERDICT_STEERING_WITNESS_FAMILIES: Final[frozenset[str]] = frozenset(
    family for family, spec in WITNESS_FAMILY_REGISTRY.items() if spec.witness_kind is not None
)
"""The 11 witness-carrying registry families (claim-only rows excluded)."""


class WitnessGapKind(str, Enum):
    """Closed vocabulary of explicit witness-coverage gap causes (r71 A3).

    Every producer-side completeness downgrade is one of these typed, source-linked
    causes; a new gap constructor without a registered row REDs the r71 meta-test.
    """

    OPAQUE_INPUT_LEAF = "opaque_input_leaf"
    UNWITNESSABLE_ESCAPE_SOURCE = "unwitnessable_escape_source"
    UNATTRIBUTABLE_BOOL_ESCAPE = "unattributable_bool_escape"
    UNATTRIBUTABLE_OPAQUE_ESCAPE = "unattributable_opaque_escape"
    MUTABLE_WRITEBACK_ESCAPE = "mutable_writeback_escape"
    RAW_POINTER_ESCAPE = "raw_pointer_escape"
    ESCAPE_OBSERVER_UNCERTAIN = "escape_observer_uncertain"
    CROSS_THREAD_TENSOR_ACCESS = "cross_thread_tensor_access"
    UNWITNESSABLE_STATE_ESCAPE = "unwitnessable_state_escape"
    UNOBSERVED_PREDICATE = "unobserved_predicate"
    UNCLASSIFIED_TERMINAL_BOOL = "unclassified_terminal_bool"
    UNANCHORABLE_ARM_EDGE = "unanchorable_arm_edge"
    FORWARD_VALUE_OVERRIDE = "forward_value_override"
    INPUT_METADATA_VIEW_READ = "input_metadata_view_read"
    PRUNED_RNG_CONTROL = "pruned_rng_control"
    PRUNED_ALIAS_MUTATION = "pruned_alias_mutation"
    UNMODELLED_HOST_WRITE = "unmodelled_host_write"
    LIFECYCLE_LEDGER_MUTATION = "lifecycle_ledger_mutation"
    LIFECYCLE_LEDGER_DISPATCH = "lifecycle_ledger_dispatch"
    RNG_MONITOR_UNCERTAIN = "rng_monitor_uncertain"
    CAPTURE_VERIFICATION_FAILED = "capture_verification_failed"


@dataclass(frozen=True, slots=True)
class GapSpec:
    """Closed gap-registry row: the structural source and resulting floor per cause."""

    source_family: str
    resulting_completeness: "WitnessCompleteness"


WITNESS_GAP_REGISTRY: Final[Mapping[WitnessGapKind, GapSpec]] = MappingProxyType(
    {
        WitnessGapKind.OPAQUE_INPUT_LEAF: GapSpec(
            "model_input_literal", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.UNWITNESSABLE_ESCAPE_SOURCE: GapSpec(
            "tensor_derived_scalar_literal", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.UNATTRIBUTABLE_BOOL_ESCAPE: GapSpec(
            "scalar_bool", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.UNATTRIBUTABLE_OPAQUE_ESCAPE: GapSpec(
            "tensor_derived_scalar_literal", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.MUTABLE_WRITEBACK_ESCAPE: GapSpec(
            "tensor_derived_scalar_literal", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.RAW_POINTER_ESCAPE: GapSpec(
            "tensor_derived_scalar_literal", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.ESCAPE_OBSERVER_UNCERTAIN: GapSpec(
            "tensor_derived_scalar_literal", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.CROSS_THREAD_TENSOR_ACCESS: GapSpec(
            "tensor_derived_scalar_literal", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.UNWITNESSABLE_STATE_ESCAPE: GapSpec(
            "unbound_state_escape", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.UNOBSERVED_PREDICATE: GapSpec(
            "scalar_bool", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.UNCLASSIFIED_TERMINAL_BOOL: GapSpec(
            "scalar_bool", WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        ),
        WitnessGapKind.UNANCHORABLE_ARM_EDGE: GapSpec(
            "conditional_arm_entry", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.FORWARD_VALUE_OVERRIDE: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
        ),
        WitnessGapKind.INPUT_METADATA_VIEW_READ: GapSpec(
            "model_input_metadata", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.PRUNED_RNG_CONTROL: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.PRUNED_ALIAS_MUTATION: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
        ),
        WitnessGapKind.UNMODELLED_HOST_WRITE: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
        ),
        WitnessGapKind.LIFECYCLE_LEDGER_MUTATION: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
        ),
        WitnessGapKind.LIFECYCLE_LEDGER_DISPATCH: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.RNG_MONITOR_UNCERTAIN: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
        ),
        WitnessGapKind.CAPTURE_VERIFICATION_FAILED: GapSpec(
            "capture", WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
        ),
    }
)
"""Closed gap registry: source family + resulting completeness floor per gap kind.

``source_family`` is a :data:`WITNESS_FAMILY_REGISTRY` family for obligation-anchored
gaps (stripping such a gap leaves its obligation undischarged -> parse-refuse) or the
symbolic ``"capture"`` source for capture-event-only causes whose only closure is the
coherent-reauthoring boundary (documented in the contract's threat-model subsection).
"""


@dataclass(frozen=True, slots=True)
class WitnessCoverageGap:
    """One ordered, typed, source-linked witness-coverage gap record (r71 A3).

    Replaces the producer's former direct ``witness_completeness`` enum assignments:
    every downgrade cause persists as an explicit gap carrying its closed
    :class:`WitnessGapKind`, the registry source family, the canonical source member
    (a slot/state/site identity, or ``"capture"`` for capture-global causes), the
    capture order, and the resulting completeness (which must equal the registry row).
    The parser derives the completeness FLOOR from these records plus the structural
    obligations; the persisted summary is a redundant assertion only.
    """

    gap_kind: WitnessGapKind
    source_family: str
    source_member: str
    order: int
    resulting_completeness: WitnessCompleteness


def derived_witness_completeness(
    gaps: "tuple[WitnessCoverageGap, ...]",
) -> WitnessCompleteness:
    """Derive the completeness FLOOR from the ordered gap ledger (r71 A3).

    THE single derivation authority: the producer's persisted summary, the parser's
    floor check, readiness, and the ``_path_faithfulness`` VERIFIED gate all consult
    this function; no verdict/readiness code may read the raw persisted
    ``witness_completeness`` outside the parse-time equality check (source-scan
    meta-test). The floor is the resulting completeness of the FIRST gap in capture
    order (matching the historical first-cause-wins producer cascade), or ``COMPLETE``
    for an empty ledger. A persisted summary claiming STRONGER than this floor is an
    internally contradictory descriptor (``context_field_invalid``).
    """

    if not gaps:
        return WitnessCompleteness.COMPLETE
    first = min(gaps, key=lambda gap: gap.order)
    return first.resulting_completeness


@dataclass(frozen=True, slots=True)
class CallControlObligation:
    """Owner-record scalar-bool / loop-predicate obligation on one call (r71 A2).

    Stamped on the OWNING :class:`RunnableCallDescriptor` (the call producing the
    predicate slot) and consumed by the runtime check builder
    (``_call_witness_checks``): every obligation demands an exact same-kind witness
    with this call/site identity XOR a typed :class:`WitnessCoverageGap`.
    ``conditional_id`` links the predicate to its arm-entry edges (E2 pairing).
    """

    kind: ControlWitnessKind
    output_slot_id: str
    site_label: str
    conditional_id: int | None


@dataclass(frozen=True, slots=True)
class ControlDependencyEdge:
    """Owner-record conditional arm-entry edge on the CHILD call (r71 A2).

    Consumed by scheduling/topological validation (parent call precedes child call;
    labels resolve against descriptor slots) and by the runtime arm check domain.
    """

    conditional_id: int
    arm_kind: str
    parent_op_label: str
    child_op_label: str


def control_dependency_site_label(edge: ControlDependencyEdge) -> str:
    """Return the canonical arm-entry witness ``site_label`` for one edge record."""

    return (
        f"conditional:{edge.conditional_id}:{edge.arm_kind}:"
        f"{edge.parent_op_label}->{edge.child_op_label}"
    )


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
class RequiredWitnessFamily:
    """One required-witness inventory row: family, disposition, exact member IDs (r69 A)."""

    family: str
    disposition: str
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RequiredWitnessInventory:
    """REQUIRED descriptor-native presence ledger for replay-critical witness families.

    Structurally OUTSIDE ``control_witnesses`` -- the stream being validated can never
    declare its own required membership -- and authored by both producer and parser
    from the ONE closed :data:`WITNESS_FAMILY_REGISTRY`. Carries one row per
    registered family (explicit empty member sets included) plus the closed
    registry/version discriminator; the parser requires EXACT family and member
    coverage and refuses any deficit, surplus, duplicate, unknown family, or
    malformed row as ``context_field_invalid`` (analysis load survives, readiness is
    unavailable, execution cannot attach). This is a presence ledger, never a
    replacement for the facts themselves: ``site_count`` values inside facts become
    redundant consistency checks, never required-set authority, and absence of a
    family/member can never restore weaker semantics.
    """

    registry_version: str
    families: tuple[RequiredWitnessFamily, ...]


@dataclass(frozen=True, slots=True)
class InputBoundaryTensorSite:
    """One tensor leaf of the REQUIRED input-boundary record (r71 A2).

    ``metadata_reads`` is the EXPLICIT (possibly empty) closed-vocabulary set of
    metadata-predicate names the captured forward read on this leaf: the totalized
    ``model_input_metadata`` envelope witness for this (site, path) must carry exactly
    these fact names (the witness carries the observed VALUES; this owner record
    carries the read-site existence). Cross-checked at parse against the tensor
    MODEL_INPUT slot bindings -- never against the witness stream.
    """

    container_path: ContainerPath
    slot_id: str
    metadata_reads: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InputBoundarySite:
    """One REQUIRED top-level input-boundary site record (r71 A2).

    THE runtime site/arity and tree-binding authority (one exact record per captured
    arg/kwarg, canonical ``arg:<i>`` / ``kwarg:<name>`` position). Replaces the
    required-witness inventory as ``_inventory_site_positions``' authority -- the
    inventory is a redundant mirror. Consumed by runtime site selection, arity
    contracts, and the metadata-envelope domain.
    """

    position: str
    tensor_sites: tuple[InputBoundaryTensorSite, ...]


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
    schema: Literal["selected_activation_v2"]
    members: tuple[ActivationPayloadMember, ...]
    original_input_digests: tuple[SlotByteDigest, ...]
    capture_state_digests: tuple[StateByteDigest, ...]
    input_fingerprints: tuple[InputAttestationFingerprint, ...]


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

    capability: Literal["sparse_recorded_taken_path_v2"]
    backend: str
    call_recipe: Literal["non_tensor_args_tensor_slots_context_and_obligations_v3"]
    callable_ref_schema: Literal[1]
    state_binding: Literal["module_path_role_v1"]
    input_binding: Literal["model_site_io_role_v1"]
    control_witness: Literal["scalar_bool_and_arm_entry_v1"]
    initializer_policy_version: Literal["torchlens_role_init_v2"]
    payload_layers: PayloadLayersDescriptor
    callable_registry: tuple[CallableRegistryEntry, ...]
    calls: tuple[RunnableCallDescriptor, ...]
    tensor_slots: tuple[TensorSlotDescriptor, ...]
    input_boundary: tuple[InputBoundarySite, ...]
    control_witnesses: tuple[ControlWitness, ...]
    coverage_gaps: tuple[WitnessCoverageGap, ...]
    required_witness_inventory: RequiredWitnessInventory
    witness_completeness: WitnessCompleteness
    rng_profile: RunnableRngProfile
    ambient_context: AmbientExecutionContext
    compatibility: RunnableCompatibility
    preflight: ProducerPreflight
    unsupported_sites: tuple[RunnableDiagnostic, ...]


def is_mode_sensitive_qualname(qualname: str | None) -> bool:
    """Return whether a qualname is a train/eval mode-sensitive op (BatchNorm family).

    BatchNorm / InstanceNorm produce numerically different results in eval (running
    statistics) vs train (batch statistics) mode. Single-sourced here (r71 A) so the
    parse-time ``module_training_mode`` domain derivation and the runtime
    ``_mode_sensitive_op_unwitnessed`` belt can never disagree. Dropout is intentionally
    excluded: its train arm is already RNG-tainted (``not_applicable``) and its eval arm
    is identity, so it never creates a mode-driven false VERIFIED.
    """

    if not qualname:
        return False
    tail = qualname.rsplit(".", 1)[-1]
    if tail.endswith("_"):
        tail = tail[:-1]
    return "batch_norm" in tail or tail.endswith("instance_norm")


@dataclass(frozen=True, slots=True)
class ReplayWitnessStructure:
    """The frozen WITNESS-FREE replay-structure view (r71 A2).

    Built ONLY from replay-essential structures -- calls, tensor slots, the required
    input-boundary record, and the callable registry (plus, when the container source
    is available, the container-record snapshot identities). By construction it cannot
    contain or expose ``control_witnesses``, ``coverage_gaps``,
    ``required_witness_inventory``, or ``witness_completeness``: the typed input makes
    accidental self-referential coverage derivation impossible (r70 free-F1 root).

    ``container_members`` is ``None`` when the container source (the rehydrated
    ``Trace._containers`` records) is not in scope -- the pure-parse stage; the
    container family's independent anchor is then enforced at readiness attach, where
    the loaded trace is available. Producer-side it is always populated.
    """

    calls: tuple[RunnableCallDescriptor, ...]
    tensor_slots: tuple[TensorSlotDescriptor, ...]
    input_boundary: tuple[InputBoundarySite, ...]
    callable_registry: tuple[CallableRegistryEntry, ...]
    container_members: tuple[str, ...] | None

    @classmethod
    def from_descriptor(
        cls,
        descriptor: "SparseRunDescriptor",
        *,
        container_members: "tuple[str, ...] | None" = None,
    ) -> "ReplayWitnessStructure":
        """Project a descriptor onto its witness-free replay structure."""

        return cls(
            calls=descriptor.calls,
            tensor_slots=descriptor.tensor_slots,
            input_boundary=descriptor.input_boundary,
            callable_registry=descriptor.callable_registry,
            container_members=container_members,
        )


def derive_required_witness_members(
    structure: ReplayWitnessStructure,
) -> "dict[str, list[str]]":
    """Derive every family's REQUIRED member identities from replay structure (r71 A2).

    THE independent required-coverage authority: iterates the closed
    :data:`WITNESS_FAMILY_REGISTRY` anchors over the witness-free
    :class:`ReplayWitnessStructure` only. The producer authors witnesses, gaps, and
    the inventory mirror FROM this output (structure before witnesses); the parser
    independently rebuilds the structure, re-derives, and requires exact discharge.
    Deleting or replacing the witness stream, the inventory, or the persisted summary
    cannot change this output (the r71 independence meta-test).

    Per-family anchor sources (the merged r71 anchor table):

    * ``scalar_bool`` / ``loop_predicate`` -- ``CallControlObligation`` owner records.
    * ``conditional_arm_entry`` -- ``ControlDependencyEdge`` owner records on the
      child call.
    * ``tensor_derived_scalar_literal`` -- ``TensorSlotDescriptor.host_escape``.
    * ``inert_sink`` -- ``TensorSlotDescriptor.inert_sink`` (claim only).
    * ``input_structure`` -- the required input-boundary site records.
    * ``model_input_literal`` -- none here (independent bidirectional ceiling).
    * ``model_input_metadata`` -- the totalized envelope domain: one member per
      input-boundary tensor site.
    * ``module_training_mode`` -- mode-sensitive calls derived from the callable
      registry / call DAG (a REQUIRED-minimum: honest artifacts may declare the mode
      without any mode-sensitive op).
    * ``state_metadata`` -- the declared state-name universe x the closed two-fact
      vocabulary (totalized, E1).
    * ``unbound_state_escape`` / ``unbound_state_inert`` --
      ``StateSlotBinding.host_escape_disposition`` claims.
    * ``container`` -- the container-record snapshot identities when in scope.
    """

    members: dict[str, list[str]] = {family: [] for family in WITNESS_FAMILY_REGISTRY}
    for call in structure.calls:
        for obligation in call.control_obligations:
            family = (
                "loop_predicate"
                if obligation.kind is ControlWitnessKind.LOOP_PREDICATE
                else "scalar_bool"
            )
            members[family].append(f"{call.call_id}::{obligation.site_label}")
        for edge in call.control_dependencies:
            members["conditional_arm_entry"].append(control_dependency_site_label(edge))
    state_names: dict[str, None] = {}
    for slot in structure.tensor_slots:
        if slot.host_escape:
            members["tensor_derived_scalar_literal"].append(slot.slot_id)
        if slot.inert_sink:
            members["inert_sink"].append(slot.slot_id)
        binding = slot.state_binding
        if binding is not None:
            state_names.setdefault(binding.state_dict_name, None)
            if binding.host_escape_disposition == "escaped":
                members["unbound_state_escape"].append(f"{binding.state_dict_name}::{slot.slot_id}")
            elif binding.host_escape_disposition == "inert":
                members["unbound_state_inert"].append(f"{binding.state_dict_name}::{slot.slot_id}")
    metadata_prefix = WITNESS_FAMILY_REGISTRY["model_input_metadata"].site_prefix
    for site in structure.input_boundary:
        members["input_structure"].append(site.position)
        position = decode_input_site_position(site.position)
        for tensor_site in site.tensor_sites:
            members["model_input_metadata"].append(
                f"{metadata_prefix}{position!r}:{list(tensor_site.container_path)!r}"
            )
    registry_by_id = {entry.registry_id: entry for entry in structure.callable_registry}
    mode_sensitive = False
    for call in structure.calls:
        entry = registry_by_id.get(call.registry_id)
        if entry is not None and is_mode_sensitive_qualname(entry.key.qualname):
            mode_sensitive = True
            break
    if mode_sensitive:
        mode_prefix = WITNESS_FAMILY_REGISTRY["module_training_mode"].site_prefix
        members["module_training_mode"].append(str(mode_prefix))
    for name in state_names:
        members["state_metadata"].append(f"{name}::grad_fn")
        members["state_metadata"].append(f"{name}::requires_grad")
    if structure.container_members is not None:
        members["container"].extend(structure.container_members)
    return members


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


NONDETERMINISTIC_SOURCE_VOCABULARY: Final[frozenset[str]] = frozenset(
    {"seeded_rng", "host_rng", "uninitialized_alloc"}
)
"""Closed vocabulary for ``RunReport.nondeterministic_sources`` (r53 F4)."""


@dataclass(frozen=True, slots=True)
class RunReport:
    """Honesty, state, resolution, and contract metadata for one run.

    ``nondeterministic_sources`` (r53 F4) is the closed, sorted, deduplicated
    declared-source vocabulary (``NONDETERMINISTIC_SOURCE_VOCABULARY``), derived
    ONLY by the single report finalizer. It distinguishes a
    declared-nondeterministic path-only ``verified`` (an unseeded ``torch.rand``
    or an uninitialized ``empty`` product escaping to the output) from a
    deterministic ``verified``; it never alters verdict semantics.
    """

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
    nondeterministic_sources: tuple[str, ...]


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
    "AmbientExecutionContext",
    "ArchivedActivation",
    "AutocastDeviceContext",
    "CANONICAL_INITIALIZER_BY_ROLE",
    "LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS",
    "RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION",
    "RUNNABLE_CALLABLE_REF_SCHEMA_VERSION",
    "RUNNABLE_CALL_RECIPE_VERSION",
    "RUNNABLE_INITIALIZER_POLICY_VERSION",
    "RUNNABLE_TLSPEC_SCHEMA_VERSION",
    "CallControlObligation",
    "CallExecutionContext",
    "CallableRegistryEntry",
    "InputAttestationFingerprint",
    "InputBoundarySite",
    "InputBoundaryTensorSite",
    "ContractCheck",
    "ControlDependencyEdge",
    "ControlWitness",
    "ControlWitnessKind",
    "DivergencePolicy",
    "FamilySpec",
    "GapSpec",
    "ReplayWitnessStructure",
    "VERDICT_STEERING_WITNESS_FAMILIES",
    "WITNESS_FAMILY_REGISTRY",
    "WITNESS_FAMILY_REGISTRY_VERSION",
    "WITNESS_GAP_REGISTRY",
    "WitnessCoverageGap",
    "WitnessGapKind",
    "control_dependency_site_label",
    "derive_required_witness_members",
    "derived_witness_completeness",
    "is_mode_sensitive_qualname",
    "InitializerPolicy",
    "InputSlotBinding",
    "LiteralArgumentRef",
    "LiteralAtom",
    "LiteralAtomKind",
    "LiteralMapping",
    "LiteralMappingEntry",
    "LiteralSequence",
    "LiteralSequenceKind",
    "LiteralSlice",
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
