"""Backend-neutral capture event records for the unified TorchLens pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .._io import FieldPolicy
from .container import ContainerSpec

if TYPE_CHECKING:
    from ..intervention.types import FunctionRegistryKey
    from .intervention import FireResult, InterventionTemplateRef
    from .refs import ParamRef, TensorRef
    from .semantics import BackendSemantics, CapturePolicy

# "intervention_replacement" is deliberately NOT an operation kind: an
# intervention is an EDIT (an InterventionAppliedEvent referencing its target),
# never a synthetic operation, so a functionless op can no longer be expressed
# as a legal kind. (No producer ever constructed the retired literal.)
OpEventKind = Literal["op", "source", "synthetic_output"]
EdgeUseKind = Literal["arg", "kwarg", "container", "module", "buffer", "output", "control"]
JaxEquationKind = Literal[
    "primitive",
    "scan_read",
    "scan_stack",
    "cond_decision",
    "while_decision",
]
BackwardTrigger = Literal[
    "backward",
    "autograd_grad",
    "autograd_backward",
    "recording_backward",
    "implicit",
    "replay",
]
BackwardStatus = Literal["ok", "error"]


@dataclass(frozen=True, slots=True)
class _AtenTensorFact:
    """Value-free tensor metadata observed at one dispatcher boundary."""

    container_path: tuple[object, ...]
    tensor_impl_capability: str
    logical_version: int | None
    storage_alias_group: int | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "container_path": FieldPolicy.KEEP,
        "tensor_impl_capability": FieldPolicy.KEEP,
        "logical_version": FieldPolicy.KEEP,
        "storage_alias_group": FieldPolicy.KEEP,
        "shape": FieldPolicy.KEEP,
        "stride": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "device": FieldPolicy.KEEP,
        "layout": FieldPolicy.KEEP,
        "requires_grad": FieldPolicy.KEEP,
    }


@dataclass(frozen=True, slots=True)
class _AtenExecutionContext:
    """Immutable execution-environment stamp for one dispatcher call."""

    pytorch_version: str
    backend: str
    device_model: str | None
    device_capability: tuple[int, int] | None
    grad_mode: bool
    inference_mode: bool
    module_training_summary: tuple[tuple[str, bool], ...]
    autocast: tuple[tuple[str, bool, str], ...]
    deterministic_algorithms: bool
    tf32_matmul_policy: bool | None
    sdpa_policy: tuple[tuple[str, bool], ...]
    compile_stance: str
    owner_thread_coverage: tuple[int, ...]
    completeness_witness_mode: str

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "pytorch_version": FieldPolicy.KEEP,
        "backend": FieldPolicy.KEEP,
        "device_model": FieldPolicy.KEEP,
        "device_capability": FieldPolicy.KEEP,
        "grad_mode": FieldPolicy.KEEP,
        "inference_mode": FieldPolicy.KEEP,
        "module_training_summary": FieldPolicy.KEEP,
        "autocast": FieldPolicy.KEEP,
        "deterministic_algorithms": FieldPolicy.KEEP,
        "tf32_matmul_policy": FieldPolicy.KEEP,
        "sdpa_policy": FieldPolicy.KEEP,
        "compile_stance": FieldPolicy.KEEP,
        "owner_thread_coverage": FieldPolicy.KEEP,
        "completeness_witness_mode": FieldPolicy.KEEP,
    }


@dataclass(frozen=True, slots=True)
class _AtenCallEvent:
    """Value-free facts for one observed ATen dispatcher call."""

    capture_phase: str
    forward_pass_index: int | None
    backward_epoch_index: int | None
    owner_func_call_id: int | None
    parent_grad_fn_call_ref: tuple[int, int, int] | None
    namespace: str
    operator: str
    overload: str
    schema: str | None
    schema_fingerprint: str | None
    module_call_stack: tuple[tuple[str, int], ...]
    input_tensor_facts: tuple[_AtenTensorFact, ...]
    output_tensor_facts: tuple[_AtenTensorFact, ...]
    mutation_kind: str
    view_copy_kind: str
    autocast_context: tuple[tuple[str, bool, str], ...]
    dispatch_key_context: str | None
    grad_fn_ref: str | None
    grad_fn_link_status: str
    grad_fn_link_provenance: str | None
    algorithmic_flops: int | None
    flop_status: str
    flop_formula_source: str | None
    flop_formula_version: str | None
    outcome: str
    exception_type: str | None
    execution_context: _AtenExecutionContext
    seq: int = 0


@dataclass(frozen=True, slots=True)
class _ModePausedInteriorEvent:
    """Boundary-only disclosure for a strict constructor's unobserved interior."""

    capture_phase: str
    sequence_before: int
    sequence_after: int
    owner_func_call_id: int | None
    reason: str = "strict_subclass_constructor"
    seq: int = 0


@dataclass(frozen=True, slots=True)
class BlobRef:
    """Portable reference to an externalized tensor/blob payload."""

    uri: str
    format: str
    dtype: str | None
    shape: tuple[int, ...] | None
    byte_length: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class BackwardPassStart:
    """Core event marking the beginning of one autograd engine invocation."""

    pass_index: int
    trigger: BackwardTrigger
    implicit: bool
    outer_context: str | None
    call_context_ref: object | None
    root_meta: tuple[object, ...]
    root_grad_arguments: object | None
    inputs_subset: tuple[object, ...]
    order: int | None
    origin_backward_pass: int | None
    save_grads_policy_repr: str | None
    engine_flags: dict[str, object] | None
    forward_op_count_at_trigger: int | None
    timestamp: float
    seq: int = 0


@dataclass(frozen=True, slots=True)
class OpGradObserved:
    """Core event emitted when a tensor hook observes an operation gradient."""

    op_label: str
    pass_index: int
    payload_ref: object | None
    transformed_payload_ref: object | None
    shape: tuple[int, ...] | None
    dtype: str | None
    memory: int | None
    timestamp: float
    seq: int = 0


@dataclass(frozen=True, slots=True)
class ParamGradObserved:
    """Core event emitted when an AccumulateGrad hook observes a parameter gradient."""

    param_address: str
    pass_index: int
    payload_ref: object | None
    shape: tuple[int, ...] | None
    dtype: str | None
    memory: int | None
    timestamp: float
    seq: int = 0


@dataclass(frozen=True, slots=True)
class BackwardPassEnd:
    """Core event marking completion of one autograd engine invocation.

    ``close_path`` is the implicit-pass close-path disclosure (L9 memo 1.2;
    provisional spelling, DOCUMENTED-UNSTABLE pending naming-session/S2
    routing): ``"engine_drain"`` when the queued engine final callback
    journaled the close, ``"sync_point"`` for every backstop path, ``None``
    for explicit (non-implicit) passes. Sidecar-event-only in wave 2 -- the
    projected ``BackwardPass`` record field waits for the wave-3 bump.
    """

    pass_index: int
    duration: float | None
    peak_memory: int | None
    status: BackwardStatus
    order_attribution_coverage: float | None
    close_path: str | None = None
    seq: int = 0


@dataclass(frozen=True, slots=True)
class GradFnDiscovered:
    """Torch enrichment event for a discovered autograd node object.

    ``source`` is deep-frozen by the stream writer
    (:meth:`~torchlens.ir.capture_events.CaptureEvents.append_backward`
    snapshots it into a read-only mapping): the backward projection copies it
    by value at materialize time, so it must be immutable on the event or an
    in-place mutation could bypass ``backward_revision``.
    """

    object_id: int
    class_name: str
    class_qualname: str
    is_custom: bool
    op_label: str | None
    param_ref: object | None
    created_in_pass: int | None
    creator_object_id: int | None
    source: Mapping[str, object | None]
    topology: tuple[int, ...]
    seq: int = 0


BackwardCoverageGapReason = Literal[
    "registration_error",
    "framework_unhookable",
    "dead_node",
    "unsupported_kind",
    "capture_exception",
    "suppressed",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class BackwardCoverageGap:
    """Core event recording one autograd node the walk could not observe.

    A hook-registration or discovery skip is a typed journal fact, never a
    silent ``continue``: only proven framework-contract exclusions preserve a
    complete-coverage claim, and validation fails closed on every other
    reason.
    """

    pass_index: int
    object_id: int | None
    class_qualname: str | None
    reason: BackwardCoverageGapReason
    detail: str | None
    timestamp: float
    seq: int = 0


@dataclass(frozen=True, slots=True)
class GradFnFired:
    """Torch enrichment event emitted from an autograd node hook.

    ``fire_started_monotonic`` / ``fire_finished_monotonic`` are the L9
    per-fire timing pair (provisional spellings, DOCUMENTED-UNSTABLE): BOTH
    stamps come from ``time.perf_counter()`` in the same process, paired at
    capture time by the per-node keyed LIFO, and enter this ONE event
    together -- projection never re-pairs them. An untimed fire (empty LIFO,
    key mismatch, timing-registration failure) carries ``(None, None)``,
    never a cross-fire or cross-clock pair. The wall-clock ``timestamp``
    stays the event-ordering stamp and is NEVER a duration operand.
    """

    object_id: int
    pass_index: int
    grad_input_refs: object | None
    grad_output_refs: object | None
    intervention_fire_ref: object | None
    timestamp: float
    fire_started_monotonic: float | None = None
    fire_finished_monotonic: float | None = None
    seq: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointInvocationObserved:
    """Torch enrichment event minted for one classified checkpoint invocation.

    L9 memo 2.3 (provisional spelling, DOCUMENTED-UNSTABLE): the token is a
    per-trace monotonic ordinal minted ONLY in the patched
    ``saved_tensors_hooks.__enter__`` for ``_checkpoint_hook`` instances on
    the armed owner thread outside any engine invocation. Pack counts and
    unpack window evidence accumulate in runtime token state, not on this
    frozen event; the projected summary lands on the DROP-gated Trace
    checkpoint-invocation witness field.
    """

    token: int
    timestamp: float
    seq: int = 0


@dataclass(frozen=True, slots=True)
class OutputRef:
    """Captured output metadata and optional payload references."""

    tensor: TensorRef
    transformed_tensor: TensorRef | None
    has_saved_activation: bool
    output_device: str | None
    activation_transform: object | None
    detach_saved_activations: bool
    visualizer_path: str | None
    multi_output_index: int | None
    in_multi_output: bool
    container_path: tuple[object, ...]
    container_spec: ContainerSpec | None
    child_versions: tuple[tuple[str, TensorRef], ...]


@dataclass(frozen=True, slots=True)
class OutputVersionEvent:
    """Pre-child parent output snapshot for replay validation."""

    parent_raw_label: str
    child_raw_label: str
    child_output_path: tuple[object, ...]
    payload: object
    transform_state: object | None
    detach_grad_policy: bool
    seq: int = 0


@dataclass(frozen=True, slots=True)
class FunctionCallRef:
    """Backend-neutral function call summary captured for an op event."""

    func: object | None
    func_name: str | None
    func_qualname: str | None
    func_call_id: int | None
    code_context: tuple[object, ...]
    func_duration: float | None
    flops_forward: int | None
    flops_backward: int | None
    func_rng_states: object | None
    func_autocast_state: object | None
    arg_names: tuple[str, ...]
    num_args_total: int
    num_pos_args: int
    num_kwargs: int
    non_tensor_pos_args: tuple[object, ...]
    non_tensor_kwargs: tuple[tuple[str, object], ...]
    func_non_tensor_args: tuple[object, ...]
    is_inplace: bool
    func_config: tuple[tuple[str, object], ...]
    func_id: FunctionRegistryKey | None = None


@dataclass(frozen=True, slots=True)
class ArgTemplateRef:
    """References to saved argument values and replay templates."""

    saved_args: object | None
    saved_kwargs: object | None
    args_template: object | None
    kwargs_template: object | None
    has_saved_args: bool


@dataclass(frozen=True, slots=True)
class ParentEdge:
    """Raw parent dependency edge for graph construction."""

    parent_label_raw: str
    arg_position: object
    edge_use: str


def edge_use_kind(record: object) -> str | None:
    """Return the semantic edge-use kind stored on an edge record.

    Parameters
    ----------
    record
        Edge-use record. Current backends may provide an object with an
        ``edge_use`` attribute or the legacy tuple shape
        ``(parent_label, arg_position, edge_use)``.

    Returns
    -------
    str | None
        Semantic edge-use kind, or ``None`` when the record does not expose one.
    """

    attr_kind = getattr(record, "edge_use", None)
    if isinstance(attr_kind, str):
        return attr_kind
    if isinstance(record, tuple) and len(record) >= 3 and isinstance(record[2], str):
        return record[2]
    return None


def is_control_edge_use(record: object) -> bool:
    """Return whether an edge record expresses a control dependency.

    Parameters
    ----------
    record
        Edge-use record to classify.

    Returns
    -------
    bool
        ``True`` when ``record`` is marked as a control edge.
    """

    return edge_use_kind(record) == "control"


def is_value_edge_use(record: object) -> bool:
    """Return whether an edge record participates in value replay.

    Parameters
    ----------
    record
        Edge-use record to classify.

    Returns
    -------
    bool
        ``True`` for data/argument-style edges, ``False`` for control edges.
        Unknown records conservatively count as value edges for compatibility.
    """

    return not is_control_edge_use(record)


@dataclass(frozen=True, slots=True)
class ModuleFrame:
    """Single active module-call frame at capture time."""

    address: str
    address_normalized: str | None
    module_type: str
    call_index: int
    fx_qualpath: str | None
    entry_argnames: tuple[str, ...]


InterventionEditKind = Literal["replaced", "fired"]
InterventionEditOrigin = Literal["raw_forward_hook", "live_fire", "push"]


@dataclass(frozen=True, slots=True)
class InterventionAppliedEvent:
    """Journal edit record for one observed intervention on a captured value.

    Interventions are EDITS referencing an existing identity, never op kinds:
    the record is appended only by the capture sites that directly observed
    the edit (a raw ``register_forward_hook`` returning a new object, or a
    live-fire hook reporting ``replaced=True`` while intervention machinery
    is armed for this capture), so it is the trace-level ground truth the
    functionless-op validation carve-out requires. A placeholder minted
    during PLAIN capture can never mint one of these and must still fail
    validation (2026-06-02 lesson).

    Causal binding: the observing site stamps ``run_token`` (the owning
    stream's run nonce), ``target_seq`` (the journal seq of the edited op's
    event at observation time), and ``target_func_call_id``. Validation
    accepts an edit only when the token matches the validated stream's nonce
    AND the journal really contains the bound target event, so a bare record
    appended through the ordinary writer (a forged edit) and a genuine record
    replayed into a DIFFERENT run's journal both stay refused. The sanctioned
    merge path (``CaptureEvents.concat``) re-binds tokens and target seqs for
    events that were genuinely bound to their source run. An in-process
    forger who also copies a live stream's nonce and a real target binding is
    outside this record's threat model (coherent reauthoring), the same
    documented boundary the runnable contract draws.
    """

    label_raw: str
    kind: InterventionEditKind
    origin: InterventionEditOrigin
    timestamp: float
    seq: int = 0
    run_token: int | None = None
    target_seq: int = 0
    target_func_call_id: int | None = None


BufferWriteKind = Literal["reassign", "inplace", "fused", "data_reassign"]


@dataclass(frozen=True, slots=True)
class BufferWriteEvent:
    """Captured registered-buffer write event."""

    address: str
    kind: BufferWriteKind
    producer_label_raw: str | None
    version_label_raw: str | None
    value: Any
    value_changed: bool | None
    object_id: int
    storage_key: tuple[Any, ...] | None
    buffer_version: int | None
    source_func_name: str | None
    seq: int = 0


@dataclass(frozen=True, slots=True)
class ModulePrepEvent:
    """Prep-time module metadata emitted before a forward pass."""

    address: str
    all_addresses: tuple[str, ...]
    module_type_str: str
    cls_qualname: str
    class_name: str
    address_children: tuple[str, ...]
    class_source_file: str | None
    class_source_line: int | None
    init_source_file: str | None
    init_source_line: int | None
    forward_source_file: str | None
    forward_source_line: int | None
    class_docstring: str | None
    init_signature: str | None
    init_docstring: str | None
    forward_signature: str | None
    forward_docstring: str | None
    forward_pre_hooks: object | None
    forward_hooks: object | None
    backward_pre_hooks: object | None
    backward_hooks: object | None
    full_backward_pre_hooks: object | None
    full_backward_hooks: object | None
    training_at_prep: bool
    custom_attributes: tuple[tuple[str, object], ...]
    custom_methods: tuple[str, ...]
    seq: int = 0


@dataclass(frozen=True, slots=True)
class ModuleEnterEvent:
    """Module forward-entry metadata emitted during exhaustive capture."""

    address: str
    call_index: int
    call_label: str
    training: bool
    code_context: tuple[object, ...]
    call_stack: tuple[str, ...]
    forward_start_time: float
    forward_args: object | None
    forward_kwargs: object | None
    forward_args_template: object | None
    forward_kwargs_template: object | None
    layer_argnames: tuple[tuple[str, object], ...]
    input_labels: tuple[str, ...] = ()
    seq: int = 0


@dataclass(frozen=True, slots=True)
class PreHookProvenanceEvent:
    """Typed sidecar for one module invocation's user pre-hook provenance."""

    address: str
    call_index: int | None
    inputs_before_pre_hooks: object | None
    inputs_after_pre_hooks: object | None
    effects: tuple[object, ...]
    capture_complete: bool
    incomplete_reasons: tuple[str, ...]
    seq: int = 0


@dataclass(frozen=True, slots=True)
class ModuleExitEvent:
    """Module forward-exit metadata emitted during exhaustive capture."""

    address: str
    call_index: int
    call_label: str
    forward_duration: float
    output_structure: object | None
    output_tensor_labels_raw: tuple[str, ...]
    per_output_atomic: tuple[tuple[str, tuple[ModuleFrame, ...], bool, tuple[str, int] | None], ...]
    output_names: tuple[str | None, ...] = ()
    # Typed container paths for each captured module output tensor. Optional with
    # an empty-tuple default so the field stays trailing (defaulted) and all
    # consumers guard on a falsy value; absent == "no paths captured".
    output_paths: tuple[tuple[object, ...], ...] = ()
    # TRUE tensor-leaf count of the module's real output object, recorded from
    # the output walk BEFORE labeling/boundary minting can fail. This is the
    # proof the gradient-coverage classifier uses to distinguish "the module
    # genuinely produced no tensor output" (a legitimate exclusion) from
    # "capture failed to attach the output" (a fail-closed gap). ``-1`` means
    # unrecorded (unknown), which consumers treat as unproven.
    output_tensor_leaf_count: int = -1
    seq: int = 0


@dataclass(frozen=True, slots=True)
class OpEvent:
    """Single backend operation event emitted during capture."""

    kind: str
    label_raw: str
    layer_label_raw: str
    layer_type: str
    raw_index: int
    type_index: int
    step_index: int
    source_trace: object | None
    source_trace_id: str | None
    tracing_finished: bool
    construction_done: bool
    function: FunctionCallRef
    output: OutputRef
    templates: ArgTemplateRef | None
    parents: tuple[ParentEdge, ...]
    parent_arg_positions: dict[str, dict[Any, str]]
    _edge_uses: tuple[object, ...]
    params: tuple[ParamRef, ...]
    parent_params: tuple[object, ...]
    module_stack: tuple[ModuleFrame, ...]
    modules: tuple[tuple[str, int], ...]
    backend_semantics: BackendSemantics
    policy: CapturePolicy
    predicate_matched: bool
    pass_index: int
    grad_fn_class_qualname: str | None
    grad_fn_handle: object | None
    equivalence_class: str | None
    is_transform: bool
    transform_kind: str | None
    transform_chain: tuple[str, ...]
    transform_config: dict[str, object]
    transform_fn_name: str | None
    transform_fn_qualname: str | None
    transform_fn_source: object | None
    is_output_parent: bool
    has_internal_source_ancestor: bool
    internal_source_ancestors: frozenset[str]
    input_ancestors: frozenset[str]
    root_ancestors: frozenset[str]
    func_call_id: int | None
    is_bottom_level: bool
    is_scalar_bool: bool | None
    bool_value: bool | None
    intervention_fired: bool
    intervention_replaced: bool
    fire_results: tuple[FireResult, ...]
    intervention_template_ref: InterventionTemplateRef | None
    record_context: object | None = None
    capture_spec: object | None = None
    unattributed_tensor_args: tuple[str, ...] = ()
    dropped_edge_tensor_args: tuple[str, ...] = ()
    input_was_parameter: bool = False
    seq: int = 0


@dataclass(frozen=True, slots=True)
class ConditionalEvent:
    """Captured conditional-control-flow event."""

    conditional_id: int
    record: object
    arm_entry_edges: tuple[tuple[str, str], ...]
    edge_call_indices: tuple[tuple[str, str, int, str, int], ...]


@dataclass(slots=True)
class InterventionState:
    """Final Trace intervention state folded from capture/runtime metadata."""

    has_direct_writes: bool
    spec_revision: int
    out_recipe_revision: int
    append_sequence_id: int
    warned_direct_write: bool
    warned_mutate_in_place: bool
    last_run: object | None


def __getattr__(name: str) -> object:
    """Return compatibility attributes that moved out of the event schema module.

    Parameters
    ----------
    name
        Attribute name requested from :mod:`torchlens.ir.events`.

    Returns
    -------
    object
        Moved compatibility attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not a compatibility export.
    """

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
