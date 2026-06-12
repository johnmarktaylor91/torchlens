"""Backend-neutral capture event records for the unified TorchLens pipeline."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .intervention import FireResult, InterventionTemplateRef
    from .refs import ParamRef, TensorRef
    from .semantics import BackendSemantics, CapturePolicy

OpEventKind = Literal["op", "source", "synthetic_output", "intervention_replacement"]
EdgeUseKind = Literal["arg", "kwarg", "container", "module", "buffer", "output"]
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
    seq: int


@dataclass(frozen=True, slots=True)
class BackwardPassEnd:
    """Core event marking completion of one autograd engine invocation."""

    pass_index: int
    duration: float | None
    peak_memory: int | None
    status: BackwardStatus
    order_attribution_coverage: float | None


@dataclass(frozen=True, slots=True)
class GradFnDiscovered:
    """Torch enrichment event for a discovered autograd node object."""

    object_id: int
    class_name: str
    class_qualname: str
    is_custom: bool
    op_label: str | None
    param_ref: object | None
    created_in_pass: int | None
    creator_object_id: int | None
    source: dict[str, object | None]
    topology: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GradFnFired:
    """Torch enrichment event emitted from an autograd node hook."""

    object_id: int
    pass_index: int
    grad_input_refs: object | None
    grad_output_refs: object | None
    intervention_fire_ref: object | None
    timestamp: float
    seq: int


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    """Backend-neutral description of an output container."""

    kind: str
    type_name: str | None
    fields: tuple[str, ...]
    length: int | None
    metadata: tuple[tuple[str, object], ...]


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


@dataclass(frozen=True, slots=True)
class ModuleFrame:
    """Single active module-call frame at capture time."""

    address: str
    address_normalized: str | None
    module_type: str
    call_index: int
    fx_qualpath: str | None
    entry_argnames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BufferEvent:
    """Captured module buffer metadata event."""

    address: str
    name: str
    module_address: str
    buffer_pass: int
    parent_label_raw: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    memory: int | None
    module_stack: tuple[ModuleFrame, ...]


BufferWriteKind = Literal["reassign", "inplace", "fused"]


@dataclass(frozen=True, slots=True)
class BufferWriteEvent:
    """Captured registered-buffer write event."""

    address: str
    kind: BufferWriteKind
    producer_label_raw: str | None
    version_label_raw: str | None
    value: Any
    value_changed: bool
    object_id: int
    storage_key: tuple[Any, ...] | None
    buffer_version: int | None
    source_func_name: str | None


@dataclass(frozen=True, slots=True)
class ModuleEvent:
    """Captured module-call event consumed by postprocessing."""

    address: str
    all_addresses: tuple[str, ...]
    call_index: int
    call_label: str
    layers_raw: tuple[str, ...]
    input_layers_raw: tuple[str, ...]
    output_layers_raw: tuple[str, ...]
    forward_args_summary: object
    forward_kwargs_summary: object
    forward_args: object | None
    forward_kwargs: object | None
    call_parent: str | None
    call_children: tuple[str, ...]


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


@dataclass(frozen=True, slots=True)
class ModuleExitEvent:
    """Module forward-exit metadata emitted during exhaustive capture."""

    address: str
    call_index: int
    call_label: str
    forward_duration: float
    output_structure: object | None
    output_tensor_labels_raw: tuple[str, ...]
    has_user_forward_hooks: bool
    per_output_atomic: tuple[tuple[str, tuple[ModuleFrame, ...], bool, tuple[str, int] | None], ...]
    output_names: tuple[str | None, ...] = ()


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


@dataclass(slots=True)
class TraceBuildState:
    """Transient capture/postprocess state discarded before returning a Trace."""

    raw_layer_dict: dict[str, object] = field(default_factory=OrderedDict)
    raw_layer_labels_list: list[str] = field(default_factory=list)
    mod_entered: dict[int, list[str]] = field(default_factory=dict)
    mod_exited: dict[int, list[str]] = field(default_factory=dict)
    mod_call_index: dict[int, int] = field(default_factory=dict)
    mod_call_labels: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    exhaustive_module_stack: list[object] = field(default_factory=list)
    module_build_data: dict[str, Any] = field(default_factory=dict)
    module_metadata: dict[Any, Any] = field(default_factory=dict)
    module_forward_args: dict[Any, Any] = field(default_factory=dict)
    module_containment_engine: str = "hook_stack"
    current_func_barcode: object | None = None
    grad_fn_strong_refs: list[Any] = field(default_factory=list)
    in_exhaustive_pass: bool = True
    layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    unsaved_layers_lookup_keys: set[str] = field(default_factory=set)
    output_container_specs_by_raw_label: dict[str, ContainerSpec] = field(default_factory=dict)
    output_container_specs: tuple[ContainerSpec, ...] = ()
    input_tensor_addresses: list[int] = field(default_factory=list)
    module_events: list[ModuleEvent] = field(default_factory=list)
    module_prep_events: list[ModulePrepEvent] = field(default_factory=list)
    module_enter_events: list[ModuleEnterEvent] = field(default_factory=list)
    module_exit_events: list[ModuleExitEvent] = field(default_factory=list)
    conditional_events: list[ConditionalEvent] = field(default_factory=list)
