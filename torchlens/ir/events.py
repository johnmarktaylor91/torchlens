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
    has_saved_outs: bool
    output_device: str | None
    out_postfunc: object | None
    detach_saved_activations: bool
    visualizer_path: str | None
    multi_output_index: int | None
    is_part_of_iterable_output: bool
    container_path: tuple[object, ...]
    container_spec: ContainerSpec | None
    child_versions: tuple[tuple[str, TensorRef], ...]


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

    buffer_address: str
    name: str
    address: str
    buffer_pass: int
    parent_label_raw: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    memory: int | None
    module_stack: tuple[ModuleFrame, ...]


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
class OpEvent:
    """Single backend operation event emitted during capture."""

    kind: str
    label_raw: str
    layer_label_raw: str
    layer_type: str
    capture_index: int
    type_index: int
    compute_index: int
    source_trace_id: str | None
    tracing_finished: bool
    construction_done: bool
    function: FunctionCallRef
    output: OutputRef
    templates: ArgTemplateRef | None
    parents: tuple[ParentEdge, ...]
    params: tuple[ParamRef, ...]
    module_stack: tuple[ModuleFrame, ...]
    backend_semantics: BackendSemantics
    policy: CapturePolicy
    predicate_matched: bool
    is_bottom_level: bool
    is_scalar_bool: bool | None
    bool_value: bool | None
    intervention_fired: bool
    intervention_replaced: bool
    fire_results: tuple[FireResult, ...]
    intervention_template_ref: InterventionTemplateRef | None
    record_context: object | None = None
    capture_spec: object | None = None


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
    last_run_ctx: object | None


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
    module_events: list[ModuleEvent] = field(default_factory=list)
    conditional_events: list[ConditionalEvent] = field(default_factory=list)
