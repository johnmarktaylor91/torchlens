"""Layer: aggregate per-layer metadata (accessors live in ``_layer_accessors.py``).

``OpAccessor``/``LayerAccessor`` were split to ``_layer_accessors.py`` under
the R43 file-size ratchet; both are re-exported here for historical import
sites.

Layer groups one or more Op entries that represent the same
logical layer across recurrent ops.  For non-recurrent models (the
common case), every Layer wraps exactly one Op.

**Delegation pattern**: For single-pass layers, per-pass fields (out,
grad, step_index, etc.) are accessible directly on the Layer
via ``_single_pass_or_error()`` and ``__getattr__`` delegation to ``ops[0]``.
For multi-pass layers, accessing these fields raises ``ValueError`` (NOT
``AttributeError``) directing the user to ``layer_log.ops[N].field``.

Why ValueError instead of AttributeError: Python's property protocol treats
``AttributeError`` from a ``@property`` as "attribute doesn't exist" and falls
through to ``__getattr__``.  Using ``ValueError`` avoids this trap and gives
the user a clear error message.

**_build_layer_logs merge rules** (in ``postprocess/finalization.py``):
When merging multiple ops into one Layer, these aggregate fields are merged:
  - ``has_input_ancestor``: OR across ops
  - ``io_role``: character-level merge of "I", "O", "IO" strings
  - ``is_atomic_module``: OR across ops
  - ``is_in_conditional_body``: OR across ops
  - ``conditional_role_stacks`` / ``conditional_branch_stack_ops``:
    unique per-pass stack signatures and their pass numbers
  - ``conditional_arm_children`` and derived child views:
    pass-stripped ordered unions across ops
All other 78+ fields use the first pass's values only, except the fields
``_reconcile_multipass_layer_fields`` reconciles when passes differ (shapes,
memory, module stacks, and parameters, which become the union over passes).
``output_of_modules`` / ``output_of_module_calls`` are NOT updated across ops
(correct because same-layer grouping requires identical structural position).
"""

import copy
import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from .._constants import LAYER_LOG_FIELD_ORDER
from .._errors import InvalidArgumentError, RecordBindingError
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from .._io.state_keys import refuse_callable_shadowing_state_keys
from .._ir.refs import DtypeRef
from .._quantities import Bytes, Macs
from .._selection.selection import _SelectionOperand
from .._trace_core.op_store import DetachedOpStore
from .._trace_core.relation_views import (
    LAYER_FROZENSET_VIEW_FIELDS,
    LAYER_TUPLE_VIEW_FIELDS,
)
from .._utils._record_group_table import (
    LAYER_MOVED_MEMBERS,
    LAYER_STORAGE_ATTRS,
    OP_MOVED_MEMBERS,
    op_attr,
    op_spelling,
)
from .._utils.facade import refuse_removed_member, removed_set_remedy
from . import _doors, _familiar_names

# LayerAccessor is re-exported (not used here) so every historical
# ``from .layer import ...`` site and pickle qualname keeps resolving.
from ._layer_accessors import LayerAccessor, OpAccessor  # noqa: F401
from ._layer_mirror import LAYER_VIEW_OP_PATHS, _layer_mirror_read, _layer_rep_op, layer_view_read
from ._layer_sites import layer_shape_summary, layer_site_peers
from ._layer_views import (  # noqa: F401  (materialize_layer_mirrors is re-exported)
    _layer_log_to_row,
    layer_child_ops_per_layer,
    layer_children,
    layer_children_per_pass,
    layer_co_parents,
    layer_edges_vary_across_ops,
    layer_get_children,
    layer_get_parents,
    layer_leaf_module_ops,
    layer_parent_arg_positions,
    layer_parent_ops_per_layer,
    layer_parents,
    layer_parents_per_pass,
    layer_siblings,
    layer_total_macs_forward,
    materialize_layer_mirrors,
)
from ._record_group_base import _LAYER_DELEGATED_PASS_FIELDS
from ._record_groups import LayerControlFlow, LayerCost, MovedMemberField
from ._record_groups_autograd import LayerAutograd
from ._record_groups_call import LayerCall
from ._record_groups_graph import LayerGraph
from ._record_groups_module_context import LayerModuleContext
from ._record_groups_payload import LayerPayload
from ._record_groups_source import LayerSource
from ._state_adapter import state_items
from ._value_repr import layer_card, layer_repr_line
from .field_policy import (
    build_record_field_policy_table,
    portable_state_spec_from_policy,
)

if TYPE_CHECKING:
    import pandas as pd

    from ..receptive_field._view import ReceptiveFieldView
    from .op import Op
    from .trace import Trace


#: Public Layer members removed by the do fold (remove-and-rename, no alias).
_REMOVED_LAYER_MEMBERS: dict[str, str] = {
    "attach_hooks": "use layer.do(site_or_hooks, ...) -- do stages the hooks",
    "set": removed_set_remedy("Layer", "layer", "value"),
}

# Typed container defaults for every non-Optional container field Layer stores
# directly (same fix as ``Op._LAYER_PASS_LOG_CONTAINER_DEFAULTS``): plain builtin
# types let ``coerce_container_typed_state`` repair a wrong-typed legacy value,
# and an absent field restores as an empty typed container instead of crashing.
_LAYER_LOG_CONTAINER_DEFAULTS: dict[str, Any] = {
    "arg_names": (),
    "param_shapes": [],
    # Relation view fields: declared restore type is the
    # IMMUTABLE view; ``coerce_container_typed_state`` normalizes legacy
    # list/set state on load so loaded Layers present the same immutable
    # relation surface as live finished captures.
    "_param_barcodes": (),
    "_param_logs": (),
    "equivalent_ops": frozenset(),
    "in_conditionals": (),
    "conditional_role_stacks": (),
    "conditional_branch_stack_ops": {},
    "conditional_arm_children": {},
    "modules": (),
    "output_of_modules": (),
    "output_of_module_calls": (),
    "conditional_entry_children": (),
    "conditional_then_children": (),
    "conditional_elif_children": {},
    "conditional_else_children": (),
    "annotations": {},
    "call_labels": [],
}


# ---------------------------------------------------------------------------
# The aggregate facade.
#
# ``Layer`` no longer copies ~86 representative fields from its first pass at
# build time (the "~78-field per-pass copy" of the pre-columnar era). Each
# mirror field is a class-level data descriptor that reads through to the
# representative op — the FIRST pass in ``self.ops`` — on demand, applying the
# exact normalization ``__init__`` used to apply at copy time (``as_bytes``,
# ``as_flops``, the ``Bytes(... or 0)`` total). Writes land in the instance
# ``__dict__`` as per-layer shadows, so the multi-pass merge passes
# (``_build_layer_logs``, ``_reconcile_multipass_layer_fields``), direct user
# writes, and loaded pickle/.tlspec state behave exactly as the former copies
# did — a shadowed field permanently stops mirroring. Deletes leave the
# ``_LAYER_DELETED`` tombstone so a deleted field stays deleted (state
# enumeration skips it) while plain attribute reads fall through to the
# historical ``__getattr__`` delegation, byte-identical to the dict-era
# post-delete behavior.
# ---------------------------------------------------------------------------

#: Shadow-miss sentinel local to mirror reads.
_LAYER_UNSET = object()

#: Tombstone marking an explicitly deleted mirror field.
_LAYER_DELETED = object()


# Mirror-field spec table and stored-state order: split to _layer_spec.py
# under the R43 file-size ratchet (pure schema data, no behavior).
from ._layer_spec import (  # noqa: E402
    _LAYER_MIRROR_SPEC,
    _LAYER_STATE_ORDER,
    _LAYER_STATE_ORDER_SET,
)


#: Layer stored relation fields and their immutable view types (the same
#: universe the relation freeze converts; ``equivalent_ops`` is handled by
#: its dedicated property but shares the normalization).
def _build_layer_view_types() -> dict[str, type]:
    """Map each Layer stored relation field to its immutable view type."""

    view_types: dict[str, type] = dict.fromkeys(LAYER_TUPLE_VIEW_FIELDS, tuple)
    view_types.update(dict.fromkeys(LAYER_FROZENSET_VIEW_FIELDS, frozenset))
    view_types["equivalent_ops"] = frozenset
    return view_types


_LAYER_RELATION_VIEW_TYPES: dict[str, type] = _build_layer_view_types()


def _layer_relations_finished(layer: "Layer") -> bool:
    """Return whether this Layer's backing capture is finished.

    Mirrors the Op descriptor's finished predicate through the
    representative op's store — sealed, relation-frozen (preview backends
    convert without sealing), or detached (copy/pickle/fork/loaded) — and
    falls back to the owning trace's core op store while the pass accessor
    is not yet populated. Layers with neither (husked after cleanup, bare
    restore shells) are finished by definition.
    """

    store = None
    rep = _layer_rep_op(layer)
    if rep is not None:
        try:
            store = object.__getattribute__(rep, "_core")
        except AttributeError:
            store = None
    if store is None:
        ref = layer.__dict__.get("_source_trace_ref")
        trace = ref() if ref is not None else None
        core = trace.__dict__.get("_trace_core") if trace is not None else None
        store = core.ops if core is not None else None
    if store is None:
        return True
    return bool(
        getattr(store, "frozen", False)
        or getattr(store, "dataflow_edges", None) is not None
        or store.__class__ is DetachedOpStore
    )


def _layer_normalize_relation_write(layer: "Layer", name: str, value: Any) -> Any:
    """Normalize a relation container assigned to a finished Layer.

    Applies exactly the relation freeze's conversion rules (subclass-
    inclusive, unlike the freeze's staging-exact checks): sequences
    normalize to ``tuple`` on tuple-view fields, sets to ``frozenset`` on
    frozenset-view fields. Other value shapes (the dict form of
    ``conditional_branch_stack_ops``, scalars, ``None``) pass through
    unchanged, and building-phase writes stay raw so postprocess aliasing
    is preserved.
    """

    view_type = _LAYER_RELATION_VIEW_TYPES.get(name)
    if view_type is None or value.__class__ is view_type:
        return value
    if view_type is tuple:
        if not isinstance(value, (list, tuple)):
            return value
    elif not isinstance(value, (set, frozenset, list, tuple)):
        return value
    if not _layer_relations_finished(layer):
        return value
    return view_type(value)


class _LayerMirrorField:
    """Data descriptor for one Layer field mirrored from the first-pass op."""

    __slots__ = ("_name", "_source", "_normalize")

    def __init__(self, name: str, source: str, normalize: Any) -> None:
        """Bind the descriptor to its field name, op source, and normalizer."""

        self._name = name
        self._source = source
        self._normalize = normalize

    def __repr__(self) -> str:
        """Return a debugging repr naming the mirrored field."""
        return f"<Layer mirror descriptor {self._name!r}>"

    def __get__(self, layer: Any, objtype: Any = None) -> Any:
        """Return the per-layer shadow when present, else the op mirror."""

        if layer is None:
            return self
        value = layer.__dict__.get(self._name, _LAYER_UNSET)
        if value is not _LAYER_UNSET:
            if value is _LAYER_DELETED:
                raise AttributeError(self._name)
            return value
        rep = _layer_rep_op(layer)
        if rep is None:
            raise AttributeError(self._name)
        value = getattr(rep, self._source)
        normalize = self._normalize
        return normalize(value) if normalize is not None else value

    def __set__(self, layer: Any, value: Any) -> None:
        """Write a per-layer shadow (mirroring permanently stops).

        Relation-view fields normalize to their immutable view on finished
        layers, so a direct assignment can never re-expose a mutable
        relation container (the invariant the op-cell descriptors enforce).
        """

        layer.__dict__[self._name] = _layer_normalize_relation_write(layer, self._name, value)

    def __delete__(self, layer: Any) -> None:
        """Tombstone the field so it stays deleted instead of re-mirroring."""

        instance_dict = layer.__dict__
        if instance_dict.get(self._name, _LAYER_UNSET) is _LAYER_DELETED:
            raise AttributeError(self._name)
        instance_dict[self._name] = _LAYER_DELETED


class Layer(_SelectionOperand):
    """Aggregate per-layer metadata for a logged model operation.

    Groups one or more Op objects (one per invocation of this layer).
    For non-recurrent models, every Layer has exactly one pass.

    Aggregate fields (function identity, param identity, flags, module containment)
    live directly on Layer.  Per-pass fields (outs, graph edges,
    execution state, grads) live on the Op objects in ``self.ops``.

    For single-pass layers, per-pass fields are accessible directly via
    ``__getattr__`` delegation (e.g. ``layer_log.out`` transparently
    reads from ``ops[0].out``).

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> layer = trace.layers["relu_1_2"]
    >>> type(layer).__name__, layer.func_name
    ('Layer', 'relu')
    >>> tuple(layer.out.shape)
    (1, 4, 8, 8)
    """

    if TYPE_CHECKING:
        # The mirror fields are runtime-installed data descriptors
        # (``_install_layer_mirror_descriptors``); declared here so static
        # analysis sees the public surface.
        layer_label: Any
        short_layer_label: Any
        layer_type: Any
        type_index: Any
        step_index: Any
        ordinal_index: Any
        raw_index: Any
        num_passes: Any
        func: Any
        func_name: Any
        func_qualname: Any
        is_inplace: Any
        _autograd_grad_fn_class_name: Any
        _autograd_grad_fn_class_qualname: Any
        _grad_fn_object_id: Any
        _autograd_grad_fn_handle: Any
        _autograd_grad_fn: Any
        _call_arg_names: Any
        _call_num_args_total: Any
        _call_num_pos_args: Any
        _call_num_kwargs: Any
        in_multi_output: Any
        multi_output_index: Any
        multi_output_name: Any
        shape: Any
        _payload_transformed_out_shape: Any
        dtype: Any
        dtype_ref: Any
        _payload_transformed_out_dtype: Any
        device_ref: Any
        backend_address: Any
        resolver_status: Any
        _cost_activation_memory: Any
        _cost_transformed_activation_memory: Any
        _cost_autograd_memory: Any
        _cost_total_autograd_memory: Any
        _autograd_num_autograd_tensors: Any
        _output_device: Any
        visualizer_path: Any
        _activation_transform: Any
        is_edited: Any
        _detach_saved_activations: Any
        save_grads: Any
        _payload_transformed_grad_shape: Any
        _payload_transformed_grad_dtype: Any
        transformed_gradient_memory: Any
        _cost_flops_forward: Any
        _cost_flops_backward: Any
        _param_barcodes: Any
        _param_logs: Any
        param_shapes: Any
        num_params: Any
        num_params_trainable: Any
        num_params_frozen: Any
        _cost_total_param_memory: Any
        _call_config: Any
        _graph_equivalence_class: Any
        is_input: Any
        input_was_parameter: Any
        is_output: Any
        _graph_is_final_output: Any
        _graph_is_buffer: Any
        buffer_address: Any
        _graph_buffer_source: Any
        buffer_write_kind: Any
        buffer_value_changed: Any
        buffer_replay_validated: Any
        _graph_buffer_source_func_name: Any
        _graph_is_internal_source: Any
        _graph_is_internal_sink: Any
        _control_flow_is_terminal_bool: Any
        _control_flow_is_scalar_bool: Any
        _control_flow_bool_value: Any
        module: Any
        modules: Any
        _module_context_output_of_modules: Any
        _module_context_output_of_module_calls: Any
        _control_flow_entry_children: Any
        _control_flow_then_children: Any
        _control_flow_elif_children: Any
        _control_flow_else_children: Any
        _graph_has_input_ancestor: Any
        _graph_io_role: Any
        _graph_buffer_pass: Any
        _module_context_is_atomic_module: Any

    _PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "short_layer_label": FieldPolicy.KEEP,
        "layer_type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "step_index": FieldPolicy.KEEP,
        "ordinal_index": FieldPolicy.KEEP,
        "raw_index": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "source_trace": FieldPolicy.DROP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "func": FieldPolicy.DROP,
        "func_name": FieldPolicy.KEEP,
        "func_qualname": FieldPolicy.KEEP,
        "is_inplace": FieldPolicy.KEEP,
        "grad_fn_class_name": FieldPolicy.KEEP,
        "grad_fn_class_qualname": FieldPolicy.KEEP,
        "grad_fn_object_id": FieldPolicy.KEEP,
        "grad_fn_handle": FieldPolicy.DROP,
        "grad_fn": FieldPolicy.DROP,
        "arg_names": FieldPolicy.KEEP,
        "num_args_total": FieldPolicy.KEEP,
        "num_pos_args": FieldPolicy.KEEP,
        "num_kwargs": FieldPolicy.KEEP,
        "in_multi_output": FieldPolicy.KEEP,
        "multi_output_index": FieldPolicy.KEEP,
        "multi_output_name": FieldPolicy.KEEP,
        "shape": FieldPolicy.KEEP,
        "transformed_out_shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "dtype_ref": FieldPolicy.KEEP,
        "transformed_out_dtype": FieldPolicy.KEEP,
        "device_ref": FieldPolicy.KEEP,
        "backend_address": FieldPolicy.KEEP,
        "resolver_status": FieldPolicy.KEEP,
        "activation_memory": FieldPolicy.KEEP,
        "transformed_activation_memory": FieldPolicy.KEEP,
        "transformed_out": FieldPolicy.DROP,
        "autograd_memory": FieldPolicy.KEEP,
        "total_autograd_memory": FieldPolicy.KEEP,
        "num_autograd_tensors": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "visualizer_path": FieldPolicy.KEEP,
        "activation_transform": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "is_edited": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "save_grads": FieldPolicy.KEEP,
        "transformed_grad": FieldPolicy.DROP,
        "transformed_grad_shape": FieldPolicy.KEEP,
        "transformed_grad_dtype": FieldPolicy.KEEP,
        "transformed_gradient_memory": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "_param_barcodes": FieldPolicy.KEEP,
        "_param_logs": FieldPolicy.KEEP,
        "param_shapes": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "total_param_memory": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "equivalence_class": FieldPolicy.KEEP,
        "equivalent_ops": FieldPolicy.KEEP,
        "is_input": FieldPolicy.KEEP,
        "input_was_parameter": FieldPolicy.KEEP,
        "is_output": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "is_buffer": FieldPolicy.KEEP,
        "address": FieldPolicy.KEEP,
        "buffer_source": FieldPolicy.KEEP,
        "buffer_write_kind": FieldPolicy.KEEP,
        "buffer_value_changed": FieldPolicy.KEEP,
        "buffer_replay_validated": FieldPolicy.KEEP,
        "buffer_source_func_name": FieldPolicy.KEEP,
        "is_internal_source": FieldPolicy.KEEP,
        "is_internal_sink": FieldPolicy.KEEP,
        "is_terminal_bool": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "bool_value": FieldPolicy.KEEP,
        "in_conditionals": FieldPolicy.KEEP,
        "terminal_bool_for": FieldPolicy.KEEP,
        "is_in_conditional_body": FieldPolicy.DROP,
        "conditional_role_stacks": FieldPolicy.KEEP,
        "conditional_branch_stack_ops": FieldPolicy.KEEP,
        "conditional_arm_children": FieldPolicy.KEEP,
        "module": FieldPolicy.KEEP,
        "modules": FieldPolicy.KEEP,
        "output_of_modules": FieldPolicy.KEEP,
        "output_of_module_calls": FieldPolicy.KEEP,
        "conditional_entry_children": FieldPolicy.KEEP,
        "conditional_then_children": FieldPolicy.KEEP,
        "conditional_elif_children": FieldPolicy.KEEP,
        "conditional_else_children": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "is_atomic_module": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
        "call_labels": FieldPolicy.KEEP,
    }
    _FIELD_POLICY = build_record_field_policy_table(
        LAYER_LOG_FIELD_ORDER, _PORTABLE_STATE_SPEC, schema_key="layer"
    )
    _PORTABLE_STATE_SPEC = portable_state_spec_from_policy(_FIELD_POLICY)

    #: Aggregate-facade mirror contract (state-contract service): a pickled
    #: Layer's state carries the Op-shaped mirror fields the facade reads
    #: through to ``ops[0]``, so the known/unknown partition for Layer folds
    #: in the Op record contract. Resolved lazily by module path (the Op
    #: import above is TYPE_CHECKING-only).
    _PORTABLE_STATE_MIRROR_CONTRACT: ClassVar[tuple[str, str]] = (
        "torchlens._data_classes.op",
        "Op",
    )

    def __init__(self, first_pass: "Op") -> None:
        """Initialize the aggregate facade for one layer.

        The facade stores ONLY the genuinely per-layer state (the trace
        back-reference, the aggregate merge containers, and the pass
        accessor); the ~86 representative fields the dict era copied from
        ``first_pass`` are class-level mirror descriptors that read through
        to the first pass in ``self.ops`` on demand (``_LAYER_MIRROR_SPEC``).
        Callers populate ``self.ops`` immediately after construction, exactly
        as before — no mirror field is read until they do.

        Args:
            first_pass: The Op for pass 1 of this layer.
        """
        # Store as weakref to break circular reference (Trace -> layer_logs -> Layer -> Trace).
        _sml = first_pass.source_trace
        self._source_trace_ref: weakref.ReferenceType[Trace] | None = (
            weakref.ref(_sml) if _sml is not None else None
        )
        # Build-time SNAPSHOTS, not mirrors: ``_build_conditional_records``
        # (end of step 15.5) rebinds these two fields on the OPS after the
        # aggregate Layers are built, and the public Layer contract keeps the
        # pre-rebind values (the dict-era copies never saw the update).
        self._control_flow_in_conditionals = first_pass._control_flow_in_conditionals
        self._control_flow_terminal_bool_for = first_pass._control_flow_terminal_bool_for
        # Cached conditional-body predicate (the ``is_in_conditional_body``
        # property's storage slot; multi-pass merge ORs into it).
        self._control_flow_is_in_conditional_body = first_pass._control_flow_is_in_conditional_body
        # Raw ``__dict__`` writes by contract: these are BUILD-PHASE staging
        # containers the multi-pass merge mutates in place, so they must
        # never pass through the finished-layer view normalization (a
        # refresh-built Layer over detached-backed ops would otherwise
        # freeze them at construction).
        self.__dict__["conditional_role_stacks"] = cast("list[list[tuple[int, str]]]", [])
        self.__dict__["conditional_branch_stack_ops"] = cast(
            "dict[tuple[tuple[int, str], ...], list[int]]", {}
        )
        self._control_flow_arm_children: dict[int, dict[str, list[str]]] = {}
        # Genuinely per-layer USER-owned state, never a mirror: the dict era
        # initialized a fresh empty dict here rather than copying from the
        # first pass, so layer annotations must not alias ``ops[0]``'s dict
        # (mirroring leaked op annotations into the layer and vice versa).
        # The ONE exception is the reserved "collective" namespace:
        # a collective boundary layer surfaces its first pass's portable
        # collective_boundary_v1 payload, seeded here as an independent
        # deep copy so neither side can mutate the other. Other op
        # annotation keys (user keys, save_mode/saved_out_version) stay
        # op-only.
        self.annotations: dict[str, Any] = {}
        first_pass_annotations = getattr(first_pass, "annotations", None)
        if isinstance(first_pass_annotations, dict) and "collective" in first_pass_annotations:
            self.annotations["collective"] = copy.deepcopy(first_pass_annotations["collective"])

        # Pass management
        self.ops = OpAccessor()
        self.call_labels: list[str] = []

    @property
    def cost(self) -> LayerCost:
        """Compute, memory and timing facts of this Layer (``layer.cost.total_flops``)."""

        return LayerCost(self)

    @property
    def control_flow(self) -> LayerControlFlow:
        """Conditional and branch facts of this Layer (``layer.control_flow.depth``)."""

        return LayerControlFlow(self)

    @property
    def call(self) -> LayerCall:
        """How this Layer's function was called (``layer.call.config``)."""

        return LayerCall(self)

    @property
    def source(self) -> LayerSource:
        """Where this Layer's call sits in the user's source (``layer.source.fx_path``)."""

        return LayerSource(self)

    @property
    def autograd(self) -> LayerAutograd:
        """Autograd facts of this Layer (``layer.autograd.grad_fn``)."""

        return LayerAutograd(self)

    @property
    def module_context(self) -> LayerModuleContext:
        """Module-nesting facts of this Layer (``layer.module_context.call_depth``)."""

        return LayerModuleContext(self)

    @property
    def graph(self) -> LayerGraph:
        """Graph position of this Layer (``layer.graph.has_parents``)."""

        return LayerGraph(self)

    @property
    def _payload_transformed_out(self) -> Any:
        """Transformed output of the single pass (computed, so not a stored Layer column)."""

        return self._single_pass_or_error("transformed_out")

    @property
    def _payload_transformed_grad(self) -> Any:
        """Transformed gradient of the single pass (computed, so not a stored Layer column)."""

        return self._single_pass_or_error("transformed_grad")

    @property
    def payload(self) -> LayerPayload:
        """Saved transformed payloads of this Layer (``layer.payload.transformed_out``)."""

        return LayerPayload(self)

    @property
    def macs_backward(self) -> Macs | None:
        """Backward MACs are not derivable and always ``None`` (estimates only)."""

        return None

    @property
    def total_gradient_memory(self) -> Bytes:
        """Sum gradient memory across all Ops in this Layer."""

        return Bytes(sum(int(op.gradient_memory or 0) for op in self.ops.values()))

    @property
    def macs_total(self) -> Macs | None:
        """Total MACs are not derivable and always ``None`` (see Op.macs_total)."""

        return None

    @property
    def total_macs_backward(self) -> Macs | None:
        """Backward MACs are not derivable and always ``None`` (estimates only)."""

        return None

    @property
    def param_names(self) -> list[str]:
        """Return short names of parameters used by this Layer."""

        return [param.name for param in self._param_logs]

    @property
    def param_dtypes(self) -> list[Any]:
        """Return dtypes of parameters used by this Layer."""

        return [param.dtype for param in self._param_logs]

    @property
    def has_params(self) -> bool:
        """Whether this layer uses model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors used by this Layer."""

        return sum(1 for param in self._param_logs if param.is_trainable)

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors used by this Layer."""

        return sum(1 for param in self._param_logs if not param.is_trainable)

    @property
    def has_trainable_params(self) -> bool:
        """Whether this Layer uses at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self) -> bool:
        """Whether this Layer uses at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def _graph_is_compute(self) -> bool:
        """Whether this Layer's representative Op is a compute Op."""

        return bool(self.ops and self.ops[0].graph.is_compute)

    @property
    def _graph_is_orphan(self) -> bool:
        """Whether any Op in this Layer is disconnected from the main graph."""

        return any(op.graph.is_orphan for op in self.ops.values())

    @property
    def fx_label(self) -> str | None:
        """Return a torch.fx-style label for a single-Op Layer."""

        return cast(str | None, self._single_pass_or_error("fx_label"))

    @property
    def in_submodule(self) -> bool:
        """Whether this layer was computed inside a submodule."""
        return self.module is not None

    def _intervention_site(self) -> Any:
        """Return the site spelling for this Layer's intervention doors.

        A bare layer label addresses only a single-pass layer; a multi-pass
        Layer passes itself, which lowers to every pass's pass-qualified label
        (the all-passes spelling).
        """

        return self if self.num_passes > 1 else self.layer_label

    @property
    def op_labels(self) -> list[str]:
        """Op labels belonging to this Layer (glossary name for ``call_labels``)."""

        return self.call_labels

    @property
    def is_buffer_source(self) -> bool:
        """Whether this Layer represents a buffer overwrite boundary.

        Glossary name for the stored ``is_buffer`` flag; aggregate over the
        Layer (true when its representative Op is a buffer source).
        """

        return bool(self.graph.is_buffer)

    @property
    def buffer_overwrite_index(self) -> Any:
        """Which overwrite of the buffer this Layer represents.

        Glossary name for the stored ``buffer_pass`` index.
        """

        return self.graph.buffer_pass

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref).

        Never returns ``None``: a Layer detached from its Trace (standalone
        pickle strips the weakref; cleanup clears it) refuses with the same
        typed ``RecordBindingError`` family as the collected-Trace case, so
        no ``None`` can escape behind the ``-> Trace`` signature and crash a
        caller untyped.
        """
        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            raise RecordBindingError(
                "This Layer is not bound to a Trace (standalone pickle, "
                "cleanup, or a record never attached to a Trace)",
                code="record_not_bound",
                remedy="read the layer through a live Trace accessor",
            )
        obj = ref()
        if obj is None:
            raise RecordBindingError(
                "the Trace this Layer was read from has been garbage-collected: a Layer holds its Trace weakly, so a chained read on a temporary (tl.trace(...)[key].out, t.fork().do(...)[key].out) loses it as soon as the record is taken",
                code="trace_reference_collected",
                remedy="keep the Trace (t = tl.trace(...)) or fork (f = t.fork().do(...)) in a variable while reading its records",
            )
        return cast("Trace", obj)

    @source_trace.setter
    def source_trace(self, value: "Trace | None") -> None:
        """Set the owning Trace back-reference.

        Parameters
        ----------
        value:
            Owning model log, or ``None`` to clear the reference.
        """
        self._source_trace_ref = weakref.ref(value) if value is not None else None

    @property
    def _source_trace(self) -> "Trace | None":
        """Owning Trace, if bound and still alive (tolerant internal read)."""

        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            return None
        return cast("Trace | None", ref())

    @property
    def trace(self) -> "Trace":
        """Alias for the owning Trace back-reference."""

        return self.source_trace

    @property
    def _graph_equivalent_ops(self) -> Any:
        """Labels of ops equivalent to this layer.

        On finished traces the value is the group's ONE cached immutable
        ``frozenset`` view (live group views), read through the mirror
        from the representative op and passed through unchanged — alias-safe
        because it cannot be mutated. A raw staging ``set`` (mid-postprocess
        reads, legacy loads before coercion) still hands back a private copy
        so no holder can alias-corrupt the shared group container. A per-layer
        shadow (direct write, loaded state) takes precedence over the mirror,
        exactly like every other mirror field.
        """

        value = self.__dict__.get("equivalent_ops", _LAYER_UNSET)
        if value is _LAYER_DELETED:
            # Fall back to ``__getattr__`` delegation, matching a plain
            # missing attribute.
            raise AttributeError("equivalent_ops")
        if value is _LAYER_UNSET:
            rep = _layer_rep_op(self)
            if rep is None:
                raise AttributeError("equivalent_ops")
            value = rep.graph.equivalent_ops
        return set(value) if value.__class__ is set else value

    @_graph_equivalent_ops.setter
    def _graph_equivalent_ops(self, value: Any) -> None:
        """Shadow ``equivalent_ops`` per layer, normalizing to the immutable view type."""

        self.__dict__["equivalent_ops"] = _layer_normalize_relation_write(
            self, "equivalent_ops", value
        )

    @_graph_equivalent_ops.deleter
    def _graph_equivalent_ops(self) -> None:
        """Tombstone ``equivalent_ops`` so batch removal of finished layers works."""

        # ``state_items`` skips the tombstone, so cleanup ``delattr``s this
        # name; without a deleter the property raises "can't delete
        # attribute", breaking batch removal of finished layers.
        if self.__dict__.get("equivalent_ops", _LAYER_UNSET) is _LAYER_DELETED:
            raise AttributeError("equivalent_ops")
        self.__dict__["equivalent_ops"] = _LAYER_DELETED

    def _layer_state_value(self, name: str) -> Any:
        """Return one declared field's live state value (shadow, then mirror).

        Raises ``AttributeError`` for absent state: a tombstoned field, an
        unreadable mirror, or a stored-only field missing from ``__dict__`` —
        exactly the keys the dict-era ``__dict__`` snapshot omitted.
        """

        value = self.__dict__.get(name, _LAYER_UNSET)
        if value is _LAYER_DELETED:
            raise AttributeError(name)
        if value is not _LAYER_UNSET:
            return value
        if name in _LAYER_MIRROR_SPEC:
            return _layer_mirror_read(self, name)
        if name == "equivalent_ops":
            rep = _layer_rep_op(self)
            if rep is None:
                raise AttributeError(name)
            # The RAW canonical container (finished: the ONE frozen group
            # view; staging: the shared staging set), matching the value the
            # dict era stored — never the property's per-read staging copy.
            return rep.graph.equivalent_ops
        raise AttributeError(name)

    def __tl_state_items__(self) -> Iterator[tuple[str, Any]]:
        """Yield live state ``(field_name, value)`` pairs in declared order.

        Declared fields come first in the exact ``__dict__`` insertion order
        of the dict-era ``__init__`` (the order every pickle golden was frozen
        against); mirror fields resolve through the representative op. Extra
        instance attributes (user-set names) follow in insertion
        order. Tombstoned fields are omitted, matching dict-era deletion.
        """

        instance_dict = self.__dict__
        for name in _LAYER_STATE_ORDER:
            try:
                yield name, self._layer_state_value(name)
            except AttributeError:
                continue
        for name, value in instance_dict.items():
            if name not in _LAYER_STATE_ORDER_SET and value is not _LAYER_DELETED:
                yield name, value

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Install a state mapping as per-layer ``__dict__`` shadows.

        The explicit counterpart of ``__tl_state_items__`` (no record
        class relies on the generic introspection fallback). ``Layer`` is
        dict-backed by design — restored fields become per-layer shadows over
        the mirror descriptors, byte-identical to the dict-era
        ``__dict__.update``.
        """

        self.__dict__.update(mapping)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with weakrefs and raw autograd handles stripped."""
        state = dict(state_items(self))
        state["_source_trace_ref"] = None
        # `grad_fn_handle` holds the live torch autograd `Node` (e.g.
        # `AddmmBackward0`), which is not picklable. `Layer._FIELD_POLICY`
        # declares it `FieldPolicy.DROP` for exactly this reason; enforce that
        # here (mirroring `Op.__getstate__`) so plain `pickle.dumps(trace)` of
        # any model with trainable params does not crash. `grad_fn` (a picklable
        # `GradFn` record after a backward pass) is intentionally retained.
        state["grad_fn_handle"] = None
        # (mirroring ``Op.__getstate__``): user transform callables
        # (FieldPolicy.DROP) serialize to the loaded-artifact form (None) -- a
        # lambda transform= made pickle.dumps crash on rolled layer records
        # while Trace.save succeeded on the same trace.
        if state.get("activation_transform") is not None:
            state["activation_transform"] = None
        if state.get("grad_transform") is not None:
            state["grad_transform"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        read_tlspec_version(state, cls_name=type(self).__name__, cls=type(self))
        resolver_status_was_present = "resolver_status" in state
        layer_setstate_defaults: dict[str, Any] = {
            **_LAYER_LOG_CONTAINER_DEFAULTS,
            "_source_trace_ref": None,
            "annotations": {},
            "autograd_memory": None,
            "total_autograd_memory": None,
            "num_autograd_tensors": None,
            "transformed_out": None,
            "transformed_out_shape": None,
            "transformed_out_dtype": None,
            "dtype_ref": DtypeRef.from_value(state.get("dtype")),
            "device_ref": None,
            "backend_address": None,
            "resolver_status": "resolved",
            "transformed_activation_memory": None,
            "transformed_grad": None,
            "transformed_grad_shape": None,
            "transformed_grad_dtype": None,
            "transformed_gradient_memory": None,
        }
        default_fill_state(state, defaults=layer_setstate_defaults)
        # Repair present-but-wrong-typed container fields from legacy states
        # (e.g. `equivalent_ops` serialized as a `list` where a `set` is now
        # declared). `default_fill_state` only fills absent keys; this closes
        # the same gap `Trace`/`Op` already close for their own fields.
        coerce_container_typed_state(state, layer_setstate_defaults)
        if state.get("dtype_ref") is None:
            state["dtype_ref"] = DtypeRef.from_value(state.get("dtype"))
        if not resolver_status_was_present:
            state["resolver_status"] = "resolved"
        for field_name in (
            "activation_memory",
            "transformed_activation_memory",
            "autograd_memory",
            "total_autograd_memory",
            "transformed_gradient_memory",
            "total_param_memory",
        ):
            if state.get(field_name) is not None:
                state[field_name] = Bytes(state[field_name])
        refuse_callable_shadowing_state_keys(type(self), state)
        self.__dict__.update(state)

    # ********************************************
    # ******* Single-pass delegation *************
    # ********************************************
    # For single-pass layers, per-pass fields are transparently accessible
    # on the Layer itself.  For multi-pass layers, attempting to access
    # these fields raises ValueError directing the user to a specific pass.

    def _single_pass_or_error(self, field_name: str) -> Any:
        """Access a per-pass field, requiring exactly one pass.

        Raises ValueError (not AttributeError) for multi-pass layers.
        Using ValueError avoids the Python property/__getattr__ trap:
        if a @property raises AttributeError, Python silently treats
        the attribute as missing and falls through to __getattr__.
        """
        if self.num_passes > 1:
            spelled, label = op_spelling(field_name), self.layer_label
            raise InvalidArgumentError(
                f"Layer '{label}' has {self.num_passes} ops. Access '{field_name}' on one pass: "
                f"trace['{label}:1'].{spelled} or trace['{label}:last'].{spelled}",
                code="layer_pass_ambiguous",
                remedy=f"access {spelled!r} on one pass via .ops[k]",
                layer_label=self.layer_label,
                field_name=field_name,
            )
        return getattr(self.ops[0], op_attr(field_name))

    def __selection__(self) -> object:
        """Lift this layer's whole output as an ACT selection term (ALL passes)."""
        from .._selection.selection import _selection_from_layer

        return _selection_from_layer(self)

    @property
    def receptive_field(self) -> "ReceptiveFieldView":
        """Return the only pass's receptive field or reject a recurrent layer.

        Raises
        ------
        AmbiguousPassError
            If this logical layer contains multiple executed passes.
        """

        return _doors.receptive_field()._layer_fields.layer_receptive_field(self)

    @property
    def projective_field(self) -> "ReceptiveFieldView":
        """Return the only pass's projective field or reject pass ambiguity."""

        return _doors.receptive_field()._layer_fields.layer_projective_field(self)

    @property
    def _graph_site_peers(self) -> tuple["Layer", ...]:
        """Layers sharing any of this layer's site keys (live index)."""

        return layer_site_peers(self)

    @property
    def shape_summary(self) -> str | None:
        """Across-pass output-shape summary string (derived, not persisted)."""

        return layer_shape_summary(self)

    def _warn_synthesized_read(self, site: str) -> None:
        """Fire the once-per-trace synthesized-value read warning when due.

        The FIRST raw value read on a trace whose
        input values were synthesized warns once per Trace -- the handoff
        case a persisted provenance record alone cannot reach. The inline
        precheck keeps the gold/legacy fast path to two attribute reads;
        only genuinely synthesized traces import the gate machinery.
        """

        ops = getattr(self, "ops", None)
        if not ops:
            return
        try:
            trace = ops[0].trace
        except Exception:  # noqa: BLE001 - detached/cleaned records have no trace
            return
        record = trace._provenance_input_preprocessor
        if (
            record is None
            or getattr(record, "verified", True)
            or getattr(record, "source", "") != "torchlens._quickstart.input_resolver"
        ):
            return
        from .._quickstart._gate import warn_on_raw_read

        warn_on_raw_read(trace, site=site)

    @property
    def out(self) -> Any:
        """Return the saved out for a single-pass layer.

        Returns
        -------
        Any
            Saved out from the only pass.
        """
        self._warn_synthesized_read("Layer.out")
        return self._single_pass_or_error("out")

    @property
    def has_saved_activation(self) -> bool:
        """Return whether the single pass has a saved out.

        Returns
        -------
        bool
            ``True`` when an out was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_saved_activation"))

    @property
    def grad(self) -> Any:
        """Return the saved grad for a single-pass layer.

        The gradient of the layer's output for the latest backward pass, as
        it leaves this trace's edit chain: a ``tl.grad(site)`` edit that fired
        is what this read shows. A site the backward never reached has no
        grad (``has_grad`` is ``False``); TorchLens never fills in zeros.

        Returns
        -------
        Any
            Saved grad from the only pass.
        """
        return self._single_pass_or_error("grad")

    @property
    def has_grad(self) -> bool:
        """Return whether the single pass has a saved grad.

        ``False`` when no backward pass reached the layer's output.

        Returns
        -------
        bool
            ``True`` when a grad was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_grad"))

    @property
    def pass_index(self) -> int:
        """Return the pass number for a single-pass layer.

        Returns
        -------
        int
            Pass number from the only pass.
        """
        return cast(int, self._single_pass_or_error("pass_number"))

    @property
    def lookup_keys(self) -> list[str]:
        """Return lookup keys for a single-pass layer.

        Returns
        -------
        list[str]
            Lookup keys from the only pass.
        """
        return cast(list[str], self._single_pass_or_error("lookup_keys"))

    # ********************************************
    # ***** Aggregate graph properties ***********
    # ********************************************
    # Graph-edge properties compute the union across all ops, returning
    # no-pass labels (i.e. Layer-level identifiers).  This gives a
    # complete picture of which layers are connected across all recurrent
    # iterations.  Order is preserved (first-seen insertion order).

    @property
    def children(self) -> tuple[str, ...]:
        """Union of child layers (no-pass labels) across all ops.

        Immutable view: computed per read, so mutating the
        returned container could never reach stored state anyway; the tuple
        makes that contract explicit and matches the Op relation surface.
        """
        return layer_children(self)

    @property
    def parents(self) -> tuple[str, ...]:
        """Union of parent layers (no-pass labels) across all ops.

        Immutable view; see ``children``.
        """
        return layer_parents(self)

    @property
    def _graph_has_children(self) -> bool:
        """Return whether any pass has child layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph children.
        """
        return any(p.graph.has_children for p in self.ops.values())

    @property
    def _graph_has_parents(self) -> bool:
        """Return whether any pass has parent layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph parents.
        """
        return any(p.graph.has_parents for p in self.ops.values())

    @property
    def siblings(self) -> list[str]:
        """Union of sibling layers (no-pass labels) across all ops.

        Every pass spelling of THIS layer is excluded: another pass of the same
        layer is a per-op sibling, but at the aggregate level it folds back to
        this layer's own no-pass label, and a layer is never its own sibling.
        """
        return layer_siblings(self)

    @property
    def _graph_has_siblings(self) -> bool:
        """Return whether this layer has any sibling layers besides itself.

        Returns
        -------
        bool
            ``True`` when the aggregate ``siblings`` view is non-empty.
        """
        return len(self.siblings) > 0  # record-groups: flat

    @property
    def co_parents(self) -> list[str]:
        """Union of spouse layers (no-pass labels) across all ops.

        Every pass spelling of THIS layer is excluded, mirroring ``siblings``:
        a layer is never its own co-parent at the aggregate level.
        """
        return layer_co_parents(self)

    @property
    def _graph_has_co_parents(self) -> bool:
        """Return whether this layer has any co-parent layers besides itself.

        Returns
        -------
        bool
            ``True`` when the aggregate ``co_parents`` view is non-empty.
        """
        return len(self.co_parents) > 0  # record-groups: flat

    @property
    def _control_flow_is_in_conditional_body(self) -> bool:
        """Whether this layer is in a conditional arm body."""

        return bool(self.__dict__.get("_is_in_conditional_body", False)) or any(
            role.role == "body" for role in self.control_flow.in_conditionals or []
        )

    @_control_flow_is_in_conditional_body.setter
    def _control_flow_is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during aggregation."""

        self.__dict__["_is_in_conditional_body"] = value

    @_control_flow_is_in_conditional_body.deleter
    def _control_flow_is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        self.__dict__.pop("_is_in_conditional_body", None)

    @property
    def _tracing_finished(self) -> bool:
        """Return whether the owning trace has finished capture/postprocess.

        Reads the weakref slot directly: a detached or collected Layer is
        finished by definition, and this predicate must stay readable where
        ``source_trace`` refuses typed (repr on detached records).
        """

        ref = self.__dict__.get("_source_trace_ref")
        sml = ref() if ref is not None else None
        if sml is None:
            return True
        return sml._tracing_finished

    # ********************************************
    # ****** Convenience properties **************
    # ********************************************

    @property
    def label(self) -> str:
        """For single-pass layers, return the pass-qualified label."""
        return cast(str, self._single_pass_or_error("label"))

    @property
    def short_label(self) -> str:
        """For single-pass layers, return the short pass-qualified label."""
        return cast(str, self._single_pass_or_error("short_label"))

    @property
    def params(self) -> Any:
        """Access parameter metadata by address, short name, or index."""
        from .param import ParamAccessor

        param_dict = {pl.address: pl for pl in self._param_logs}
        return ParamAccessor(param_dict)

    # ********************************************
    # **** Rolled-vis computed properties ********
    # ********************************************
    # These provide per-pass edge tracking for rolled (recurrence-aware)
    # graph visualization.  Computed on-the-fly from the ops dict.
    # Used by the visualization renderer to draw pass-annotated edges.

    @property
    def _graph_children_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: child layer labels (no-pass) for each pass."""
        return layer_children_per_pass(self)

    @property
    def _graph_parents_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: parent layer labels (no-pass) for each pass."""
        return layer_parents_per_pass(self)

    @property
    def _graph_child_ops_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each child layer, which ops connect to it."""
        return layer_child_ops_per_layer(self)

    @property
    def _graph_parent_ops_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each parent layer, which ops connect from it."""
        return layer_parent_ops_per_layer(self)

    @property
    def edges_vary_across_ops(self) -> bool:
        """Whether graph edges differ across ops."""
        return layer_edges_vary_across_ops(self)

    @property
    def leaf_module_ops(self) -> set[Any]:
        """Set of module ops exited across all ops."""
        return layer_leaf_module_ops(self)

    # ********************************************
    # ******* Fallback __getattr__ ***************
    # ********************************************

    def __getattr__(self, name: str) -> Any:
        """Fallback lookup: a removed member redirects typed, else delegate to ops[0].

        Runs only after normal lookup failed. Single-pass layers forward to the
        underlying Op (``layer_log.func_rng_states`` needs no property); private
        names never delegate, so ``self.__dict__`` reads cannot recurse.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        # Before delegating: FacadeTeachingError is an AttributeError, so an Op
        # redirect caught below would otherwise fall through to a bare miss.
        refuse_removed_member("Layer", name, _REMOVED_LAYER_MEMBERS)
        refuse_removed_member(type(self).__name__, name, LAYER_MOVED_MEMBERS)
        if name == "param_memory":
            raise AttributeError(name)
        ops = self.__dict__.get("ops")
        if ops and name in LAYER_VIEW_OP_PATHS:  # the Op group path, never the Op refusal below
            return layer_view_read(self, name)
        # A former flat Op spelling refuses here, typed, instead of being lost to
        # the plain miss below when the delegated Op read refuses it.
        refuse_removed_member(type(self).__name__, name, OP_MOVED_MEMBERS)
        if ops and len(ops) == 1:
            try:
                return getattr(ops[0], name)
            except AttributeError:
                pass
        if ops and len(ops) > 1 and name in _LAYER_DELEGATED_PASS_FIELDS:
            return self._single_pass_or_error(name)
        raise _familiar_names.missing_attribute_error(type(self).__name__, name)

    # ********************************************
    # ************ User-facing custom_methods ***********
    # ********************************************

    def get_children(self) -> list["Layer"]:
        """Return child Layer objects for this layer.

        Returns
        -------
        list[Layer]
            Child layers resolved through the owning model log; orphan
            relation labels resolve through ``trace.graph.orphans`` (unresolvable
            skipped).
        """
        return layer_get_children(self)

    def get_parents(self) -> list["Layer"]:
        """Return parent Layer objects for this layer.

        Returns
        -------
        list[Layer]
            Parent layers resolved through the owning model log; orphan
            relation labels resolve through ``trace.graph.orphans`` (unresolvable
            skipped).
        """
        return layer_get_parents(self)

    def show(
        self,
        kind: Literal["auto", "heatmap", "channels", "rgb", "hist"] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Display this layer's saved out.

        Parameters
        ----------
        kind:
            Picture kind. ``None`` (the default) or ``"auto"`` chooses from tensor shape.
        **kwargs:
            Forwarded to the tensor display helper.

        Returns
        -------
        Any
            Matplotlib figure when plotting is available, otherwise a text
            fallback explaining why no plot was produced.
        """

        from ..viz._tensor_display import show_tensor

        return show_tensor(self, method=kind or "auto", **kwargs)

    def _source_trace_or_error(self) -> "Trace":
        """Return the owning Trace, or raise a detached-log error.

        Returns
        -------
        Trace
            Source Trace that owns this layer log.

        Raises
        ------
        AttributeError
            If this layer log is detached from its source Trace.
        """

        ref = self.__dict__.get("_source_trace_ref")
        source = ref() if ref is not None else None
        if source is None or getattr(source, "_loaded_from_bundle", False):
            raise AttributeError(
                "This Layer is detached from its source Trace "
                "(perhaps loaded from disk or after cleanup). "
                "Use trace.do(label, transform) directly."
            )
        return cast("Trace", source)

    def do(
        self,
        edit: Any,
        /,
        *,
        model: Any = None,
        inputs: Any = None,
        intervention: Any = None,
    ) -> "Trace":
        """Apply an intervention to this layer through the owning Trace.

        Parameters
        ----------
        edit:
            Value, hook or helper applied to this layer's output, as in ``Trace.do``.
        model, inputs:
            Required when ``intervention.engine="rerun"``.
        intervention:
            Grouped intervention options (``InterventionOptions``).

        Returns
        -------
        Trace
            Source Trace after applying the intervention.
        """

        return self._source_trace_or_error().do(
            self._intervention_site(), edit, model=model, inputs=inputs, intervention=intervention
        )

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Layer as a one-row pandas DataFrame.

        Per-pass fields (``_MULTI_PASS_PER_CALL_LAYER_FIELDS``, e.g.
        ``transformed_out``/``transformed_grad``) are reported as ``None``
        for multi-pass (recurrent) Layers instead of raising -- access them
        per-pass via ``layer.ops[i].transformed_out`` instead.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``LAYER_LOG_FIELD_ORDER``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e
        row = _layer_log_to_row(self)
        return pd.DataFrame([row], columns=LAYER_LOG_FIELD_ORDER)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        """Render a bounded Layer display from captured metadata and cached values."""

        return layer_card(self)

    def __repr__(self) -> str:
        """Render a bounded Layer display from captured metadata and cached values."""

        return layer_repr_line(self)

    def _detached_from_trace(self) -> bool:
        """Whether this Layer lost its owning Trace (R52-A placeholder gate).

        A collected or standalone-pickled Layer keeps readable shadow
        fields, but its graph context is gone -- the repr contract renders
        the explicit placeholder rather than a context-free stats line.
        """

        ref = self.__dict__.get("_source_trace_ref")
        return ref is None or ref() is None

    def _repr_html_(self) -> str:
        """Return the pass-aware notebook Layer card.

        Per-pass fields render via ``ops[k]`` rows (pass rows are the
        card's SHAPE); a single-pass layer degenerates to the Op card
        through the one shared code path. Never raises: any internal
        failure degrades to a one-line ``card unavailable`` fragment, and
        the treescope-bridge suppression sentinel replaces the card
        exactly when the bridge just rendered this object.
        """

        from ..report._notebook.cards import layer_repr_html

        return layer_repr_html(self)

    def __len__(self) -> int:
        """Return the number of operation passes aggregated into this layer."""

        return cast(int, self.num_passes)


def _install_layer_mirror_descriptors() -> None:
    """Install the per-field mirror descriptors on the ``Layer`` class.

    Refuses to overwrite an existing class attribute: a mirror name colliding
    with a hand-written ``@property`` (or method) would silently change public
    behavior, so the collision fails at import time instead.
    """

    for name, (source, normalize) in _LAYER_MIRROR_SPEC.items():
        attr = LAYER_STORAGE_ATTRS.get(name, name)
        if attr in vars(Layer):
            raise RuntimeError(
                f"Layer mirror field {attr!r} collides with an existing class attribute"
            )
        setattr(Layer, attr, _LayerMirrorField(name, source, normalize))


class _LayerViewField:
    """Data descriptor for one plain-stored Layer relation-view field.

    The relation-view fields NOT in the mirror spec are ordinary instance
    attributes; this descriptor preserves their plain ``__dict__`` storage
    and read/delete semantics while routing writes through the finished-
    layer view normalization, so no assignment path can re-expose a mutable
    relation container on a finished Layer.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        """Bind the descriptor to its field name."""

        self._name = name

    def __repr__(self) -> str:
        """Return a debugging repr naming the stored field."""

        return f"<Layer view descriptor {self._name!r}>"

    def __get__(self, layer: Any, objtype: Any = None) -> Any:
        """Return the stored value; missing fields raise ``AttributeError``."""

        if layer is None:
            return self
        value = layer.__dict__.get(self._name, _LAYER_UNSET)
        if value is _LAYER_UNSET:
            raise AttributeError(self._name)
        return value

    def __set__(self, layer: Any, value: Any) -> None:
        """Store the value, view-normalized on finished layers."""

        layer.__dict__[self._name] = _layer_normalize_relation_write(layer, self._name, value)

    def __delete__(self, layer: Any) -> None:
        """Delete the stored value; a missing field raises ``AttributeError``."""

        try:
            del layer.__dict__[self._name]
        except KeyError:
            raise AttributeError(self._name) from None


def _install_layer_view_descriptors() -> None:
    """Install plain view descriptors for the non-mirror relation fields."""

    for name in _LAYER_RELATION_VIEW_TYPES:
        if name == "equivalent_ops" or name in _LAYER_MIRROR_SPEC:
            continue
        attr = LAYER_STORAGE_ATTRS.get(name, name)
        if attr in vars(Layer):
            raise RuntimeError(
                f"Layer view field {attr!r} collides with an existing class attribute"
            )
        setattr(Layer, attr, _LayerViewField(name))


def _install_layer_moved_member_refusals() -> None:
    """Refuse every former flat Layer spelling typed, over any ``__dict__`` shadow.

    A restored or materialized shadow keeps its persisted key in
    ``__dict__``; the data descriptor wins over it, so the former spelling
    still names its replacement instead of reading the shadow.
    """

    stored = frozenset(LAYER_LOG_FIELD_ORDER)
    for name, remedy in LAYER_MOVED_MEMBERS.items():
        attr = LAYER_STORAGE_ATTRS[name]
        if name in stored and attr not in vars(Layer):
            # A plain ``__dict__`` field: its storage attribute reads the persisted key.
            setattr(Layer, attr, _LayerViewField(name))
        if name not in vars(Layer):
            setattr(Layer, name, MovedMemberField("Layer", name, remedy))


_install_layer_mirror_descriptors()
_install_layer_view_descriptors()
_install_layer_moved_member_refusals()
