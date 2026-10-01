"""Layer and LayerAccessor: aggregate per-layer metadata and dict-like accessor.

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
All other 78+ fields use the first pass's values only.
``output_of_modules`` / ``output_of_module_calls`` are NOT updated across ops
(correct because same-layer grouping requires identical structural position).
"""

import copy
import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from .._errors import AmbiguousOpLookupError, InvalidArgumentError, RecordBindingError
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from ..constants import LAYER_LOG_FIELD_ORDER, LAYER_PASS_LOG_FIELD_ORDER
from ..ir.refs import DtypeRef
from ..quantities import Bytes, Duration, Flops, Macs, as_macs
from ..selection import _SelectionOperand
from ._accessor_base import Accessor
from ._repr import format_config_items, format_shape_list
from .field_policy import build_record_field_policy_table, portable_state_spec_from_policy

if TYPE_CHECKING:
    import pandas as pd

    from ..receptive_field._view import ReceptiveFieldView
    from .op import Op
    from .trace import Trace


_LAYER_DELEGATED_PASS_FIELDS = frozenset((*LAYER_PASS_LOG_FIELD_ORDER, "out_ref", "grad_ref"))

# Per-pass fields that ``Layer`` resolves via ``_single_pass_or_error``: they
# describe ONE op/pass and deliberately raise ``ValueError`` on a multi-pass
# (recurrent) Layer, directing the user to a specific ``layer.ops[i]`` pass.
# The field-order-driven ``to_pandas`` row builders cannot let that raise, so
# they report these as None for multi-pass Layers -- mirroring
# ``_MULTI_CALL_PER_PASS_MODULE_FIELDS`` one level up in module.py. No
# information is lost: the per-pass value remains reachable via
# ``layer.ops[i].transformed_out`` / ``.transformed_grad``.
_MULTI_PASS_PER_CALL_LAYER_FIELDS: frozenset[str] = frozenset(
    {
        "transformed_out",
        "transformed_grad",
    }
)

# Typed container defaults for every non-Optional container field Layer
# stores directly (as opposed to delegating to ``self.ops[i]``). Same defect
# class as ``Op._LAYER_PASS_LOG_CONTAINER_DEFAULTS`` /
# ``Trace._MODEL_LOG_CONTAINER_DEFAULTS``: without this,
# ``coerce_container_typed_state`` cannot repair a present-but-wrong-typed
# legacy value (e.g. ``equivalent_ops`` serialized as a ``list`` where a
# ``set`` is now declared), and an absent field crashes instead of restoring
# an empty typed container. Plain builtin types are used deliberately.
_LAYER_LOG_CONTAINER_DEFAULTS: dict[str, Any] = {
    "arg_names": (),
    "param_shapes": [],
    # Relation view fields (M6, JMT-FORK-1): declared restore type is the
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
# The M8 aggregate facade.
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

    from .._trace_core.relation_views import (
        LAYER_FROZENSET_VIEW_FIELDS,
        LAYER_TUPLE_VIEW_FIELDS,
    )

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
    from .._trace_core.op_store import DetachedOpStore

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


def _layer_rep_op(layer: "Layer") -> "Op | None":
    """Return the representative (first-pass) op backing one Layer's mirrors.

    Reads raw ``__dict__`` storage so descriptor bodies never re-enter the
    attribute protocol. ``OpAccessor._list`` is pass-index sorted, so index 0
    is the first pass — the op the former ``__init__`` copied from.
    """

    ops = layer.__dict__.get("ops")
    if ops is None:
        return None
    item_list = ops.__dict__.get("_list")
    return item_list[0] if item_list else None


def _layer_mirror_read(layer: "Layer", name: str) -> Any:
    """Return the mirror value for ``name`` (no shadow consulted).

    Raises ``AttributeError`` when the layer has no representative op yet or
    the op's source cell is unset — the caller (descriptor or state
    enumeration) translates that into the historical missing-field behavior.
    """

    rep = _layer_rep_op(layer)
    if rep is None:
        raise AttributeError(name)
    source, normalize = _LAYER_MIRROR_SPEC[name]
    value = getattr(rep, source)
    return normalize(value) if normalize is not None else value


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


def materialize_layer_mirrors(layer_log: "Layer") -> None:
    """Materialize every unmaterialized mirror field into ``__dict__``.

    Called before the backing ops are husked (``Trace.cleanup()``, log-entry
    removal): a user-held Layer keeps exactly the readable state the dict-era
    copies would have kept. Tombstoned and already-shadowed fields are left
    untouched; unreadable mirrors (an already-scrubbed op cell) stay absent,
    matching a field the dict era had already deleted.
    """

    instance_dict = layer_log.__dict__
    for field_name in _LAYER_MIRROR_SPEC:
        if field_name in instance_dict:
            continue
        try:
            instance_dict[field_name] = _layer_mirror_read(layer_log, field_name)
        except AttributeError:
            continue
    if "equivalent_ops" not in instance_dict:
        rep = _layer_rep_op(layer_log)
        if rep is not None:
            try:
                instance_dict["equivalent_ops"] = rep.equivalent_ops
            except AttributeError:
                pass


def _layer_log_to_row(layer_log: "Layer") -> dict[str, Any]:
    """Convert a Layer into one DataFrame row.

    Parameters
    ----------
    layer_log:
        Layer metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value, ordered by
        ``LAYER_LOG_FIELD_ORDER``. Per-pass fields
        (``_MULTI_PASS_PER_CALL_LAYER_FIELDS``) are reported as ``None`` for
        multi-pass Layers so the table never raises.
    """

    multi_pass = layer_log.num_passes > 1
    row: dict[str, Any] = {}
    for field_name in LAYER_LOG_FIELD_ORDER:
        if multi_pass and field_name in _MULTI_PASS_PER_CALL_LAYER_FIELDS:
            row[field_name] = None
            continue
        row[field_name] = getattr(layer_log, field_name)
    return row


class OpAccessor(Accessor["Op"]):
    """Scoped dict-like accessor for the Op entries owned by one Layer."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(self, ops: dict[int, "Op"] | None = None) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        ops:
            Mapping from 1-based pass index to Op.
        """

        ops = ops or {}
        super().__init__(ops, item_list=[op for _, op in sorted(ops.items())])

    def __getitem__(self, key: int | str) -> "Op":
        """Return an Op by 0-based position or pass-qualified label."""

        if isinstance(key, int):
            return self._list[key]
        resolved = self._resolve_substring(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"Op '{key}' not found in scoped Layer ops.")

    def __setitem__(self, key: int, value: "Op") -> None:
        """Set an Op by 1-based pass index."""

        self._dict[key] = value
        self._list = [op for _, op in sorted(self._dict.items())]

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to an Op."""

        if isinstance(key, int):
            return -len(self._list) <= key < len(self._list)
        if isinstance(key, str):
            try:
                self[key]
            except (KeyError, ValueError):
                return False
            return True
        return False

    # This scoped accessor intentionally iterates call-index keys, unlike the generic
    # Trace-level accessors that iterate log values.
    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        """Iterate call-index keys."""

        return iter(self._dict)

    def get(self, key: int, default: "Op | None" = None) -> "Op | None":
        """Return an Op by call index, or default."""

        return self._dict.get(key, default)

    def _resolve_substring(self, key: str) -> "Op | None":
        """Resolve Op by any scoped layer-label variant."""
        if len(self._dict) == 1:
            only_op = next(iter(self._dict.values()))
            if key in {
                only_op.layer_label,
                only_op.layer_label_short,
                only_op._label_raw,
                only_op.raw_label,
            }:
                return only_op
        parent_matches = [
            op_log
            for op_log in self._dict.values()
            if key in {op_log.layer_label, op_log.layer_label_short}
        ]
        if len(parent_matches) > 1:
            parent_label = parent_matches[0].layer_label
            qualified = ", ".join(op_log.label for op_log in parent_matches[:10])
            suffix = "..." if len(parent_matches) > 10 else ""
            raise AmbiguousOpLookupError(
                f"Layer '{parent_label}' has {len(parent_matches)} ops. Use a 0-based "
                "integer position or a pass-qualified label like "
                f"'{parent_label}:1'. Available Op labels: {qualified}{suffix}."
            )
        for op_log in self._dict.values():
            if key in {
                op_log.label,
                op_log.label_short,
                op_log._label_raw,
                op_log.raw_label,
            }:
                return op_log
        return None


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
    """

    if TYPE_CHECKING:
        # The M8 mirror fields are runtime-installed data descriptors
        # (``_install_layer_mirror_descriptors``); declared here so static
        # analysis sees the public surface.
        layer_label: Any
        layer_label_short: Any
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
        grad_fn_class_name: Any
        grad_fn_class_qualname: Any
        grad_fn_object_id: Any
        grad_fn_handle: Any
        grad_fn: Any
        arg_names: Any
        num_args_total: Any
        num_pos_args: Any
        num_kwargs: Any
        in_multi_output: Any
        multi_output_index: Any
        multi_output_name: Any
        shape: Any
        transformed_out_shape: Any
        dtype: Any
        dtype_ref: Any
        transformed_out_dtype: Any
        device_ref: Any
        backend_address: Any
        resolver_status: Any
        activation_memory: Any
        transformed_activation_memory: Any
        autograd_memory: Any
        total_autograd_memory: Any
        num_autograd_tensors: Any
        output_device: Any
        visualizer_path: Any
        activation_transform: Any
        intervention_replaced: Any
        detach_saved_activations: Any
        save_grads: Any
        transformed_grad_shape: Any
        transformed_grad_dtype: Any
        transformed_gradient_memory: Any
        flops_forward: Any
        flops_backward: Any
        _param_barcodes: Any
        _param_logs: Any
        param_shapes: Any
        num_params: Any
        num_params_trainable: Any
        num_params_frozen: Any
        total_param_memory: Any
        func_config: Any
        equivalence_class: Any
        is_input: Any
        input_was_parameter: Any
        is_output: Any
        is_final_output: Any
        is_buffer: Any
        address: Any
        buffer_source: Any
        buffer_write_kind: Any
        buffer_value_changed: Any
        buffer_replay_validated: Any
        buffer_source_func_name: Any
        is_internal_source: Any
        is_internal_sink: Any
        is_terminal_bool: Any
        is_scalar_bool: Any
        bool_value: Any
        module: Any
        modules: Any
        output_of_modules: Any
        output_of_module_calls: Any
        conditional_entry_children: Any
        conditional_then_children: Any
        conditional_elif_children: Any
        conditional_else_children: Any
        has_input_ancestor: Any
        io_role: Any
        buffer_pass: Any
        is_atomic_module: Any

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
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
        "intervention_replaced": FieldPolicy.KEEP,
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
    FIELD_POLICY = build_record_field_policy_table(
        LAYER_LOG_FIELD_ORDER, PORTABLE_STATE_SPEC, schema_key="layer"
    )
    PORTABLE_STATE_SPEC = portable_state_spec_from_policy(FIELD_POLICY)

    def __init__(self, first_pass: "Op") -> None:
        """Initialize the aggregate facade for one layer.

        The M8 facade stores ONLY the genuinely per-layer state (the trace
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
        self.in_conditionals = first_pass.in_conditionals
        self.terminal_bool_for = first_pass.terminal_bool_for
        # Cached conditional-body predicate (the ``is_in_conditional_body``
        # property's storage slot; multi-pass merge ORs into it).
        self.is_in_conditional_body = first_pass.is_in_conditional_body
        # Raw ``__dict__`` writes by contract: these are BUILD-PHASE staging
        # containers the multi-pass merge mutates in place, so they must
        # never pass through the finished-layer view normalization (a
        # refresh-built Layer over detached-backed ops would otherwise
        # freeze them at construction).
        self.__dict__["conditional_role_stacks"] = cast("list[list[tuple[int, str]]]", [])
        self.__dict__["conditional_branch_stack_ops"] = cast(
            "dict[tuple[tuple[int, str], ...], list[int]]", {}
        )
        self.conditional_arm_children: dict[int, dict[str, list[str]]] = {}
        # Genuinely per-layer USER-owned state, never a mirror: the dict era
        # initialized a fresh empty dict here rather than copying from the
        # first pass, so layer annotations must not alias ``ops[0]``'s dict
        # (mirroring leaked op annotations into the layer and vice versa).
        # The ONE exception is the reserved "collective" namespace (R18-7):
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
    def macs_forward(self) -> Macs | None:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_forward // 2 if self.flops_forward is not None else None)

    @property
    def macs_backward(self) -> Macs | None:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_backward // 2 if self.flops_backward is not None else None)

    @property
    def total_activation_memory(self) -> Bytes:
        """Sum activation memory across all Ops in this Layer."""

        return Bytes(sum(int(op.activation_memory or 0) for op in self.ops.values()))

    @property
    def total_gradient_memory(self) -> Bytes:
        """Sum gradient memory across all Ops in this Layer."""

        return Bytes(sum(int(op.gradient_memory or 0) for op in self.ops.values()))

    @property
    def flops_total(self) -> Flops:
        """Representative total FLOPs for this Layer."""

        return Flops((self.flops_forward or 0) + (self.flops_backward or 0))

    @property
    def total_flops_forward(self) -> Flops:
        """Sum forward FLOPs across all Ops in this Layer."""

        return Flops(sum((op.flops_forward or 0) for op in self.ops.values()))

    @property
    def total_flops_backward(self) -> Flops:
        """Sum backward FLOPs across all Ops in this Layer."""

        return Flops(sum((op.flops_backward or 0) for op in self.ops.values()))

    @property
    def total_flops_total(self) -> Flops:
        """Sum total FLOPs across all Ops in this Layer."""

        return Flops(self.total_flops_forward + self.total_flops_backward)

    @property
    def macs_total(self) -> Macs:
        """Representative total MACs for this Layer."""

        return Macs(self.flops_total // 2)

    @property
    def total_macs_forward(self) -> Macs:
        """Sum forward MACs across all Ops in this Layer."""

        return Macs(self.total_flops_forward // 2)

    @property
    def total_macs_backward(self) -> Macs:
        """Sum backward MACs across all Ops in this Layer."""

        return Macs(self.total_flops_backward // 2)

    @property
    def total_macs_total(self) -> Macs:
        """Sum total MACs across all Ops in this Layer."""

        return Macs(self.total_flops_total // 2)

    @property
    def param_names(self) -> list[str]:
        """Return short names of parameters used by this Layer."""

        return [param.name for param in self._param_logs]

    @property
    def param_dtypes(self) -> list[Any]:
        """Return dtypes of parameters used by this Layer."""

        return [param.dtype for param in self._param_logs]

    @property
    def uses_params(self) -> bool:
        """Whether this layer uses model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this layer."""
        return len(self._param_barcodes)

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
    def is_compute_layer(self) -> bool:
        """Whether this Layer's representative Op is a compute Op."""

        return bool(self.ops and self.ops[0].is_compute_op)

    @property
    def is_orphan(self) -> bool:
        """Whether any Op in this Layer is disconnected from the main graph."""

        return any(op.is_orphan for op in self.ops.values())

    @property
    def num_ops(self) -> int:
        """Number of Ops aggregated by this Layer."""

        return len(self.ops)

    @property
    def fx_label(self) -> str | None:
        """Return a torch.fx-style label for a single-Op Layer."""

        return cast(str | None, self._single_pass_or_error("fx_label"))

    @property
    def fx_qualpath(self) -> str | None:
        """Return the FX-style qualified path for a single-Op Layer."""

        return cast(str | None, self._single_pass_or_error("fx_qualpath"))

    @property
    def fx_call_index(self) -> int:
        """Return the FX-style call index for a single-Op Layer."""

        return cast(int, self._single_pass_or_error("fx_call_index"))

    @property
    def in_submodule(self) -> bool:
        """Whether this layer was computed inside a submodule."""
        return self.module is not None

    @property
    def module_call_depth(self) -> int:
        """Depth of module nesting for this layer."""
        return len(self.modules)

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

        return bool(self.is_buffer)

    @property
    def buffer_overwrite_index(self) -> Any:
        """Which overwrite of the buffer this Layer represents.

        Glossary name for the stored ``buffer_pass`` index.
        """

        return self.buffer_pass

    @property
    def is_module_input(self) -> bool:
        """Whether this Layer's representative Op feeds into at least one ModuleCall.

        Delegates from ``Op.is_module_input`` semantics for single-Op Layers.
        Raises ``ValueError`` for multi-Op Layers.
        """

        return cast(bool, self._single_pass_or_error("is_module_input"))

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref).

        Never returns ``None``: a Layer detached from its Trace (standalone
        pickle strips the weakref; cleanup clears it) refuses with the same
        typed ``RecordBindingError`` family as the collected-Trace case, so
        no ``None`` can escape behind the ``-> Trace`` signature and crash a
        caller untyped (r4 b7-opus R52-A).
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
                "Trace has been garbage-collected",
                code="trace_reference_collected",
                remedy="keep the owning Trace alive while reading its records",
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
    def equivalent_ops(self) -> Any:
        """Labels of ops equivalent to this layer.

        On finished traces the value is the group's ONE cached immutable
        ``frozenset`` view (M7 live group views), read through the M8 mirror
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
            value = rep.equivalent_ops
        return set(value) if value.__class__ is set else value

    @equivalent_ops.setter
    def equivalent_ops(self, value: Any) -> None:
        """Shadow ``equivalent_ops`` per layer, normalizing to the immutable view type."""

        self.__dict__["equivalent_ops"] = _layer_normalize_relation_write(
            self, "equivalent_ops", value
        )

    @equivalent_ops.deleter
    def equivalent_ops(self) -> None:
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
            return rep.equivalent_ops
        raise AttributeError(name)

    def __tl_state_items__(self) -> Iterator[tuple[str, Any]]:
        """Yield live state ``(field_name, value)`` pairs in declared order.

        Declared fields come first in the exact ``__dict__`` insertion order
        of the dict-era ``__init__`` (the order every pickle golden was frozen
        against); mirror fields resolve through the representative op. Extra
        instance attributes (user-set names, JMT-FORK-7) follow in insertion
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

        The explicit counterpart of ``__tl_state_items__`` (M11: no record
        class relies on the generic introspection fallback). ``Layer`` is
        dict-backed by design — restored fields become per-layer shadows over
        the M8 mirror descriptors, byte-identical to the dict-era
        ``__dict__.update``.
        """

        self.__dict__.update(mapping)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with weakrefs and raw autograd handles stripped."""
        from ._state_adapter import state_items

        state = dict(state_items(self))
        state["_source_trace_ref"] = None
        # `grad_fn_handle` holds the live torch autograd `Node` (e.g.
        # `AddmmBackward0`), which is not picklable. `Layer.FIELD_POLICY`
        # declares it `FieldPolicy.DROP` for exactly this reason; enforce that
        # here (mirroring `Op.__getstate__`) so plain `pickle.dumps(trace)` of
        # any model with trainable params does not crash. `grad_fn` (a picklable
        # `GradFn` record after a backward pass) is intentionally retained.
        state["grad_fn_handle"] = None
        # R10-7 (mirroring ``Op.__getstate__``): user transform callables
        # (FieldPolicy.DROP) serialize to the loaded-artifact form (None) -- a
        # lambda transform= made pickle.dumps crash on rolled layer records
        # while tl.save succeeded on the same trace.
        if state.get("activation_transform") is not None:
            state["activation_transform"] = None
        if state.get("grad_transform") is not None:
            state["grad_transform"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        read_tlspec_version(state, cls_name=type(self).__name__)
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
        from .._io.state_keys import refuse_callable_shadowing_state_keys

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
            raise InvalidArgumentError(
                f"Layer '{self.layer_label}' has {self.num_passes} ops. "
                f"Access '{field_name}' on a specific pass: "
                f"log['{self.layer_label}'].ops[0].{field_name}",
                code="layer_pass_ambiguous",
                remedy=f"access {field_name!r} on one pass via .ops[k]",
                layer_label=self.layer_label,
                field_name=field_name,
            )
        return getattr(self.ops[0], field_name)

    def __selection__(self) -> object:
        """Lift this layer's whole output as an ACT selection term (ALL passes)."""

        from ..selection import _selection_from_layer

        return _selection_from_layer(self)

    @property
    def receptive_field(self) -> "ReceptiveFieldView":
        """Return the only pass's receptive field or reject a recurrent layer.

        Raises
        ------
        AmbiguousPassError
            If this logical layer contains multiple executed passes.
        """

        if self.num_passes != 1 or len(self.ops) != 1:
            from ..receptive_field._errors import AmbiguousPassError

            passes = ", ".join(f"layer.ops[{index}]" for index in self.ops.keys())
            raise AmbiguousPassError(
                f"Layer {self.layer_label!r} has {self.num_passes} passes: {passes}."
            )
        return self.ops[0].receptive_field

    @property
    def projective_field(self) -> "ReceptiveFieldView":
        """Return the only pass's projective field or reject pass ambiguity."""

        if self.num_passes != 1 or len(self.ops) != 1:
            from ..receptive_field._errors import AmbiguousPassError

            passes = ", ".join(f"layer.ops[{index}]" for index in self.ops.keys())
            raise AmbiguousPassError(
                f"Layer {self.layer_label!r} has {self.num_passes} passes: {passes}."
            )
        return self.ops[0].projective_field

    @property
    def site_key(self) -> str:
        """Single structural site key, or a typed refusal on spanning/legacy.

        Site-uniformity and pass-uniformity are DIFFERENT axes: a multi-pass
        reused-module layer has ONE key; a within-call recurrence group
        spans several (read those via ``.ops[k].site_key``).
        """

        from ._layer_sites import layer_site_key

        return layer_site_key(self)

    @property
    def site_peers(self) -> tuple["Layer", ...]:
        """Layers sharing any of this layer's site keys (live index)."""

        from ._layer_sites import layer_site_peers

        return layer_site_peers(self)

    @property
    def shape_summary(self) -> str | None:
        """Across-pass output-shape summary string (derived, not persisted)."""

        from ._layer_sites import layer_shape_summary

        return layer_shape_summary(self)

    @property
    def out(self) -> Any:
        """Return the saved out for a single-pass layer.

        Returns
        -------
        Any
            Saved out from the only pass.
        """
        return self._single_pass_or_error("out")

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved out on single-pass layers."""

        return self._single_pass_or_error("tensor")

    @property
    def transformed_out(self) -> Any:
        """Transformed out on single-pass layers."""

        return self._single_pass_or_error("transformed_out")

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
    def saved_args(self) -> Any:
        """Return captured positional arguments for a single-pass layer.

        Returns
        -------
        Any
            Captured positional arguments from the only pass.
        """
        return self._single_pass_or_error("saved_args")

    @property
    def saved_kwargs(self) -> Any:
        """Return captured keyword arguments for a single-pass layer.

        Returns
        -------
        Any
            Captured keyword arguments from the only pass.
        """
        return self._single_pass_or_error("saved_kwargs")

    @property
    def grad(self) -> Any:
        """Return the saved grad for a single-pass layer.

        Returns
        -------
        Any
            Saved grad from the only pass.
        """
        return self._single_pass_or_error("grad")

    @property
    def transformed_grad(self) -> Any:
        """Transformed grad on single-pass layers."""

        return self._single_pass_or_error("transformed_grad")

    @property
    def has_grad(self) -> bool:
        """Return whether the single pass has a saved grad.

        Returns
        -------
        bool
            ``True`` when a grad was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_grad"))

    @property
    def code_context(self) -> Any:
        """Return the captured call stack for a single-pass layer.

        Returns
        -------
        Any
            Function call stack from the only pass.
        """
        return self._single_pass_or_error("code_context")

    @property
    def func_duration(self) -> Duration:
        """Return function execution time for a single-pass layer.

        Returns
        -------
        Duration
            Function timing value from the only pass.
        """
        return cast(Duration, self._single_pass_or_error("func_duration"))

    @property
    def total_func_duration(self) -> Duration:
        """Sum of function-call duration across all Ops in this Layer."""

        return Duration(sum(float(op.func_duration or 0) for op in self.ops.values()))

    @property
    def func_rng_states(self) -> Any:
        """Return RNG states captured for a single-pass layer.

        Returns
        -------
        Any
            RNG state snapshot from the only pass.
        """
        return self._single_pass_or_error("func_rng_states")

    @property
    def pass_index(self) -> int:
        """Return the pass number for a single-pass layer.

        Returns
        -------
        int
            Pass number from the only pass.
        """
        return cast(int, self._single_pass_or_error("pass_index"))

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

        Immutable view (M6, JMT-FORK-1): computed per read, so mutating the
        returned container could never reach stored state anyway; the tuple
        makes that contract explicit and matches the Op relation surface.
        """
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.children:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return tuple(result)

    @property
    def parents(self) -> tuple[str, ...]:
        """Union of parent layers (no-pass labels) across all ops.

        Immutable view (M6, JMT-FORK-1); see ``children``.
        """
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.parents:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return tuple(result)

    @property
    def has_children(self) -> bool:
        """Return whether any pass has child layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph children.
        """
        return any(p.has_children for p in self.ops.values())

    @property
    def has_parents(self) -> bool:
        """Return whether any pass has parent layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph parents.
        """
        return any(p.has_parents for p in self.ops.values())

    @property
    def num_parents(self) -> int:
        """Number of distinct parent Layers feeding this Layer."""

        return len(self.parents)

    @property
    def num_children(self) -> int:
        """Number of distinct child Layers fed by this Layer."""

        return len(self.children)

    @property
    def siblings(self) -> list[str]:
        """Union of sibling layers (no-pass labels) across all ops.

        Every pass spelling of THIS layer is excluded: another pass of the same
        layer is a per-op sibling, but at the aggregate level it folds back to
        this layer's own no-pass label, and a layer is never its own sibling.
        """
        result = []
        seen = {self.layer_label}
        for pass_log in self.ops.values():
            for label in pass_log.siblings:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    def _relation_no_pass_label(self, label: str) -> str | None:
        """Resolve one per-op relation label to its no-pass layer label.

        Mirrors the op-level orphan tolerance (r3 b3-fable R05-N1): the
        per-op ``siblings``/``co_parents`` properties deliberately resolve
        relation labels through the ``orphans`` fallback on
        ``keep_orphans=True`` traces, so the Layer aggregate must fold the
        same labels instead of crashing on a bare mainline lookup. A label
        resolvable through neither surface is skipped, matching the op-level
        behavior.
        """

        trace = self.source_trace
        try:
            return trace[label].layer_label
        except (KeyError, ValueError):
            try:
                return trace.orphans[label].layer_label
            except KeyError:
                return None

    @property
    def has_siblings(self) -> bool:
        """Return whether this layer has any sibling layers besides itself.

        Returns
        -------
        bool
            ``True`` when the aggregate ``siblings`` view is non-empty.
        """
        return len(self.siblings) > 0

    @property
    def co_parents(self) -> list[str]:
        """Union of spouse layers (no-pass labels) across all ops.

        Every pass spelling of THIS layer is excluded, mirroring ``siblings``:
        a layer is never its own co-parent at the aggregate level.
        """
        result = []
        seen = {self.layer_label}
        for pass_log in self.ops.values():
            for label in pass_log.co_parents:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def has_co_parents(self) -> bool:
        """Return whether this layer has any co-parent layers besides itself.

        Returns
        -------
        bool
            ``True`` when the aggregate ``co_parents`` view is non-empty.
        """
        return len(self.co_parents) > 0

    @property
    def is_in_conditional(self) -> bool:
        """Whether this layer participates in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this layer computes a conditional arm condition."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this layer is in a conditional arm body."""

        return bool(self.__dict__.get("_is_in_conditional_body", False)) or any(
            role.role == "body" for role in self.in_conditionals or []
        )

    @is_in_conditional_body.setter
    def is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during aggregation."""

        self.__dict__["_is_in_conditional_body"] = value

    @is_in_conditional_body.deleter
    def is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        self.__dict__.pop("_is_in_conditional_body", None)

    @property
    def conditional_depth(self) -> int:
        """Number of distinct conditionals this layer participates in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

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
    def label_short(self) -> str:
        """For single-pass layers, return the short pass-qualified label."""
        return cast(str, self._single_pass_or_error("label_short"))

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
    def children_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: child layer labels (no-pass) for each pass."""
        result = {}
        for call_index, pass_log in self.ops.items():
            children = []
            for label in pass_log.children:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and no_pass not in children:
                    children.append(no_pass)
            result[call_index] = children
        return result

    @property
    def parents_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: parent layer labels (no-pass) for each pass."""
        result = {}
        for call_index, pass_log in self.ops.items():
            parents = []
            for label in pass_log.parents:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and no_pass not in parents:
                    parents.append(no_pass)
            result[call_index] = parents
        return result

    @property
    def child_ops_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each child layer, which ops connect to it."""
        from collections import defaultdict

        result: defaultdict[str, list[int]] = defaultdict(list)
        for call_index, pass_log in self.ops.items():
            for label in pass_log.children:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and call_index not in result[no_pass]:
                    result[no_pass].append(call_index)
        return dict(result)

    @property
    def parent_ops_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each parent layer, which ops connect from it."""
        from collections import defaultdict

        result: defaultdict[str, list[int]] = defaultdict(list)
        for call_index, pass_log in self.ops.items():
            for label in pass_log.parents:
                no_pass = self._relation_no_pass_label(label)
                if no_pass is not None and call_index not in result[no_pass]:
                    result[no_pass].append(call_index)
        return dict(result)

    @property
    def edges_vary_across_ops(self) -> bool:
        """Whether graph edges differ across ops."""
        if self.num_passes <= 1:
            return False
        all_pass_lists = list(self.child_ops_per_layer.values()) + list(
            self.parent_ops_per_layer.values()
        )
        return any(len(ops) < self.num_passes for ops in all_pass_lists)

    @property
    def leaf_module_ops(self) -> set[Any]:
        """Set of module ops exited across all ops."""
        result = set()
        for pass_log in self.ops.values():
            if pass_log.is_atomic_module:
                result.add(pass_log.atomic_module_call)
        return result

    @property
    def parent_arg_positions(self) -> dict[str, dict[Any, str]]:
        """Merged parent_arg_positions across ops (set-union).

        For single-pass layers, delegates to ops[0].
        For multi-pass, merges arg locs using set-union of no-pass labels.
        """
        if self.num_passes == 1:
            return cast(dict[str, dict[Any, str]], self.ops[0].parent_arg_positions)

        result: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
        for pass_log in self.ops.values():
            for arg_type in ["args", "kwargs"]:
                for arg_key, layer_label in pass_log.parent_arg_positions[arg_type].items():
                    no_pass = self.source_trace[layer_label].layer_label
                    if arg_key not in result[arg_type]:
                        result[arg_type][arg_key] = no_pass
        return result

    # ********************************************
    # ******* Fallback __getattr__ ***************
    # ********************************************

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute lookup: delegates to ops[0] for single-pass layers.

        Only called when normal attribute lookup has already failed (Python's
        ``__getattr__`` protocol).  For single-pass layers, transparently
        forwards to the underlying Op, enabling code like
        ``layer_log.func_rng_states`` without needing an explicit property.

        Private attributes (starting with '_') are never delegated — they
        raise AttributeError immediately to avoid infinite recursion with
        ``self.__dict__`` access.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name in {
            "param_memory",
        }:
            raise AttributeError(name)
        ops = self.__dict__.get("ops")
        if ops and len(ops) == 1:
            try:
                return getattr(ops[0], name)
            except AttributeError:
                pass
        if ops and len(ops) > 1 and name in _LAYER_DELEGATED_PASS_FIELDS:
            return self._single_pass_or_error(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ********************************************
    # ************ User-facing custom_methods ***********
    # ********************************************

    def _resolve_relation_record(self, label: str) -> "Layer | None":
        """Resolve one relation label through the mainline, then orphans.

        Mirrors the op-level orphan tolerance (b3 R05-N2): the aggregates on
        a ``keep_orphans=True`` trace include orphan relation labels, so the
        object-resolving surfaces must fold the ``orphans`` fallback instead
        of crashing on the bare mainline lookup. An orphan label resolves to
        its retained orphan Op record (the only record that exists for it),
        presented through the shared record surface.
        """

        trace = self.source_trace
        try:
            return cast("Layer", trace[label])
        except (KeyError, ValueError):
            try:
                return cast("Layer", trace.orphans[label])
            except KeyError:
                return None

    def get_children(self) -> list["Layer"]:
        """Return child Layer objects for this layer.

        Returns
        -------
        list[Layer]
            Child layers resolved through the owning model log; orphan
            relation labels resolve through ``trace.orphans`` (unresolvable
            skipped).
        """
        resolved = (self._resolve_relation_record(label) for label in self.children)
        return [record for record in resolved if record is not None]

    def get_parents(self) -> list["Layer"]:
        """Return parent Layer objects for this layer.

        Returns
        -------
        list[Layer]
            Parent layers resolved through the owning model log; orphan
            relation labels resolve through ``trace.orphans`` (unresolvable
            skipped).
        """
        resolved = (self._resolve_relation_record(label) for label in self.parents)
        return [record for record in resolved if record is not None]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this layer's saved out.

        Parameters
        ----------
        method:
            Display method. ``"auto"`` chooses from tensor shape.
        **kwargs:
            Forwarded to the tensor display helper.

        Returns
        -------
        Any
            Matplotlib figure when plotting is available, otherwise a text
            fallback explaining why no plot was produced.
        """

        from ..viz._tensor_display import show_tensor

        return show_tensor(self, method=method, **kwargs)

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
        transform: Any,
        *,
        model: Any = None,
        x: Any = None,
        intervention: Any = None,
    ) -> "Trace":
        """Apply an intervention to this layer through the owning Trace.

        Parameters
        ----------
        transform:
            Transform or hook to apply to this layer's output.
        model:
            Model required when ``intervention.engine="rerun"``.
        x:
            Input required when ``intervention.engine="rerun"``.
        intervention:
            Grouped intervention options (``InterventionOptions``).

        Returns
        -------
        Trace
            Source Trace after applying the intervention.
        """

        return self._source_trace_or_error().do(
            self.layer_label,
            transform,
            model=model,
            x=x,
            intervention=intervention,
        )

    def set(
        self,
        value: Any,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Set this layer's out recipe through the owning Trace.

        Parameters
        ----------
        value:
            Static replacement value or one-shot callable.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Trace
            Source Trace with a stale intervention recipe.
        """

        return self._source_trace_or_error().set(
            self.layer_label,
            value,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )

    def attach_hooks(
        self,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
        """Attach sticky hooks to this layer through the owning Trace.

        Parameters
        ----------
        hook:
            Hook or helper to attach to this layer.
        *extra_hooks:
            Additional hooks to compose on this layer in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Any
            Trace or scoped removable hook handle, matching ``Trace.attach_hooks``.
        """

        return self._source_trace_or_error().attach_hooks(
            self.layer_label,
            hook,
            *extra_hooks,
            strict=strict,
            prepend=prepend,
            confirm_mutation=confirm_mutation,
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
        from ..constants import LAYER_LOG_FIELD_ORDER

        row = _layer_log_to_row(self)
        return pd.DataFrame([row], columns=LAYER_LOG_FIELD_ORDER)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        """Return a human-readable layer summary.

        Data-model contract: never raises. A Layer detached from its Trace
        (collected, standalone-pickled, or husked by cleanup) degrades to a
        one-line placeholder instead of propagating the typed relation
        refusal out of ``repr()``/``print()``/f-string interpolation
        (r4 b7-opus R52-A).
        """

        if not self._tracing_finished:
            return f"Layer({self.layer_label}) (pass not finished)"
        try:
            return self._describe_bound()
        except RecordBindingError:
            label = getattr(self, "layer_label", None) or "<unbound>"
            return f"<Layer {label}: detached from its Trace>"

    def _describe_bound(self) -> str:
        """Build the full summary; relation reads require a live Trace."""

        s = f"Layer {self.layer_label}:"
        if self.num_passes > 1:
            s += f" ({self.num_passes} ops)"
        s += f"\n\tOutput tensor: shape={self.shape}, dtype={self.dtype}, size={self.activation_memory}"
        if not self.is_input:
            s += f"\n\tFunction: {self.func_name} (grad_fn_handle: {self.grad_fn_class_name})"
            if self.func_config:
                config_str = format_config_items(self.func_config)
                s += f"\n\tConfig: {config_str}"
        if self.module is not None:
            s += f"\n\tComputed inside module: {self.module}"
        if len(self.param_shapes) > 0:
            params_shapes_str = format_shape_list(self.param_shapes)
            s += (
                f"\n\tParams: {params_shapes_str}; "
                f"{self.num_params} total ({self.total_param_memory})"
            )
        s += "\n\tRelated Layers:"
        s += f"\n\t\t- parents: {', '.join(self.parents) or 'none'}"
        s += f"\n\t\t- children: {', '.join(self.children) or 'none'}"
        if self.num_passes > 1:
            s += f"\n\tPasses: {', '.join(self.call_labels)}"
        return s

    def __repr__(self) -> str:
        """Return the developer representation for this layer."""

        return self.__str__()

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
        if name in vars(Layer):
            raise RuntimeError(
                f"Layer mirror field {name!r} collides with an existing class attribute"
            )
        setattr(Layer, name, _LayerMirrorField(name, source, normalize))


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
        if name in vars(Layer):
            raise RuntimeError(
                f"Layer view field {name!r} collides with an existing class attribute"
            )
        setattr(Layer, name, _LayerViewField(name))


_install_layer_mirror_descriptors()
_install_layer_view_descriptors()


class LayerAccessor(Accessor["Layer"]):
    """Dict-like accessor for Layer objects.

    Supports indexing by:
    * **layer label** (str) -- exact match against no-pass label.
    * **ordinal index** (int) -- position in execution order.
    * **pass notation** (str ``"conv2d_1_1:2"``) -- strips the pass
      suffix and returns the parent Layer.

    Available as ``trace.layers``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        layer_logs: dict[str, "Layer"],
        source_trace: Optional["Trace"] = None,
    ) -> None:
        """Initialize an accessor over aggregate layer logs.

        Parameters
        ----------
        layer_logs:
            Mapping from layer labels to aggregate ``Layer`` objects.
        source_trace:
            Trace that owns the layer logs, if still reachable.
        """

        source_ref = weakref.ref(source_trace) if source_trace is not None else None
        super().__init__(layer_logs, source_ref=source_ref)

    def _resolve_pass_qualified(self, key: str) -> "Layer | None":
        """Resolve ``layer_label:pass`` notation to the parent Layer."""
        base, _, pass_str = key.rpartition(":")
        try:
            int(pass_str)
        except ValueError:
            return None
        return self._resolve_substring(base)

    def _resolve_substring(self, key: str) -> "Layer | None":
        """Resolve exact long or short Layer labels."""
        if key in self._dict:
            return self._dict[key]
        matches = [
            layer
            for layer in self._list
            if key in {layer.layer_label, layer.layer_label, layer.layer_label_short}
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousOpLookupError(
                f"Layer lookup '{key}' is ambiguous across {len(matches)} Layers. "
                "Use the full Layer label."
            )
        return None

    def _suggest(self, key: str) -> list[str]:
        """Return similar layer labels from the source Trace."""
        source_ref = getattr(self, "_source_ref", None)
        source = source_ref() if source_ref is not None else None
        if source is not None and hasattr(source, "find_layers"):
            return source.find_layers(str(key))
        return []

    def by_operator(self, operator: str | None = None) -> dict[str, int] | list[str]:
        """Group layers by Torch operator name.

        Parameters
        ----------
        operator:
            Optional operator name. When supplied, matching layer labels are returned.

        Returns
        -------
        Dict[str, int] | List[str]
            Counts by operator, or labels for one operator.
        """

        if operator is not None:
            return [
                layer.layer_label
                for layer in self._list
                if (layer.func_name or layer.layer_type) == operator
            ]
        counts: dict[str, int] = {}
        for layer in self._list:
            key = str(layer.func_name or layer.layer_type)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_module(self, module: str | None = None) -> dict[str, int] | list[str]:
        """Group layers by containing module address.

        Parameters
        ----------
        module:
            Optional module address. When supplied, matching layer labels are returned.

        Returns
        -------
        Dict[str, int] | List[str]
            Counts by module, or labels for one module.
        """

        if module is not None:
            return [
                layer.layer_label
                for layer in self._list
                if layer.module == module or module in getattr(layer, "modules", [])
            ]
        counts: dict[str, int] = {}
        for layer in self._list:
            key = str(layer.module or "self")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_module_and_operator(
        self,
        module: str | None = None,
        operator: str | None = None,
    ) -> dict[tuple[str, str], int] | list[str]:
        """Group layers by module and operator.

        Parameters
        ----------
        module:
            Optional module address filter.
        operator:
            Optional operator-name filter.

        Returns
        -------
        Dict[Tuple[str, str], int] | List[str]
            Counts by ``(module, operator)`` or labels matching both filters.
        """

        if module is not None and operator is not None:
            return [
                layer.layer_label
                for layer in self._list
                if (layer.module == module or module in getattr(layer, "modules", []))
                and (layer.func_name or layer.layer_type) == operator
            ]
        counts: dict[tuple[str, str], int] = {}
        for layer in self._list:
            key = (str(layer.module or "self"), str(layer.func_name or layer.layer_type))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def total(self) -> int:
        """Return the number of aggregate layers.

        Returns
        -------
        int
            Number of layer logs.
        """

        return len(self)

    def __repr__(self) -> str:
        """Return a compact multi-line accessor summary."""

        if len(self) == 0:
            return "LayerAccessor({})"
        items = []
        for ll in self._list:
            items.append(
                f"  '{ll.layer_label}': {ll.func_name or 'input'} "
                f"(shape={list(ll.shape) if ll.shape else '?'}, "
                f"ops={ll.num_passes})"
            )
        inner = "\n".join(items)
        return f"LayerAccessor({len(self)} layers):\n{inner}"

    def to_pandas(self) -> "pd.DataFrame":
        """One row per unique layer (aggregate view), ordered by ``LAYER_LOG_FIELD_ORDER``.

        Builds each row the same way as ``Layer.to_pandas()`` so every field
        in ``LAYER_LOG_FIELD_ORDER`` is exported -- this used to hand-roll a
        12-field subset that silently dropped most populated Layer fields.
        Per-pass fields (``_MULTI_PASS_PER_CALL_LAYER_FIELDS``) are reported
        as ``None`` for multi-pass (recurrent) layers instead of raising.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        if not self._list:
            return pd.DataFrame(columns=LAYER_LOG_FIELD_ORDER)
        rows = [_layer_log_to_row(ll) for ll in self._list]
        return pd.DataFrame(rows, columns=LAYER_LOG_FIELD_ORDER)
