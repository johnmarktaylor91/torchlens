"""Concern-group table for the ``Op`` and ``Layer`` record members (pure data).

``Op`` and ``Layer`` group their members by concern the way ``Trace.cost``
does: ``op.cost.flops_forward``, ``layer.graph.has_children``. Each row
below names one former flat member, the group it moved to (or ``rename``
for a flat rename, ``removed`` for a removed member) and its new leaf
name. The persisted field names and the writer contract are unchanged: a
stored member keeps its column and is read through a private storage
attribute (``storage_attr``), and its group property reads that attribute.
A former flat spelling refuses typed (``facade_redirect``) naming the new
spelling.

Only rows whose group is in ``ACTIVE_GROUPS`` are applied; the rest are
the remaining part of the regrouping and change nothing yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Groups whose rows are applied (moved, renamed or removed).
ACTIVE_GROUPS: frozenset[str] = frozenset(
    {
        "cost",
        "control_flow",
        "call",
        "source",
        "autograd",
        "module_context",
        "payload",
        "graph",
        "rename",
        "removed",
    }
)

#: ``(record, former flat name)`` rows of an active group that stay flat for
#: now. ``Op.site_key`` is the selector address the public selector text tells
#: users to pass; removing it waits on the replacement spelling for that text.
PENDING_REMOVALS: frozenset[tuple[str, str]] = frozenset({("Op", "site_key")})

#: ``(former flat name, group | "rename" | "removed", new leaf name)``.
OP_MEMBER_MOVES: tuple[tuple[str, str, str], ...] = (
    ("activation_memory", "cost", "activation_memory"),
    ("activation_transform", "removed", ""),
    ("address", "rename", "buffer_address"),
    ("arg_expressions", "source", "arg_expressions"),
    ("arg_names", "call", "arg_names"),
    ("args_template", "call", "args_template"),
    ("arithmetic_intensity", "cost", "arithmetic_intensity"),
    ("atomic_module", "module_context", "atomic_module"),
    ("atomic_module_address", "module_context", "atomic_module_address"),
    ("atomic_module_call", "module_context", "atomic_module_call"),
    ("autograd_memory", "cost", "autograd_memory"),
    ("bool_value", "control_flow", "bool_value"),
    ("buffer_sink_ops", "graph", "buffer_sink_ops"),
    ("buffer_source", "graph", "buffer_source"),
    ("buffer_source_func_name", "graph", "buffer_source_func_name"),
    ("buffer_source_ops", "graph", "buffer_source_ops"),
    ("bytes_delta_at_call", "cost", "bytes_delta_at_call"),
    ("bytes_peak_at_call", "cost", "bytes_peak_at_call"),
    ("bytes_read", "cost", "bytes_read"),
    ("bytes_written", "cost", "bytes_written"),
    ("co_parents", "graph", "co_parents"),
    ("code_context", "source", "frames"),
    ("compute_record", "cost", "compute_record"),
    ("conditional_arm_children", "control_flow", "arm_children"),
    ("conditional_branch_depth", "control_flow", "branch_depth"),
    ("conditional_branch_stack", "control_flow", "branch_stack"),
    ("conditional_context_kind", "control_flow", "context_kind"),
    ("conditional_depth", "control_flow", "depth"),
    ("conditional_elif_children", "control_flow", "elif_children"),
    ("conditional_else_children", "control_flow", "else_children"),
    ("conditional_entry_children", "control_flow", "entry_children"),
    ("conditional_then_children", "control_flow", "then_children"),
    ("conditional_wrapper_kind", "control_flow", "wrapper_kind"),
    ("derived_grad", "autograd", "derived_grad"),
    ("detach_saved_activations", "removed", ""),
    ("dropped_edge_tensor_args", "call", "dropped_edge_tensor_args"),
    ("edge_substitutions", "graph", "edge_substitutions"),
    ("equivalence_class", "graph", "equivalence_class"),
    ("equivalent_ops", "graph", "equivalent_ops"),
    ("flops_backward", "cost", "flops_backward"),
    ("flops_forward", "cost", "flops_forward"),
    ("flops_total", "cost", "flops_total"),
    ("func_autocast_state", "call", "autocast_state"),
    ("func_call_id", "call", "call_id"),
    ("func_config", "call", "config"),
    ("func_duration", "cost", "duration"),
    ("func_id", "call", "func_id"),
    ("func_non_tensor_args", "call", "non_tensor_args"),
    ("func_qualname", "source", "func_qualname"),
    ("func_rng_states", "call", "rng_states"),
    ("fx_call_index", "source", "fx_call_index"),
    ("fx_qualpath", "source", "fx_path"),
    ("grad_dtype", "autograd", "grad_dtype"),
    ("grad_fn", "autograd", "grad_fn"),
    ("grad_fn_class_name", "autograd", "grad_fn_class_name"),
    ("grad_fn_class_qualname", "autograd", "grad_fn_class_qualname"),
    ("grad_fn_cls", "removed", ""),
    ("grad_fn_handle", "autograd", "grad_fn_handle"),
    ("grad_fn_object_id", "removed", ""),
    ("grad_for", "autograd", "grad_for"),
    ("grad_ref", "removed", ""),
    ("grad_shape", "autograd", "grad_shape"),
    ("grads", "autograd", "grads"),
    ("has_children", "graph", "has_children"),
    ("has_co_parents", "graph", "has_co_parents"),
    ("has_input_ancestor", "graph", "has_input_ancestor"),
    ("has_internal_source_ancestor", "graph", "has_internal_source_ancestor"),
    ("has_output_descendant", "graph", "has_output_descendant"),
    ("has_parents", "graph", "has_parents"),
    ("has_saved_args", "call", "has_saved_args"),
    ("has_siblings", "graph", "has_siblings"),
    ("in_conditionals", "control_flow", "in_conditionals"),
    ("injection_provenance", "call", "injection_provenance"),
    ("input_memory", "cost", "input_memory"),
    ("input_to_module_calls", "module_context", "input_to_module_calls"),
    ("input_to_modules", "module_context", "input_to_modules"),
    ("internal_source_ancestors", "graph", "internal_source_ancestors"),
    ("internal_source_parents", "graph", "internal_source_parents"),
    ("io_role", "graph", "io_role"),
    ("is_atomic_module", "module_context", "is_atomic_module"),
    ("is_buffer", "graph", "is_buffer"),
    ("is_compute_op", "graph", "is_compute"),
    ("is_final_output", "graph", "is_final_output"),
    ("is_in_conditional", "control_flow", "is_in_conditional"),
    ("is_in_conditional_body", "control_flow", "is_in_conditional_body"),
    ("is_in_conditional_evaluation", "control_flow", "is_in_conditional_evaluation"),
    ("is_internal_source", "graph", "is_internal_source"),
    ("is_internally_initialized", "call", "is_internally_initialized"),
    ("is_module_input", "module_context", "is_module_input"),
    ("is_module_output", "module_context", "is_module_output"),
    ("is_orphan", "graph", "is_orphan"),
    ("is_output_parent", "graph", "is_output_parent"),
    ("is_scalar_bool", "control_flow", "is_scalar_bool"),
    ("is_terminal_bool", "control_flow", "is_terminal_bool"),
    ("is_terminal_conditional_bool", "control_flow", "is_terminal_conditional_bool"),
    ("is_transform", "call", "is_func_transform"),
    ("kwargs_template", "call", "kwargs_template"),
    ("macs_forward", "cost", "macs_forward"),
    ("materialize_grad", "payload", "materialize_grad"),
    ("materialize_out", "payload", "materialize_out"),
    ("max_distance_from_input", "graph", "max_distance_from_input"),
    ("max_distance_to_output", "graph", "max_distance_to_output"),
    ("min_distance_from_input", "graph", "min_distance_from_input"),
    ("min_distance_to_output", "graph", "min_distance_to_output"),
    ("module_call_depth", "module_context", "call_depth"),
    ("module_call_stack", "module_context", "call_stack"),
    ("module_entry_arg_keys", "module_context", "entry_arg_keys"),
    ("non_tensor_kwargs", "call", "non_tensor_kwargs"),
    ("non_tensor_pos_args", "call", "non_tensor_pos_args"),
    ("num_args_total", "call", "num_args_total"),
    ("num_autograd_tensors", "autograd", "num_autograd_tensors"),
    ("num_inputs", "graph", "num_inputs"),
    ("num_kwargs", "call", "num_kwargs"),
    ("num_pos_args", "call", "num_pos_args"),
    ("out_ref", "removed", ""),
    ("output_device", "removed", ""),
    ("output_of_module_calls", "module_context", "output_of_module_calls"),
    ("output_of_modules", "module_context", "output_of_modules"),
    ("param_memory", "cost", "param_memory"),
    ("parent_arg_positions", "call", "parent_arg_positions"),
    ("parent_param_ops", "graph", "parent_param_ops"),
    ("parent_params", "graph", "parent_params"),
    ("recurrent_ops", "graph", "recurrent_ops"),
    ("root_ancestors", "graph", "root_ancestors"),
    ("saved_args", "call", "saved_args"),
    ("saved_kwargs", "call", "saved_kwargs"),
    ("siblings", "graph", "siblings"),
    ("site_key", "removed", ""),
    ("terminal_bool_for", "control_flow", "terminal_bool_for"),
    ("tl_authored_root", "call", "tl_authored_root"),
    ("transform_chain", "call", "func_transform_chain"),
    ("transform_config", "call", "func_transform_config"),
    ("transform_fn_name", "call", "func_transform_fn_name"),
    ("transform_fn_qualname", "call", "func_transform_fn_qualname"),
    ("transform_fn_source", "call", "func_transform_fn_source"),
    ("transform_kind", "call", "func_transform_kind"),
    ("transformed_activation_memory", "cost", "transformed_activation_memory"),
    ("transformed_grad", "payload", "transformed_grad"),
    ("transformed_grad_dtype", "payload", "transformed_grad_dtype"),
    ("transformed_grad_shape", "payload", "transformed_grad_shape"),
    ("transformed_out", "payload", "transformed_out"),
    ("transformed_out_dtype", "payload", "transformed_out_dtype"),
    ("transformed_out_shape", "payload", "transformed_out_shape"),
    ("unattributed_tensor_args", "call", "unattributed_tensor_args"),
    ("var_names", "source", "var_names"),
)

LAYER_MEMBER_MOVES: tuple[tuple[str, str, str], ...] = (
    ("activation_memory", "cost", "activation_memory"),
    ("activation_transform", "removed", ""),
    ("address", "rename", "buffer_address"),
    ("arg_names", "call", "arg_names"),
    ("autograd_memory", "cost", "autograd_memory"),
    ("bool_value", "control_flow", "bool_value"),
    ("buffer_pass", "graph", "buffer_pass"),
    ("buffer_source", "graph", "buffer_source"),
    ("buffer_source_func_name", "graph", "buffer_source_func_name"),
    ("child_ops_per_layer", "graph", "child_ops_per_layer"),
    ("children_per_pass", "graph", "children_per_pass"),
    ("code_context", "source", "frames"),
    ("conditional_arm_children", "control_flow", "arm_children"),
    ("conditional_branch_stack_ops", "control_flow", "branch_stack_ops"),
    ("conditional_depth", "control_flow", "depth"),
    ("conditional_elif_children", "control_flow", "elif_children"),
    ("conditional_else_children", "control_flow", "else_children"),
    ("conditional_entry_children", "control_flow", "entry_children"),
    ("conditional_role_stacks", "control_flow", "role_stacks"),
    ("conditional_then_children", "control_flow", "then_children"),
    ("detach_saved_activations", "removed", ""),
    ("equivalence_class", "graph", "equivalence_class"),
    ("equivalent_ops", "graph", "equivalent_ops"),
    ("flops_backward", "cost", "flops_backward"),
    ("flops_forward", "cost", "flops_forward"),
    ("flops_total", "cost", "flops_total"),
    ("func_config", "call", "config"),
    ("func_duration", "cost", "duration"),
    ("func_rng_states", "call", "rng_states"),
    ("fx_call_index", "source", "fx_call_index"),
    ("fx_qualpath", "source", "fx_path"),
    ("grad_fn", "autograd", "grad_fn"),
    ("grad_fn_class_name", "autograd", "grad_fn_class_name"),
    ("grad_fn_class_qualname", "autograd", "grad_fn_class_qualname"),
    ("grad_fn_handle", "autograd", "grad_fn_handle"),
    ("grad_fn_object_id", "removed", ""),
    ("has_children", "graph", "has_children"),
    ("has_co_parents", "graph", "has_co_parents"),
    ("has_input_ancestor", "graph", "has_input_ancestor"),
    ("has_parents", "graph", "has_parents"),
    ("has_siblings", "graph", "has_siblings"),
    ("in_conditionals", "control_flow", "in_conditionals"),
    ("io_role", "graph", "io_role"),
    ("is_atomic_module", "module_context", "is_atomic_module"),
    ("is_buffer", "graph", "is_buffer"),
    ("is_compute_layer", "graph", "is_compute"),
    ("is_final_output", "graph", "is_final_output"),
    ("is_in_conditional", "control_flow", "is_in_conditional"),
    ("is_in_conditional_body", "control_flow", "is_in_conditional_body"),
    ("is_in_conditional_evaluation", "control_flow", "is_in_conditional_evaluation"),
    ("is_internal_sink", "graph", "is_internal_sink"),
    ("is_internal_source", "graph", "is_internal_source"),
    ("is_module_input", "module_context", "is_module_input"),
    ("is_orphan", "graph", "is_orphan"),
    ("is_scalar_bool", "control_flow", "is_scalar_bool"),
    ("is_terminal_bool", "control_flow", "is_terminal_bool"),
    ("macs_forward", "cost", "macs_forward"),
    ("module_call_depth", "module_context", "call_depth"),
    ("num_args_total", "call", "num_args_total"),
    ("num_autograd_tensors", "autograd", "num_autograd_tensors"),
    ("num_kwargs", "call", "num_kwargs"),
    ("num_pos_args", "call", "num_pos_args"),
    ("output_device", "removed", ""),
    ("output_of_module_calls", "module_context", "output_of_module_calls"),
    ("output_of_modules", "module_context", "output_of_modules"),
    ("parent_arg_positions", "call", "parent_arg_positions"),
    ("parent_ops_per_layer", "graph", "parent_ops_per_layer"),
    ("parents_per_pass", "graph", "parents_per_pass"),
    ("saved_args", "call", "saved_args"),
    ("saved_kwargs", "call", "saved_kwargs"),
    ("site_peers", "graph", "site_peers"),
    ("terminal_bool_for", "control_flow", "terminal_bool_for"),
    ("total_activation_memory", "cost", "total_activation_memory"),
    ("total_autograd_memory", "cost", "total_autograd_memory"),
    ("total_flops_backward", "cost", "total_flops_backward"),
    ("total_flops_forward", "cost", "total_flops_forward"),
    ("total_flops_total", "cost", "total_flops"),
    ("total_func_duration", "cost", "total_func_duration"),
    ("total_macs_forward", "cost", "total_macs_forward"),
    ("total_macs_total", "cost", "total_macs"),
    ("total_param_memory", "cost", "total_param_memory"),
    ("transformed_activation_memory", "cost", "transformed_activation_memory"),
    ("transformed_grad", "payload", "transformed_grad"),
    ("transformed_grad_dtype", "payload", "transformed_grad_dtype"),
    ("transformed_grad_shape", "payload", "transformed_grad_shape"),
    ("transformed_out", "payload", "transformed_out"),
    ("transformed_out_dtype", "payload", "transformed_out_dtype"),
    ("transformed_out_shape", "payload", "transformed_out_shape"),
)

_RECORD_VARS = {"Op": "op", "Layer": "layer"}
_MOVES = {"Op": OP_MEMBER_MOVES, "Layer": LAYER_MEMBER_MOVES}

#: Remedies for removed members without a direct replacement.
_REMOVED_REMEDIES: Mapping[str, str] = {
    "activation_transform": "use trace.options.activation_transform -- the transform is a capture setting",
    "detach_saved_activations": (
        "use trace.options.detach_saved_activations -- detaching is a capture setting"
    ),
    "output_device": "use trace.options.output_device -- the output device is a capture setting",
    "out_ref": "use {var}.out or {var}.payload.materialize_out() -- the lazy reference is internal",
    "grad_ref": "use {var}.grad or {var}.payload.materialize_grad() -- the lazy reference is internal",
    "grad_fn_object_id": "use {var}.autograd.grad_fn -- the autograd node record replaces its id",
    "grad_fn_cls": "use {var}.autograd.grad_fn_class_name -- the class is recorded by name",
}


def storage_attr(group: str, old: str, leaf: str) -> str:
    """Return the attribute that holds one moved member on its record.

    Parameters
    ----------
    group:
        Concern group, ``"rename"`` or ``"removed"``.
    old:
        Former flat member name (the persisted field name for stored rows).
    leaf:
        New leaf name inside the group (or the new flat name).

    Returns
    -------
    str
        ``_<group>_<leaf>`` for a grouped member, the new name for a flat
        rename and ``_<old>`` for a removed member.

    Examples
    --------
    >>> storage_attr("cost", "func_duration", "duration")
    '_cost_duration'
    """

    if group == "rename":
        return leaf
    if group == "removed":
        return f"_{old}"
    return f"_{group}_{leaf}"


def _active_rows(record: str) -> tuple[tuple[str, str, str], ...]:
    """Return the applied rows for ``record`` (``"Op"`` or ``"Layer"``)."""

    return tuple(
        row
        for row in _MOVES[record]
        if row[1] in ACTIVE_GROUPS and (record, row[0]) not in PENDING_REMOVALS
    )


def _remedy(record: str, old: str, group: str, leaf: str) -> str:
    """Return the typed-refusal remedy naming the new spelling."""

    var = _RECORD_VARS[record]
    if group == "rename":
        return f"use {var}.{leaf} -- {record}.{old} was renamed"
    if group == "removed":
        remedy = _REMOVED_REMEDIES.get(old)
        if remedy is None:
            return f"{record}.{old} has no public replacement"
        return remedy.format(var=var)
    return f"use {var}.{group}.{leaf} -- {record}.{old} moved to the {group} group"


def _storage_map(record: str) -> dict[str, str]:
    """Return ``former name -> storage attribute`` for the applied rows."""

    return {old: storage_attr(group, old, leaf) for old, group, leaf in _active_rows(record)}


def _removed_map(record: str) -> dict[str, str]:
    """Return ``former name -> remedy`` for the applied rows."""

    return {old: _remedy(record, old, group, leaf) for old, group, leaf in _active_rows(record)}


#: Op concern-group view class by group name, filled as each group module loads; a
#: capture-time ``LiveOpView`` serves the same groups (``view.call.call_id``).
OP_GROUP_VIEWS: dict[str, type] = {}

#: Former Op member name -> attribute that now holds it.
OP_STORAGE_ATTRS: Mapping[str, str] = _storage_map("Op")
#: Former Layer member name -> attribute that now holds it.
LAYER_STORAGE_ATTRS: Mapping[str, str] = _storage_map("Layer")
#: Former Op member names a Layer does not store itself (read through its pass).
_OP_ONLY_STORAGE = frozenset(OP_STORAGE_ATTRS) - frozenset(LAYER_STORAGE_ATTRS)
#: Former Op member name -> typed-refusal remedy.
OP_MOVED_MEMBERS: Mapping[str, str] = _removed_map("Op")
#: Former Layer member name -> typed-refusal remedy.
LAYER_MOVED_MEMBERS: Mapping[str, str] = _removed_map("Layer")


def op_attr(field: str) -> str:
    """Return the Op attribute that reads persisted field ``field``.

    Examples
    --------
    >>> op_attr("flops_forward")
    '_cost_flops_forward'
    >>> op_attr("label")
    'label'
    """

    return OP_STORAGE_ATTRS.get(field, field)


def op_spelling(name: str) -> str:
    """Return the public Op spelling (``"cost.duration"``) of a former member name.

    Examples
    --------
    >>> op_spelling("func_duration")
    'cost.duration'
    """

    for old, group, leaf in _active_rows("Op"):
        if old == name and group != "removed":
            return leaf if group == "rename" else f"{group}.{leaf}"
    return name


def read_layer_field(layer: Any, field: str) -> Any:
    """Read persisted or former Layer member ``field`` through its current spelling.

    Generic readers that walk ``LAYER_LOG_FIELD_ORDER`` (table rows, exports)
    use this so a moved member reads from its group or storage attribute.

    Examples
    --------
    >>> class _Stub:
    ...     label = "x"
    >>> read_layer_field(_Stub(), "label")
    'x'
    """

    path = _SPELLING_PATHS["Layer"].get(field, (field,))
    value = layer
    for part in path:
        value = getattr(value, part)
    return value


_NO_DEFAULT: Any = object()
_RECORD_KIND_BY_TYPE: dict[type, str | None] = {}
_RECORD_MODULES = {
    "Op": "torchlens._data_classes.op",
    "Layer": "torchlens._data_classes.layer",
}


def _record_kind(cls: type) -> str | None:
    """Return ``"Op"`` or ``"Layer"`` when ``cls`` is (a subclass of) that record."""

    try:
        return _RECORD_KIND_BY_TYPE[cls]
    except KeyError:
        pass
    kind = None
    for klass in cls.__mro__:
        if _RECORD_MODULES.get(klass.__name__) == klass.__module__:
            kind = klass.__name__
            break
    _RECORD_KIND_BY_TYPE[cls] = kind
    return kind


def record_member(obj: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    """``getattr`` that follows a former flat Op or Layer member to its group.

    Duck-typed readers accept an ``Op``, a ``Layer`` or another record with
    the same flat name (``Module.total_flops_forward``); on an Op or Layer the
    former name reads through its current spelling (a removed member reads its
    stored value), on anything else this is plain ``getattr``.

    Parameters
    ----------
    obj:
        Record or other object to read.
    name:
        Former flat member name.
    default:
        Value returned when the member is missing, as for ``getattr``.

    Returns
    -------
    Any
        The member value, or ``default``.

    Examples
    --------
    >>> record_member(object(), "flops_forward", None) is None
    True
    """

    kind = _record_kind(type(obj))
    path = None if kind is None else _SPELLING_PATHS[kind].get(name)
    if path is None and kind == "Layer":
        # An Op-only member reads through the layer's group, which delegates to its pass.
        path = _SPELLING_PATHS["Op"].get(name)
    try:
        if path is None:
            # A renamed or removed member keeps its value under its storage
            # attribute; internal readers still need it.
            try:
                return getattr(obj, name if kind is None else record_storage_attr(obj, name))
            except AttributeError:
                if kind != "Layer" or name not in _OP_ONLY_STORAGE:
                    raise
            # A removed Op member the Layer does not hold reads from its single pass.
            return obj._single_pass_or_error(name)
        value = obj
        for part in path:
            value = getattr(value, part)
        return value
    except AttributeError:
        if default is _NO_DEFAULT:
            raise
        return default


def _spelling_paths(record: str) -> dict[str, tuple[str, ...]]:
    """Return ``former name -> attribute path`` for the applied rows."""

    paths: dict[str, tuple[str, ...]] = {}
    for old, group, leaf in _active_rows(record):
        if group == "rename":
            paths[old] = (leaf,)
        elif group == "removed":
            paths[old] = (f"_{old}",)
        else:
            paths[old] = (group, leaf)
    return paths


_SPELLING_PATHS = {"Op": _spelling_paths("Op"), "Layer": _spelling_paths("Layer")}


def layer_attr(field: str) -> str:
    """Return the Layer attribute that reads persisted field ``field``.

    Examples
    --------
    >>> layer_attr("flops_forward")
    '_cost_flops_forward'
    """

    return LAYER_STORAGE_ATTRS.get(field, field)


def record_storage_attr(obj: Any, field: str) -> str:
    """Return the attribute of ``obj`` that stores persisted field ``field``.

    Generic state writers (rehydrate, cleanup) name fields by their persisted
    key; on an Op or Layer a moved field lives under its private storage
    attribute, on anything else the key is the attribute.

    Examples
    --------
    >>> record_storage_attr(object(), "flops_forward")
    'flops_forward'
    """

    kind = _record_kind(type(obj))
    if kind is None:
        return field
    storage = OP_STORAGE_ATTRS if kind == "Op" else LAYER_STORAGE_ATTRS
    return storage.get(field, field)


__all__ = [
    "ACTIVE_GROUPS",
    "PENDING_REMOVALS",
    "LAYER_MEMBER_MOVES",
    "LAYER_MOVED_MEMBERS",
    "LAYER_STORAGE_ATTRS",
    "OP_MEMBER_MOVES",
    "OP_MOVED_MEMBERS",
    "OP_STORAGE_ATTRS",
    "layer_attr",
    "op_attr",
    "op_spelling",
    "read_layer_field",
    "record_member",
    "record_storage_attr",
    "storage_attr",
]
