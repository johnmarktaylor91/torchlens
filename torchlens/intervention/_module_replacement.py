"""Record a post-capture ``do(tl.module(...))`` edit as its own replacement op.

A capture-time module edit (``intervene=tl.when(tl.module(addr), ...)``)
fires at the module's exit and is recorded as an ``interventionreplacement``
op between the module's last op and its consumers: the last op keeps the
value it computed, the replacement carries the edited value, the edited
module (and every enclosing call whose output it was) reports the
replacement, and a submodule whose output is that same last op keeps
reporting the pre-edit value. The replay engine used to commit a
post-capture ``trace.do(tl.module(addr), edit)`` IN PLACE at the module's
last op, so an inner submodule sharing that op reported the edited value.

These helpers give the replay path the capture-time shape: the hook still
fires at the last op (that is where the replay engine recomputes the
module's output), and after the push the edited value moves into a new
replacement op spliced after the last op, the last op gets its pre-edit
state back, and the consumers, module calls, lookup keys and the sticky
hook spec are re-pointed at the replacement. Later pushes pass the parent's
value through the replacement (``_replay_sites._thread_passthrough_site``)
and fire the module's hooks there, so the replacement stays the module's
output op across pushes.

Labels are not renumbered after capture: the replacement takes the next
unused position number, every existing op keeps its label.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from .._data_classes._site_key import call_instance_id, render_site_key, site_axis
from .._selection.target import TargetSpec
from ._replay_values import rollback_snapshot

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .types import HookSpec

# This module stays outside the package's import cycle (scripts/checks/layering.py):
# the record classes it rewrites are matched by their fields, the Layer class is taken
# from the trace's own layer log, and the caller drops the trace's accessor caches.

REPLACEMENT_TYPE = "interventionreplacement"
REPLACEMENT_FUNC_NAME = "intervention_replacement"
#: Hook-spec metadata key naming the ``tl.module`` address a do() plan item came from.
MODULE_BOUNDARY_KEY = "module_boundary_address"

# Fields the clone replaces at once, so the copy may share them with the source.
_SHALLOW_REPLACED = frozenset(
    {
        "edits",
        "lookup_keys",
        "children",
        "parents",
        "parent_arg_positions",
        "_edge_uses",
        "modules",
        "module_call_stack",
        "output_of_modules",
        "output_of_module_calls",
        "input_to_module_calls",
        "equivalent_ops",
        "parent_params",
        "parent_param_ops",
        "param_shapes",
        "non_tensor_pos_args",
        "non_tensor_kwargs",
        "input_shapes",
        "input_dtypes",
        "input_memory",
        # Tensor-bearing or object-valued fields: the replacement resets or
        # shares them, and a deep copy of a non-leaf tensor raises.
        "grad",
        "grad_fn",
        "module",
        "atomic_module_call",
        "input_activations",
        "unattributed_tensor_args",
        "dropped_edge_tensor_args",
        "func_non_tensor_args",
        "_grad_records",
        "_param_logs",
        "_param_barcodes",
        "recurrent_ops",
        "edge_substitutions",
        "edge_replacement_stamps",
        "activation_transform",
        "transform_config",
        "func_config",
        "annotations",
    }
)


def module_boundary_address(resolved: Any) -> str | None:
    """Return the ``tl.module`` address a resolved selection came from, or ``None``."""

    selector = getattr(resolved, "selector", None)
    if getattr(selector, "selector_kind", None) != "module":
        return None
    value = getattr(selector, "selector_value", None)
    return value if isinstance(value, str) else None


def tag_module_boundary_items(resolved: Any, plan: list[dict[str, Any]]) -> None:
    """Stamp a ``tl.module`` selection plan's interior items with the module address."""

    address = module_boundary_address(resolved)
    if address is None:
        return
    for item in plan:
        if not item["is_leaf"]:
            item[MODULE_BOUNDARY_KEY] = address


def snapshot_module_boundary_sites(
    trace: Trace, specs: Iterable[HookSpec]
) -> list[tuple[HookSpec, str, Op, dict[str, Any]]]:
    """Return ``(spec, address, site, pre-push snapshot)`` rows for a push's module hooks.

    A hook attached straight from ``do(tl.module(addr), edit)`` carries the
    module selector as its target; one lowered from a selection plan carries
    a label target plus the ``MODULE_BOUNDARY_KEY`` metadata. Either way the
    sites are what the engine itself resolves the target to.
    """

    boundary: list[tuple[HookSpec, str, Op, dict[str, Any]]] = []
    for spec in specs:
        address = _module_boundary_of(spec)
        if address is None:
            continue
        sites = trace._resolve_sites(
            spec.site_target, strict=False, max_fanout=len(trace._layer_list)
        )
        for site in sites:
            boundary.append((spec, address, site, rollback_snapshot(site)))
    return boundary


def _module_boundary_of(spec: HookSpec) -> str | None:
    """The module address a sticky hook spec edits the output of, or ``None``."""

    address = (spec.metadata or {}).get(MODULE_BOUNDARY_KEY)
    if address is None and spec.site_target.selector_kind == "module":
        address = spec.site_target.selector_value
    return str(address) if address is not None else None


def materialize_module_replacements(
    trace: Trace, boundary: Iterable[tuple[HookSpec, str, Op, dict[str, Any]]]
) -> list[Op]:
    """Move each module-boundary push result into its own replacement op.

    Parameters
    ----------
    trace:
        Trace the push just mutated in place.
    boundary:
        The ``snapshot_module_boundary_sites`` rows taken before the push.

    Returns
    -------
    list[Op]
        The replacement ops minted, in push order. When non-empty the caller
        drops the trace's op, layer and module-call accessor caches.
    """

    minted: list[Op] = []
    for spec, address, site, before in boundary:
        if site.out is before["out"] or not isinstance(site.out, torch.Tensor):
            continue  # the push did not commit at this site: nothing to move
        moved_calls = _moved_call_labels(site, address)
        if not moved_calls:
            continue  # not that module's output op: the in-place record stands
        replacement = _mint_replacement(trace, site, before, moved_calls)
        _restore_last_op(site, before, replacement, moved_calls)
        _rewire_consumers(trace, site, replacement)
        _splice_into_containers(trace, site, replacement, moved_calls)
        _repoint_module_calls(trace, site, replacement, moved_calls)
        if spec.site_target.selector_kind == "label":  # a module target now resolves to R
            spec.site_target = TargetSpec(
                "label",
                replacement.layer_label,
                strict=spec.site_target.strict,
                metadata=dict(spec.site_target.metadata),
            )
        minted.append(replacement)
    return minted


def _call_address(call_label: str) -> str:
    return call_label.rsplit(":", 1)[0]


def _moved_call_labels(site: Op, address: str) -> list[str]:
    """Return the module calls whose output moves to the replacement.

    The selected module's call plus every enclosing call that reported the
    same op as its output; a call INSIDE the selected module keeps the
    module's last op (it computed the pre-edit value).
    """

    calls = list(getattr(site, "output_of_module_calls", None) or ())
    if address in calls:
        selected = _call_address(address)
    elif address in {_call_address(call) for call in calls}:
        selected = address
    else:
        return []
    return [call for call in calls if not _call_address(call).startswith(selected + ".")]


def _is_inner_call(call_label: str, moved_calls: list[str]) -> bool:
    """Whether ``call_label`` lies inside the innermost moved module call."""

    innermost = max((_call_address(call) for call in moved_calls), key=len)
    return _call_address(call_label).startswith(innermost + ".")


def _next_type_index(trace: Trace) -> int:
    return 1 + sum(1 for op in trace._layer_list if op.type == REPLACEMENT_TYPE)


def _mint_replacement(trace: Trace, site: Op, before: dict[str, Any], moved_calls: list[str]) -> Op:
    """Clone the edited last op into the replacement op (edited value, no callable)."""

    replacement = site._copy_with_shallow_fields(_SHALLOW_REPLACED, _store=None)
    modules = [m for m in (site.modules or ()) if not _is_inner_call(m, moved_calls)]
    fields: dict[str, Any] = {
        **_identity_fields(trace, site, before),
        **_callable_free_fields(site),
        **_wiring_fields(site, before, moved_calls, modules),
        "site_key": _replacement_site_key(trace, modules),
    }
    for name, value in fields.items():
        replacement._internal_set(name, value)
    return replacement


def _identity_fields(trace: Trace, site: Op, before: dict[str, Any]) -> dict[str, Any]:
    """Labels, indices and edit provenance of the replacement op."""

    position = len(trace._layer_list) + 1
    type_index = _next_type_index(trace)
    layer_label = f"{REPLACEMENT_TYPE}_{type_index}_{position}"
    return {
        "_label_raw": f"{layer_label}_raw",
        "_layer_label_raw": f"{layer_label}_raw",
        "label": f"{layer_label}:1",
        "layer_label": layer_label,
        "short_label": f"{REPLACEMENT_TYPE}_{type_index}:1",
        "short_layer_label": f"{REPLACEMENT_TYPE}_{type_index}",
        "type": REPLACEMENT_TYPE,
        "type_index": type_index,
        "pass_number": 1,
        "num_passes": 1,
        "raw_index": 1 + max((op.raw_index or 0) for op in trace._layer_list),
        "ordinal_index": position,
        "lookup_keys": [],
        "edits": [r for r in site.edits if not any(r is o for o in before["edits"])],
        "is_edited": True,
        "equivalence_class": f"{REPLACEMENT_TYPE}::{site.equivalence_class}",
        "equivalent_ops": [],
        "is_input": False,
        "is_output": False,
        "is_final_output": False,
        "io_role": None,
        "grad": None,
        "has_grad": False,
        "_grad_records": [],
        "_param_logs": [],
        "_param_barcodes": [],
    }


def _callable_free_fields(site: Op) -> dict[str, Any]:
    """Fields of an op that has no callable, arguments or parameters of its own."""

    return {
        "func": None,
        "func_id": None,
        "func_call_id": None,
        "func_name": REPLACEMENT_FUNC_NAME,
        "func_qualname": None,
        "var_names": [],
        "flops_forward": None,
        "flops_backward": None,
        "arg_names": (),
        "num_args_total": 0,
        "num_pos_args": 0,
        "num_kwargs": 0,
        "non_tensor_pos_args": [],
        "non_tensor_kwargs": {},
        "saved_args": None,
        "saved_kwargs": None,
        "has_saved_args": False,
        "args_template": None,
        "kwargs_template": None,
        "is_inplace": False,
        "grad_fn_class_name": None,
        "grad_fn_class_qualname": None,
        "grad_fn_object_id": None,
        "grad_fn_handle": getattr(site.out, "grad_fn", None),
        "grad_fn": None,
        "func_non_tensor_args": None,
        "unattributed_tensor_args": (),
        "dropped_edge_tensor_args": (),
        "atomic_module_call": None,
        "edge_substitutions": {},
        "edge_replacement_stamps": {},
        "annotations": dict(site.annotations or {}),
        "in_multi_output": False,
        "multi_output_index": None,
        "multi_output_name": None,
        "container_path": (),
        "container_spec": None,
        "parent_params": [],
        "parent_param_ops": [],
        "param_shapes": [],
        "num_params": 0,
        "num_params_trainable": 0,
        "num_params_frozen": 0,
        "is_atomic_module": False,
    }


def _wiring_fields(
    site: Op, before: dict[str, Any], moved_calls: list[str], modules: list[str]
) -> dict[str, Any]:
    """Graph edges and module membership: parent is the last op, children are its consumers."""

    return {
        "parents": (site.label,),
        "parent_arg_positions": {"args": {}, "kwargs": {}},
        "_edge_uses": [],
        "children": tuple(site.children or ()),
        "has_children": bool(site.children),
        "out_versions_by_child": dict(site.out_versions_by_child or {}),
        "num_inputs": 1,
        "input_shapes": [before["shape"]],
        "input_dtypes": [before["dtype"]],
        "input_memory": [before["activation_memory"]],
        "module": modules[-1] if modules else None,
        "modules": modules,
        "module_call_stack": [
            m for m in (site.module_call_stack or ()) if not _is_inner_call(m, moved_calls)
        ],
        "output_of_modules": [_call_address(call) for call in moved_calls],
        "output_of_module_calls": list(moved_calls),
        "is_module_output": True,
        "input_to_module_calls": list(site.input_to_module_calls or ()),
    }


def _replacement_site_key(trace: Trace, modules: list[str]) -> str | None:
    """Mint the structural site key the next replacement at this module position gets."""

    if not any(getattr(op, "site_key", None) for op in trace._layer_list):
        return None
    instance = call_instance_id(modules)
    axis = site_axis(modules)
    ordinal = 1 + sum(
        1
        for op in trace._layer_list
        if op.type == REPLACEMENT_TYPE
        and call_instance_id(op.modules or ()) == instance
        and site_axis(op.modules or ()) == axis
    )
    return render_site_key(axis, REPLACEMENT_TYPE, None, ordinal)


def _restore_last_op(
    site: Op, before: dict[str, Any], replacement: Op, moved_calls: list[str]
) -> None:
    """Give the module's last op its pre-push state back and hand its outputs over."""

    for name, value in before.items():
        site._internal_set(name, value)
    moved = set(moved_calls)
    moved_addresses = {_call_address(call) for call in moved_calls}
    remaining = [call for call in (site.output_of_module_calls or ()) if call not in moved]
    site._internal_set("children", (replacement.label,))
    site._internal_set("has_children", True)
    site._internal_set("out_versions_by_child", {})
    site._internal_set("has_out_variations", False)
    site._internal_set("output_of_module_calls", remaining)
    site._internal_set(
        "output_of_modules",
        [a for a in (site.output_of_modules or ()) if a not in moved_addresses],
    )
    site._internal_set("is_module_output", bool(remaining))
    site._internal_set("input_to_module_calls", [])


def _spellings(site: Op) -> set[str]:
    return {
        s
        for s in (site.label, site.layer_label, site._label_raw, site._layer_label_raw)
        if isinstance(s, str)
    }


def _is_record(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _rewrite_refs(value: Any, old: set[str], new: str) -> Any:
    """Return ``value`` with every parent reference to ``old`` pointing at ``new``.

    A ``ParentRef`` (any record with a ``parent_label``) is re-pointed; a
    ``CapturedArgTemplate`` (a record with ``args`` and ``kwargs``) is rewritten
    inside; tuples, lists and dicts recurse.
    """

    if _is_record(value) and hasattr(value, "parent_label"):
        return dataclasses.replace(value, parent_label=new) if value.parent_label in old else value
    if _is_record(value) and hasattr(value, "args") and hasattr(value, "kwargs"):
        return dataclasses.replace(
            value,
            args=tuple(_rewrite_refs(a, old, new) for a in value.args),
            kwargs=tuple((k, _rewrite_refs(v, old, new)) for k, v in value.kwargs),
        )
    if isinstance(value, tuple):
        return tuple(_rewrite_refs(v, old, new) for v in value)
    if isinstance(value, list):
        return [_rewrite_refs(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_refs(v, old, new) for k, v in value.items()}
    return value


def _rewire_consumers(trace: Trace, site: Op, replacement: Op) -> None:
    """Point every consumer of the module's last op at the replacement."""

    old = _spellings(site)
    new = replacement.label
    for child_label in replacement.children or ():
        child = trace._layer_dict_all_keys.get(child_label)
        if child is None:
            continue
        child._internal_set("parents", tuple(new if p in old else p for p in (child.parents or ())))
        positions = child.parent_arg_positions or {}
        child._internal_set(
            "parent_arg_positions",
            {
                domain: {
                    k: (new if v in old else v) for k, v in (positions.get(domain) or {}).items()
                }
                for domain in ("args", "kwargs")
            },
        )
        child._internal_set("args_template", _rewrite_refs(child.args_template, old, new))
        child._internal_set("kwargs_template", _rewrite_refs(child.kwargs_template, old, new))
        child._internal_set(
            "_edge_uses",
            [_rewrite_refs(r, old, new) for r in (child._edge_uses or ())],
        )


def _splice_into_containers(
    trace: Trace, site: Op, replacement: Op, moved_calls: list[str]
) -> None:
    """Insert the replacement after the last op in every label-keyed trace container."""

    index = next(i for i, op in enumerate(trace._layer_list) if op is site)
    trace._layer_list.insert(index + 1, replacement)
    if site.label in trace.op_labels:
        trace.op_labels.insert(trace.op_labels.index(site.label) + 1, replacement.label)
    keys = [replacement.label, replacement.layer_label, replacement._label_raw]
    all_keys = trace._layer_dict_all_keys
    for call in moved_calls:
        if all_keys.get(call) is site:
            keys.append(call)
        address = _call_address(call)
        if all_keys.get(address) is site:
            keys.append(address)
    trace._layer_dict_main_keys[replacement.label] = replacement
    for key in keys:
        all_keys[key] = replacement
        trace._lookup_keys_to_layer_num_dict[key] = replacement.raw_index
    trace._layer_num_to_lookup_keys_dict[replacement.raw_index] = list(keys)
    replacement._internal_set("lookup_keys", list(keys))
    taken = set(keys)
    site._internal_set("lookup_keys", [k for k in (site.lookup_keys or ()) if k not in taken])
    site_keys = trace._layer_num_to_lookup_keys_dict.get(site.raw_index)
    if site_keys is not None:
        site_keys[:] = [k for k in site_keys if k not in taken]
    trace._layer_num_calls[replacement.layer_label] = 1
    layer = type(trace._layer_logs[site.layer_label])(replacement)  # the Layer class
    layer.ops[1] = replacement
    logs: OrderedDict[str, Any] = OrderedDict()
    for label, existing in trace._layer_logs.items():
        logs[label] = existing
        if label == site.layer_label:
            logs[replacement.layer_label] = layer
    trace._layer_logs = logs
    trace.output_layers = [
        replacement.layer_label if label == site.layer_label else label
        for label in trace.output_layers
    ]


def _repoint_module_calls(trace: Trace, site: Op, replacement: Op, moved_calls: list[str]) -> None:
    """Make the moved module calls (and the consumers' calls) report the replacement."""

    for module in trace._module_logs:
        touched = False
        for call in module.calls.values():
            if _is_inner_call(call.label, moved_calls) and call.label not in moved_calls:
                continue  # inside the edited module: its output stays the last op
            if site.label in call.output_ops:
                call.output_ops = [
                    replacement.label if label == site.label else label for label in call.output_ops
                ]
                call.output_layers = [
                    replacement.layer_label if label == site.layer_label else label
                    for label in call.output_layers
                ]
                touched = True
            if site.label in call.ops and replacement.label not in call.ops:
                call.ops = [*call.ops, replacement.label]
            if site.label in call.input_ops:
                call.input_ops = [
                    replacement.label if label == site.label else label for label in call.input_ops
                ]
                call.input_layers = [
                    replacement.layer_label if label == site.layer_label else label
                    for label in call.input_layers
                ]
                touched = True
        if touched:
            module._sync_boundary_fields_from_calls()
