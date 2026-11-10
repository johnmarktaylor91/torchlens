"""Step 17 body: mark every final op finished and bind template display keys.

The step sets ``_tracing_finished`` on each op of the trace's main lookup and
caches, on every ``ParentRef`` of the op's call templates, the display key of
the op it refers to. Two bodies give the same stored result. The column body
writes the flag into the raw row of a plain building ``OpRowStore`` and binds
the keys of a flat positional template from the parent's raw cells, with no
call per field access. Anything else about an op (a pooled or encoded cell, a
nested or keyword template, a label the main lookup does not hold) sends that
op's templates to :func:`bind_template_keys`, the facade body's own walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._data_classes._op_columns import column_rows_by_label
from .._trace_core.cell_states import LazyCell
from .._trace_core.op_store import OpRowStore
from .._utils._display_constructor import bind_template_keys
from .._utils._record_group_table import op_attr

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = ("mark_ops_finished",)

#: Stored fields the column body touches, in the order ``_column_plan`` binds
#: them: the flag, the two templates, then the three display-key inputs.
_FIELDS = (
    "_tracing_finished",
    "args_template",
    "kwargs_template",
    "short_layer_label",
    "num_passes",
    "pass_number",
)

#: Template components the binding walk passes over without touching.
_INERT_COMPONENTS = frozenset({"LiteralTensor", "LiteralValue", "Unsupported"})

_object_setattr = object.__setattr__


def mark_ops_finished(trace: Trace) -> None:
    """Mark the trace's final ops finished and bind their template display keys."""

    ops = trace._layer_dict_main_keys
    records = trace._layer_dict_all_keys
    plan = _column_plan(ops, records)
    if plan is None:
        _mark_ops_finished_facade(ops, records)
    else:
        _mark_plain_rows(ops, records, plan)


def _mark_ops_finished_facade(ops: dict[str, Any], records: Any) -> None:
    """Facade body of :func:`mark_ops_finished` for traces off the column path."""

    for layer_label in ops:
        tensor = ops[layer_label]
        tensor._tracing_finished = True
        bind_template_keys(tensor._call_args_template, records)
        bind_template_keys(tensor._call_kwargs_template, records)


def _column_plan(
    ops: dict[str, Any], records: Any
) -> tuple[list[Any], dict[str, int], tuple[int, ...]] | None:
    """Return ``(raw rows, label -> row, field ids)`` when the column body applies.

    It applies when every op is a plain ``Op`` of one building row store, the
    all-keys lookup resolves each main label to that same op (so a parent
    label found in the main lookup is the record the facade walk would read),
    and each field is one the column view serves under its public name.
    """

    if not isinstance(records, dict):
        return None
    lookup = column_rows_by_label(ops)
    if lookup is None:
        return None
    columns, rows = lookup
    store = columns.store
    if store.__class__ is not OpRowStore or store._rows is None:
        return None
    for label, op in ops.items():
        if label not in records or records[label] is not op:
            return None
    try:
        columns.setter(_FIELDS[0])
        for name in _FIELDS[1:]:
            columns.getter(name)
    except ValueError:
        return None
    if [name for name in _FIELDS[3:] if op_attr(name) != name]:
        return None
    fid_by_name = store.layout.fid_by_name
    return store._rows, rows, tuple([fid_by_name[name] for name in _FIELDS])


def _mark_plain_rows(ops: dict[str, Any], records: Any, plan: tuple[Any, Any, Any]) -> None:
    """Column body of :func:`mark_ops_finished` over ``_column_plan``'s result.

    A default-omitted cell with an immutable default reads as that default. The
    inline walk stops at the first component it does not model and hands the op
    to :func:`bind_template_keys`, which rebinds the same keys in the same order.
    """

    # Complexity: one hot loop with field access inlined on purpose; a call per
    # op or per component is the cost this body removes.
    raw_rows, rows, fids = plan
    flag_fid, args_fid, kwargs_fid, short_fid, passes_fid, number_fid = fids
    for label, op in ops.items():
        cells = raw_rows[rows[label]]
        cells[flag_fid] = True
        template = cells[args_fid]
        if template.__class__ is LazyCell and not template.make:
            template = template.source
        keywords = cells[kwargs_fid]
        if keywords.__class__ is LazyCell and not keywords.make:
            keywords = keywords.source
        if template is None and keywords is None:
            continue
        members: Any = ()
        facade = keywords is not None or template.__class__.__name__ != "CapturedArgTemplate"
        if not facade:
            members = template.args
            facade = members.__class__ is not tuple or template.kwargs != ()
        if not facade:
            for member in members:
                kind = member.__class__
                if kind.__name__ != "ParentRef":
                    if kind.__name__ in _INERT_COMPONENTS and kind.__base__ is object:
                        continue
                    facade = True
                    break
                parent = member.parent_label
                if parent.__class__ is not str or parent not in rows:
                    facade = True
                    break
                parent_cells = raw_rows[rows[parent]]
                short = parent_cells[short_fid]
                passes = parent_cells[passes_fid]
                if passes.__class__ is LazyCell and not passes.make:
                    passes = passes.source
                number = parent_cells[number_fid]
                if number.__class__ is LazyCell and not number.make:
                    number = number.source
                if short.__class__ is not str or not short or passes.__class__ is not int:
                    facade = True
                    break
                if passes > 1 and number.__class__ is int:
                    short = f"{short}:{number}"
                elif passes > 1 and number is not None:
                    facade = True
                    break
                _object_setattr(member, "_display_key", short)
        if facade:
            bind_template_keys(op._call_args_template, records)
            bind_template_keys(op._call_kwargs_template, records)
