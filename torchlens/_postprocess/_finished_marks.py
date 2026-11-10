"""Step 17 body: mark every final op finished and bind template display keys.

The step sets ``_tracing_finished`` on each op of the trace's main lookup and
caches, on every ``ParentRef`` of the op's call templates, the display key of
the op it refers to. Two bodies give the same stored result. The column body
writes the flag into the raw row of a plain building ``OpRowStore`` and binds
the keys of a flat positional template inline. Any other template takes
:func:`_bind`, which walks it node for node as :func:`bind_template_keys` does
and fails open the same way. Both read a parent's display key from its raw
cells when those hold exactly what the facade would hand back, and through
:func:`display_key` otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._data_classes._op_columns import column_rows_by_label
from .._trace_core.cell_states import LazyCell
from .._trace_core.op_store import OpRowStore
from .._utils._display_constructor import bind_template_keys
from .._utils._display_keys import display_key
from .._utils._record_group_table import op_attr
from .._utils.fail_open import fail_open

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

#: Exact classes of template members the binding walk does nothing with.
_LEAF_CLASSES = frozenset({str, int, float, bool, type(None)})

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

    lookup = column_rows_by_label(ops) if isinstance(records, dict) else None
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
    inline walk covers a flat positional template; it stops at the first member
    it does not model and hands the op to :func:`_bind_both`, which rebinds the
    same keys in the same order before going on.
    """

    # Complexity: one hot loop with field access inlined on purpose; a call per
    # op or per template member is the cost this body removes.
    raw_rows, rows, fids = plan
    flag_fid, args_fid, kwargs_fid = fids[:3]
    for label, op in ops.items():
        cells = raw_rows[rows[label]]
        cells[flag_fid] = True
        template = cells[args_fid]
        if template.__class__ is LazyCell and not template.make:
            template = template.source
        keywords = cells[kwargs_fid]
        if keywords.__class__ is LazyCell and not keywords.make:
            keywords = keywords.source
        flat = (
            keywords is None
            and template.__class__.__name__ == "CapturedArgTemplate"
            and template.args.__class__ is tuple
            and template.kwargs == ()
        )
        general = not flat and (template is not None or keywords is not None)
        if flat:
            for member in template.args:
                kind = member.__class__
                if kind.__name__ != "ParentRef":
                    if kind.__name__ in _INERT_COMPONENTS and kind.__base__ is object:
                        continue
                    general = True
                    break
                parent = member.parent_label
                if parent.__class__ is not str or parent not in rows:
                    general = True
                    break
                _object_setattr(member, "_display_key", _key_of(parent, records[parent], plan))
        if general:
            _bind_both(op, template, keywords, records, plan)


def _bind_both(op: Any, template: Any, keywords: Any, records: Any, plan: Any) -> None:
    """Bind one op's two templates in the facade body's order.

    ``template`` and ``keywords`` are the op's raw template cells. A cell that
    is neither ``None`` nor a captured template is read through the facade,
    which hydrates it (and raises for an unset cell) as the facade body does.
    """

    if template is not None and template.__class__.__name__ != "CapturedArgTemplate":
        template = op._call_args_template
    _bind(template, records, plan)
    if keywords is not None and keywords.__class__.__name__ != "CapturedArgTemplate":
        keywords = op._call_kwargs_template
    _bind(keywords, records, plan)


def _bind(template: Any, records: Any, plan: Any) -> None:
    """Bind one template's keys; the fail-open twin of :func:`bind_template_keys`."""

    if template is not None:
        fail_open(lambda: _bind_keys(template, records, plan), _ignore)


def _ignore(_error: Exception) -> None:
    """Degraded form of a failed binding walk: the remaining keys stay unbound."""


def _bind_keys(value: Any, records: Any, plan: Any) -> None:
    """Walk a template and cache each parent reference's display key in place.

    Same node order and same class tests as the walk behind
    :func:`bind_template_keys`; members that walk does nothing with are
    skipped here without a call.
    """

    # Complexity: the branch structure is the mirrored walk's, kept one to one.
    name = type(value).__name__
    if name == "ParentRef":
        label = value.parent_label
        record = records.get(label)
        if record is not None:
            _object_setattr(value, "_display_key", _key_of(label, record, plan))
    elif name == "CapturedArgTemplate":
        _bind_keys(value.args, records, plan)
        _bind_keys(value.kwargs, records, plan)
    elif isinstance(value, (tuple, list)):
        for member in value:
            kind = type(member)
            inert = kind.__name__ in _INERT_COMPONENTS and kind.__base__ is object
            if kind not in _LEAF_CLASSES and not inert:
                _bind_keys(member, records, plan)
    elif isinstance(value, dict):
        for member in value.values():
            _bind_keys(member, records, plan)


def _key_of(label: Any, record: Any, plan: tuple[Any, Any, Any]) -> str:
    """Return ``display_key(record)`` for the record the lookups hold under ``label``.

    When ``label`` is a main-lookup label, ``record`` is that op (``_column_plan``
    checked it) and the key comes from the op's raw cells if they hold a
    non-empty ``str`` label and plain pass counts; any other case reads the
    record through :func:`display_key`.
    """

    raw_rows, rows, fids = plan
    if label.__class__ is not str or label not in rows:
        return display_key(record)
    cells = raw_rows[rows[label]]
    short = cells[fids[3]]
    passes = cells[fids[4]]
    if passes.__class__ is LazyCell and not passes.make:
        passes = passes.source
    number = cells[fids[5]]
    if number.__class__ is LazyCell and not number.make:
        number = number.source
    if short.__class__ is not str or not short or passes.__class__ is not int:
        return display_key(record)
    if passes <= 1 or number is None:
        return str(short)
    if number.__class__ is int:
        return f"{short}:{number}"
    return display_key(record)
