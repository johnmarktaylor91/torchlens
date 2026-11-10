"""Pin ``Param``: the trace keeps parameter rows, and a ``Param`` handed out keeps the trace.

``pin_params`` runs once a trace is finished (the end of a capture, a fork, a
pickle restore or a file load). It turns every trace container of ``Param``
records into the row-key form of ``_param_rows.py`` and names the trace as the
owner of the views. From then on the trace holds no ``Param``: a record
nobody keeps is dropped, and the next read builds it again from its row.

``handed_params`` is the ``trace.params`` door: the accessor the trace stores
cannot hold the trace (the trace would hold itself), so the door hands out a
second accessor over the same state that does, cached weakly, as
``trace.ops`` does.

State streams (pickle, deep copy, ``plain_state``) see plain lists, tuples
and dictionaries of records, exactly as before: ``plain_form`` gives the plain
form of a row-key container. One container reached from two places in one
stream (a layer reads its first op's parameter tuple) must give one plain
object, or pickle would write it twice, so a trace's stream opens a session
on that trace's resolver that remembers each plain form until the stream's
own state is dropped. A session only ever serves its own trace's containers.
"""

from __future__ import annotations

import weakref
from typing import Any

from .._trace_core.op_store import OpRowStore
from .._trace_core.record_refs import bind_own_children
from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from ._op_row_containers import adopt_op_containers
from ._param_rows import (
    _PARAM,
    PARAM_KIND,
    ROW_FORMS,
    ParamMap,
    ParamRows,
    ParamSequence,
    as_map,
    as_sequence,
    param_rows_of,
)

_new = object.__new__
_FIELD = "_param_logs"
_BY_MODULE = "_param_logs_by_module"

#: The ``ParamAccessor`` class, bound by ``param.py`` at import so this module imports no record class.
_ACCESSOR: list[type] = []


def bind_param_classes(param_class: type, accessor_class: type) -> None:
    """Bind the ``Param`` and ``ParamAccessor`` classes; ``param.py`` calls this once at import."""

    _PARAM[:] = [param_class]
    _ACCESSOR[:] = [accessor_class]


def _accessor_class() -> type:
    """Return the ``ParamAccessor`` class."""

    return _ACCESSOR[0]


def pin_params(trace: Any) -> None:
    """Turn ``trace``'s ``Param`` containers into row keys and name it the views' owner.

    A stream session still open for these rows ends here: a restored trace may
    keep the plain accessor a state walk built, and the session's plain forms
    must not keep the trace alive.
    """

    core = trace.__dict__.get("_trace_core")
    rows = param_rows_of(core) if core is not None else None
    if rows is None:
        return
    if _SESSIONS:
        _close_session(weakref.ref(rows), _SESSIONS.get(rows, {}))
    rows.table.set_owner(trace)
    memo: dict[int, Any] = {}
    state = trace.__dict__
    accessor = state.get(_FIELD)
    if accessor is not None and accessor.__class__ is _accessor_class():
        own_accessor(rows, accessor)
    by_module = state.get(_BY_MODULE)
    if by_module:
        for address, params in list(by_module.items()):
            by_module[address] = as_sequence(rows, params, memo)
    if isinstance(core.ops, OpRowStore):
        _pin_op_cells(core.ops, rows, memo)
    _pin_layer_states(state.get("_layer_logs"), rows, memo)
    cls = _accessor_class()
    for entry in list(core._views.cells.values()):
        if entry.__class__ is tuple and entry[2:] and not entry[3:]:
            built = entry[2]
            if built.__class__ is cls:
                own_accessor(rows, built)


def adopt_and_pin(trace: Any, rebuild_core: Any = None) -> None:
    """Adopt a restored trace's op containers (``adopt_op_containers``), then pin its params."""

    adopt_op_containers(trace, rebuild_core)
    pin_params(trace)


def own_accessor(rows: ParamRows, accessor: Any) -> Any:
    """Give ``accessor`` row-key containers of ``rows``; return it."""

    cells = accessor.__dict__
    cells["_dict"] = as_map(rows, cells.get("_dict"))
    cells["_list"] = as_sequence(rows, cells.get("_list"), {})
    return accessor


def own_children(trace: Any, kind: str, accessor: Any) -> Any:
    """Return a module's child accessor, with row-key containers when its children are params."""

    cls = _ACCESSOR[0] if _ACCESSOR else _accessor_class()
    if kind != PARAM_KIND or accessor.__class__ is not cls:
        return accessor
    core = trace.__dict__.get("_trace_core")
    rows = param_rows_of(core) if core is not None else None
    return accessor if rows is None else own_accessor(rows, accessor)


def _pin_op_cells(store: Any, rows: ParamRows, memo: dict[int, Any]) -> None:
    """Replace each op's non-empty parameter list or tuple with its row-key form.

    The cell is released before the new value is written, so a sealed store
    lets go of the plain list it kept in its column.
    """

    fid = store.layout.fid_by_name.get(_FIELD)
    if fid is None:
        return
    cell_get, cell_del, cell_set = store.cell_get, store.cell_del, store.cell_set
    for row in range(len(store)):
        value = cell_get(row, fid, True)
        if (value.__class__ is list or value.__class__ is tuple) and value:
            converted = as_sequence(rows, value, memo)
            cell_del(row, fid)
            cell_set(row, fid, converted)


def _pin_layer_states(layer_logs: Any, rows: ParamRows, memo: dict[int, Any]) -> None:
    """Replace each layer state's own parameter list or tuple with its row-key form."""

    if not layer_logs:
        return
    for layer_state in list(dict.values(layer_logs)):
        cells = (
            layer_state if layer_state.__class__ is dict else getattr(layer_state, "__dict__", {})
        )
        params = cells.get(_FIELD)
        if params is not None:
            converted = as_sequence(rows, params, memo)
            if converted is not params:
                cells[_FIELD] = converted


def handed_params(trace: Any) -> Any:
    """Return ``trace.params``: an accessor that keeps ``trace`` once its params are rows."""

    stored = trace.__dict__.get(_FIELD)
    cls = _accessor_class()
    if stored.__class__ is not cls or stored.__dict__.get("_list").__class__ is not ParamSequence:
        return stored
    table = stored._list._owner.table
    reference = table.handed
    handed = reference() if reference is not None else None
    if handed is None or handed._owner is not trace or handed.__dict__ is not stored.__dict__:
        handed = _new(cls)
        handed.__dict__ = stored.__dict__
        handed._owner = trace
        table.handed = weakref.ref(handed)
    return handed


# ---------------------------------------------------------------- state streams

#: Open stream sessions by trace resolver: ``id(container) -> (container, keys, plain form)``.
#: Kept off the trace, so no trace holds the plain forms; each entry lives as long
#: as the state of the stream that opened it.
_SESSIONS: weakref.WeakKeyDictionary[ParamRows, dict[int, tuple[Any, Any, Any]]] = (
    weakref.WeakKeyDictionary()
)


def _close_session(rows_ref: Any, memo: dict[int, tuple[Any, Any, Any]]) -> None:
    """End one stream session and drop the plain forms it kept."""

    rows = rows_ref()
    if rows is not None and _SESSIONS.get(rows) is memo:
        del _SESSIONS[rows]
    memo.clear()


def _keys_of(value: Any) -> tuple[Any, ...]:
    """Return the keys a row-key container holds, in order (a snapshot for the session memo)."""

    if isinstance(value, dict):
        return tuple(dict.items(value))
    return tuple(tuple.__iter__(value) if isinstance(value, tuple) else list.__iter__(value))


def plain_form(value: Any) -> Any:
    """Return the plain list, tuple or dictionary a row-key container stands in for.

    While its trace has a stream session open, one container (with the same
    keys) always gives the same plain object.
    """

    memo = _SESSIONS.get(value._owner) if _SESSIONS else None
    if memo is None:
        return value.__tl_plain__()
    keys = _keys_of(value)
    entry = memo.get(id(value))
    if entry is None or entry[0] is not value or entry[1] != keys:
        entry = memo[id(value)] = (value, keys, value.__tl_plain__())
    return entry[2]


def plain_value(value: Any) -> Any:
    """Return ``value``, or its plain form when it is a row-key ``Param`` container."""

    return plain_form(value) if value.__class__ in ROW_FORMS else value


def project_params(state: dict[str, Any]) -> dict[str, Any]:
    """Replace a trace state's ``Param`` containers with plain ones; return ``state``.

    The trace's accessor is replaced by a new accessor over plain containers.
    It lives as long as the state does, and so does the stream session it opens.
    """

    accessor = state.get(_FIELD)
    cls = _accessor_class()
    if accessor.__class__ is cls and accessor.__dict__.get("_list").__class__ is ParamSequence:
        rows = accessor.__dict__["_list"]._owner
        memo: dict[int, tuple[Any, Any, Any]] = {}
        _SESSIONS[rows] = memo
        twin = _new(cls)
        twin.__dict__ = {
            name: plain_form(value) if value.__class__ in ROW_FORMS else value
            for name, value in accessor.__dict__.items()
        }
        weakref.finalize(twin, _close_session, weakref.ref(rows), memo)
        state[_FIELD] = twin
    by_module = state.get(_BY_MODULE)
    if by_module and any(value.__class__ is ParamSequence for value in by_module.values()):
        plain = by_module.copy()
        for address, params in by_module.items():
            plain[address] = plain_value(params)
        state[_BY_MODULE] = plain
    return state


# ---------------------------------------------------------------- keys by row


def param_key(param: Any) -> Any:
    """Return a hashable key naming ``param``: its store and row, or the record itself.

    A record over a param store is named by the store and its row there, so
    views of one row built at different times share the key; a record with no
    store (still being built) is its own key. No key depends on ``id``, so it
    cannot name a different record once the first is freed. Keep the key while
    it is in use: it holds the store or the record.
    """

    cells = param.__dict__
    store = cells.get(CORE_KEY)
    return param if store is None else (store, cells.get(ROW_KEY))


def param_index(accessor: Any, param: Any) -> int:
    """Return ``param``'s 0-based position in ``accessor`` (``Param.index``)."""

    items = getattr(accessor, "_list", None)
    if items.__class__ is ParamSequence:
        key = items._owner.key_of(param)
        index = None if key is None else items._owner.position(items, key)
        if index is not None:
            return index
        raise ValueError(f"{param.address!r} is not in trace.params")
    if not isinstance(items, list):
        return list(accessor).index(param)
    for position, item in enumerate(items):
        if item is param:
            return position
    raise ValueError(f"{param.address!r} is not in trace.params")


# ---------------------------------------------------------------- forks


def fork_copy(value: Any, translator: Any) -> Any:
    """Return the fork's copy of a parent row-key container, or ``None`` before the fork has rows.

    A fork's param store is a view over the parent's with the same rows, so
    the copy has the same keys, read through the fork's views.
    """

    fork_ref = getattr(translator, "_fork_ref", None)
    fork = fork_ref() if fork_ref is not None else None
    core = fork.__dict__.get("_trace_core") if fork is not None else None
    rows = param_rows_of(core) if core is not None else None
    if rows is None:
        return None
    if value.__class__ is ParamMap:
        return as_map(rows, value)
    return rows.forked(value)


bind_own_children(own_children)
