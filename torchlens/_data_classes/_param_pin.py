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

State streams (pickle, deep copy, ``plain_state``) see plain lists and
dictionaries of records, exactly as before: ``plain_form`` gives the plain
form of a row-key container. One container reached from two places in one
stream (a layer reads its first op's parameter list) must give one plain
object, or pickle would write it twice, so a trace's stream opens a session
that remembers each plain form until the stream's own state is dropped.
"""

from __future__ import annotations

import weakref
from typing import Any

from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from ._op_row_containers import adopt_op_containers
from ._param_rows import (
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
_list_iter = list.__iter__
_FIELD = "_param_logs"
_BY_MODULE = "_param_logs_by_module"

#: The ``ParamAccessor`` class, bound on first use so this module imports no record class.
_ACCESSOR: list[type] = []


def _accessor_class() -> type:
    """Return the ``ParamAccessor`` class."""

    if not _ACCESSOR:
        from .param import ParamAccessor

        _ACCESSOR.append(ParamAccessor)
    return _ACCESSOR[0]


def pin_params(trace: Any) -> None:
    """Turn ``trace``'s ``Param`` containers into row keys and name it the views' owner."""

    core = trace.__dict__.get("_trace_core")
    rows = param_rows_of(core) if core is not None else None
    if rows is None:
        return
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
    if core.ops is not None and core.ops.__class__.__name__ == "OpRowStore":
        _pin_op_cells(core.ops, rows, memo)
    _pin_layer_states(state.get("_layer_logs"), rows, memo)
    for entry in list(core._views.cells.values()):
        if entry.__class__ is tuple and len(entry) == 3:
            built = entry[2]
            if built.__class__ is _accessor_class():
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

    if kind != PARAM_KIND or accessor.__class__ is not _accessor_class():
        return accessor
    core = trace.__dict__.get("_trace_core")
    rows = param_rows_of(core) if core is not None else None
    return accessor if rows is None else own_accessor(rows, accessor)


def _pin_op_cells(store: Any, rows: ParamRows, memo: dict[int, Any]) -> None:
    """Replace each op's non-empty parameter list with a ``ParamSequence``.

    The cell is released before the new value is written, so a sealed store
    lets go of the plain list it kept in its column.
    """

    fid = store.layout.fid_by_name.get(_FIELD)
    if fid is None:
        return
    cell_get, cell_del, cell_set = store.cell_get, store.cell_del, store.cell_set
    for row in range(len(store)):
        value = cell_get(row, fid, True)
        if value.__class__ is list and value:
            converted = as_sequence(rows, value, memo)
            cell_del(row, fid)
            cell_set(row, fid, converted)


def _pin_layer_states(layer_logs: Any, rows: ParamRows, memo: dict[int, Any]) -> None:
    """Replace each layer state's own parameter list with a ``ParamSequence``."""

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

#: Open stream sessions: each maps ``id(container)`` to ``(container, plain form)``.
_SESSIONS: list[dict[int, tuple[Any, Any]]] = []


def _close_session(memo: dict[int, tuple[Any, Any]]) -> None:
    """End one stream session and drop the plain forms it kept."""

    for index, open_memo in enumerate(_SESSIONS):
        if open_memo is memo:
            del _SESSIONS[index]
            break
    memo.clear()


def plain_form(value: Any) -> Any:
    """Return the plain list or dictionary a row-key container stands in for.

    Inside an open stream session one container always gives the same plain object.
    """

    if not _SESSIONS:
        return value.__tl_plain__()
    memo = _SESSIONS[-1]
    entry = memo.get(id(value))
    if entry is None or entry[0] is not value:
        entry = memo[id(value)] = (value, value.__tl_plain__())
    return entry[1]


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
        memo: dict[int, tuple[Any, Any]] = {}
        _SESSIONS.append(memo)
        twin = _new(cls)
        twin.__dict__ = {
            name: plain_form(value) if value.__class__ in ROW_FORMS else value
            for name, value in accessor.__dict__.items()
        }
        weakref.finalize(twin, _close_session, memo)
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


#: Per row-key sequence: ``key -> position`` (weakly keyed; verified before use).
_POSITIONS: weakref.WeakKeyDictionary[Any, dict[Any, int]] = weakref.WeakKeyDictionary()


def param_index(accessor: Any, param: Any) -> int:
    """Return ``param``'s 0-based position in ``accessor`` (``Param.index``)."""

    items = getattr(accessor, "_list", None)
    if items.__class__ is ParamSequence:
        key = items._owner.key_of(param)
        positions = _POSITIONS.get(items)
        if positions is None or len(positions) != len(items):
            positions = {}
            for position, row in enumerate(_list_iter(items)):
                positions.setdefault(row, position)
            _POSITIONS[items] = positions
        index = positions.get(key)
        if index is not None and list.__getitem__(items, index) == key:
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
