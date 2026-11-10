"""The trace's buffer accessors name their Buffers by row; the doors hand out owned Buffers.

A ``Buffer`` is a row facade: every stored field lives in one row of the
trace's ``buffer`` kind table, so the object can be built again from what the
trace owns. The trace's ``BufferAccessor`` (and each module's scoped one)
therefore keeps the row of each address, the kind table and the weak view
table, and builds each Buffer when read (``record_views.view_at``). The trace
holds no Buffer object, so a Buffer can hold its trace: a Buffer handed out by
a built trace keeps it in its ``_owner`` slot, and
``tl.trace(model, x).buffers["bn.running_mean"].final_value`` works.

The accessor the trace keeps cannot hold the trace (it would hold itself), so
``trace.buffers`` and ``module.buffers`` hand out a second accessor that
shares the stored one's state and keeps the trace, cached weakly on the stored
one, the same pattern as ``trace.layers``. An accessor that holds its records
(built directly, restored from pickle, or on a trace without a core) is handed
out unchanged.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._trace_core.record_views import kind_table, rows_in
from .buffer import BufferAccessor

if TYPE_CHECKING:
    from .buffer import Buffer
    from .module import Module
    from .trace import Trace

#: The ``TraceCore.kind_rows`` key of the Buffer table.
KIND = "buffer"

_new = object.__new__
_ref = weakref.ref


def keyed_accessor(rows: dict[str, int], core: Any, trace: Trace | None) -> BufferAccessor:
    """Return an accessor that names its Buffers by row in ``core``'s buffer table.

    Parameters
    ----------
    rows:
        Address to row in ``core.kind_rows["buffer"]``, in accessor order.
    core:
        Trace core holding the buffer table and its weak view table.
    trace:
        Trace the accessor reads for (named weakly), or ``None``.

    Returns
    -------
    BufferAccessor
        The stored accessor (it holds no trace and no Buffer).
    """

    accessor = _new(BufferAccessor)
    accessor._owner = None
    accessor._handed = None
    state = accessor.__dict__
    state["_rows"] = rows
    state["_store"] = core.kind_rows[KIND]
    state["_views"] = kind_table(core, KIND)
    state["_core_ref"] = _ref(core)
    state["_source_ref"] = _ref(trace) if trace is not None else None
    return accessor


def trace_accessor(buffer_dict: dict[str, Buffer], trace: Trace) -> BufferAccessor:
    """Return the accessor a trace keeps for ``buffer_dict``: keyed by row when every Buffer has one.

    An empty accessor holds no Buffer, so it stays the plain one (a fork copies it as before).
    """

    core = trace.__dict__.get("_trace_core")
    rows = rows_in(buffer_dict.values(), core, KIND) if buffer_dict else None
    if rows is None:
        return BufferAccessor(buffer_dict, source_trace=trace)
    return keyed_accessor(dict(zip(buffer_dict, rows, strict=True)), core, trace)


def handed(stored: Any, trace: Any) -> Any:
    """Return the accessor a door hands out for ``stored``: one that shares its state and keeps ``trace``."""

    if trace is None or stored is None or "_rows" not in stored.__dict__:
        return stored
    reference = stored._handed
    twin = reference() if reference is not None else None
    if twin is not None and twin._owner is trace:
        return twin
    twin = _new(BufferAccessor)
    twin.__dict__ = stored.__dict__
    twin._owner = trace
    twin._handed = None
    stored._handed = _ref(twin)
    return twin


def trace_buffers(trace: Trace) -> BufferAccessor:
    """Body of ``Trace.buffers``: the stored accessor, handed out so it keeps the trace."""

    return handed(trace.__dict__.get("_buffer_accessor"), trace)


def _in_scope(module_address: str, address: str) -> bool:
    """Return whether the buffer at ``address`` belongs to the module at ``module_address``."""

    return (
        module_address in {"", "self"}
        or address == module_address
        or address.startswith(f"{module_address}.")
    )


def module_buffers(module: Module) -> BufferAccessor:
    """Body of ``Module.buffers``: the module's scoped accessor, built once, handed out owned."""

    cached = module._buffer_accessor
    if cached is None:
        trace = module._source_trace
        parent = None if trace is None else trace._buffer_accessor
        if parent is None:
            module._buffer_accessor = BufferAccessor({})
            return module._buffer_accessor
        address = module.address
        parent_state = parent.__dict__
        rows = parent_state.get("_rows")
        if rows is None:
            scoped = {a: b for a, b in parent._dict.items() if _in_scope(address, a)}
            cached = BufferAccessor(scoped, source_trace=trace)
        else:
            cached = _new(BufferAccessor)
            cached._owner = None
            cached._handed = None
            cached.__dict__.update(parent_state)
            cached.__dict__["_rows"] = {a: r for a, r in rows.items() if _in_scope(address, a)}
            cached.__dict__["_source_ref"] = _ref(trace)
        module._buffer_accessor = cached
    reference = cached.__dict__.get("_source_ref")
    return handed(cached, reference() if reference is not None else None)


def bind_fork_buffers(
    parent: Trace, fork: Trace, fork_core: Any, copy_value: Callable[[Any], Any]
) -> None:
    """Give the fork its buffer accessor: keyed by the same rows in the fork's buffer table.

    The fork's buffer table is a copy-on-write view over the parent's, with the
    same rows, so the fork's accessor names the same rows there. An empty
    accessor is built new, with the state a copy of it would have. A parent
    accessor that holds its records was copied by the field pass already. As
    that copy did, the fork's accessor names no source trace.
    """

    try:
        stored = parent.__dict__["_buffer_accessor"]
    except KeyError:
        return
    if stored is None:
        return
    state = stored.__dict__
    if "_rows" not in state:
        if not state["_dict"]:
            empty = _new(BufferAccessor)
            copied = empty.__dict__
            copied["_dict"], copied["_list"] = {}, []
            if "_source_ref" in state:
                copied["_source_ref"] = None
            fork.__dict__["_buffer_accessor"] = empty
        return
    if fork_core is None or fork_core.kind_rows.get(KIND) is None:
        fork.__dict__["_buffer_accessor"] = copy_value(stored)
        return
    fork.__dict__["_buffer_accessor"] = keyed_accessor(dict(state["_rows"]), fork_core, None)
