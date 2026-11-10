"""The ``facets`` door of ``Op``, ``Module`` and ``ModuleCall``.

A record keeps its facet view in its own state, and the view names the record
by row, so the stored view cannot hold the record or the trace: the trace
would hold itself. In ``tl.trace(model, x)["relu_1"].facets["out"]`` the op
is a temporary; it is gone, and the trace with it, before the facet is read.

So a read of ``record.facets`` on a record that holds its trace hands out a
second view object over the same state that holds the record, exactly as
``Layer.ops`` hands out a second accessor. The second view is cached weakly on
the stored one: ``record.facets is record.facets`` while anyone keeps it, the
facets computed through either are shared, and a trace nobody holds is still
freed by reference counting.

A record with no trace to hold (copied or restored on its own, or read while
the trace is being built) hands out the stored view unchanged.
"""

from __future__ import annotations

import weakref
from typing import Any

_CACHE = "_facets_cache"
_get = object.__getattribute__
_set = object.__setattr__


def owned(stored: Any, record: Any) -> Any:
    """Return the facet view ``record`` hands out for its stored view.

    Parameters
    ----------
    stored:
        The view kept in the record's state.
    record:
        The record the view was read from.

    Returns
    -------
    Any
        A view over the same state that holds ``record`` when the record
        holds a trace, else ``stored``.
    """

    try:
        if _get(record, "_owner") is None:
            return stored
    except AttributeError:
        return stored
    try:
        twin = _get(stored, "_handed")()
        if _get(twin, "_owner") is record:
            return twin
    except (AttributeError, TypeError):
        pass  # none handed out yet, or the one handed out is gone
    twin = object.__new__(stored.__class__)
    _set(twin, "__dict__", _get(stored, "__dict__"))
    _set(twin, "_owner", record)
    _set(stored, "_handed", weakref.ref(twin))
    return twin


def record_facets(record: Any, view_class: Any, subject: Any = None) -> Any:
    """Return the facet view of a ``Module`` or ``ModuleCall``, building it on first read.

    ``view_class`` is ``FacetView`` (passed in, so this module imports
    nothing). ``subject`` maps the record to the object the view describes (a
    ``ModuleCall`` is described through its call-bound module).
    """

    state = record.__dict__
    stored = state.get(_CACHE)
    if stored is None:
        stored = state[_CACHE] = view_class(record if subject is None else subject(record))
    return owned(stored, record)


def op_facets(op: Any, view_class: Any) -> Any:
    """Return the facet view of an ``Op``, building it on first read."""

    stored = op._slot(_CACHE)
    if stored is None:
        stored = view_class(op)
        _set(op, _CACHE, stored)
    return owned(stored, op)
