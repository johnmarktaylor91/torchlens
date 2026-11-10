"""``Buffer.versions`` read from a Buffer that keeps its trace: a list that keeps the trace too.

The trace keeps one ``OpSequence`` per buffer (built once from the row keys in
the buffer's ``versions`` cell), and that list names the trace weakly: the
trace holds it, so it cannot hold the trace. A Buffer handed out by a built
trace keeps that trace (its ``_owner`` slot), and reading ``versions`` from it
hands out a second ``OpSequence`` over the same keys whose resolver keeps the
trace as well. ``tl.trace(model, x).buffers["bn.running_mean"].versions[-1].out``
then works in one expression, and the version ops it gives keep the trace.

The handed list is kept on the Buffer that handed it out (its ``_versions``
slot), so ``buffer.versions is buffer.versions`` and an in-place edit stays
for as long as that Buffer is held. Nothing the trace holds refers to the
handed list, so a trace nobody holds is still freed by reference counting.
A Buffer with no owner (one read while the trace is built, copied, restored,
or built on its own) reads the trace's list, as before.
"""

from __future__ import annotations

from typing import Any

from .._trace_core.record_refs import VersionsField
from ._op_row_containers import OpRows, OpSequence

_set = object.__setattr__
_list_extend = list.extend
_list_iter = list.__iter__
# The resolver state a held resolver shares with the trace's own one.
_SHARED_SLOTS = ("_core_ref", "_ids", "_last", "_store", "_tables", "_trace_ref")


class HeldOpRows(OpRows):
    """A resolver that shares a trace's op tables and keeps the trace (``_trace``)."""

    __slots__ = ("_trace",)


def _held_rows(shared: OpRows, trace: Any) -> HeldOpRows:
    """Return a resolver over ``shared``'s tables that keeps ``trace``."""

    rows = HeldOpRows.__new__(HeldOpRows)
    for name in _SHARED_SLOTS:
        _set(rows, name, getattr(shared, name))
    rows._containers = []
    rows._trace = trace
    return rows


def handed_versions(record: Any, stored: OpSequence, trace: Any) -> OpSequence:
    """Return the ``versions`` list ``record`` hands out: ``stored``'s ops, keeping ``trace``."""

    kept = getattr(record, "_versions", None)
    if kept is not None and kept[0] is stored and kept[1]._owner._trace is trace:
        return kept[1]  # type: ignore[no-any-return]
    handed = OpSequence.__new__(OpSequence)
    _list_extend(handed, list(_list_iter(stored)))
    handed._owner = _held_rows(stored._owner, trace)
    _set(record, "_versions", (stored, handed))
    return handed


class HandedVersionsField(VersionsField):
    """``Buffer.versions``: the trace's list, or a list that keeps the trace when the Buffer does."""

    __slots__ = ()

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the version ops; from a Buffer that keeps its trace, a list that keeps it too."""

        stored = VersionsField.__get__(self, record, objtype)
        if record is None or stored.__class__ is not OpSequence:
            return stored
        trace = getattr(record, "_owner", None)
        if trace is None:
            return stored
        return handed_versions(record, stored, trace)


def install_handed_versions(cls: type) -> None:
    """Replace ``cls.versions`` (a ``VersionsField``) with a ``HandedVersionsField`` on the same cell."""

    field = cls.__dict__["versions"]
    setattr(cls, "versions", HandedVersionsField(field._name, field._fid))
