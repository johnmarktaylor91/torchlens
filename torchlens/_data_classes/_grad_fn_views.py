"""How a trace holds its backward records: as state, handed out as views.

``GradFn``, ``GradFnCall`` and ``BackwardPass`` records are views over state
(see ``_record_views``). Each place a trace keeps one holds the state and
builds the record on a read:

* ``Trace._grad_fn_logs`` is a ``StateDict`` from the first grad function
  discovered, and ``Trace._backward_pass_logs`` from the first projection;
* a ``GradFn``'s ``calls`` accessor keeps its two containers as state
  (``StateBackedCalls``);
* a ``BackwardPass``'s ``grad_fn_calls`` cell is a ``StateList``;
* an ``Op``'s grad-function link cells hold a ``GradFnKey`` (``OpGradFnLink``).

A record read from its trace keeps the trace, and so does what is read from
the record: ``trace.autograd.grad_fns[0].calls[0]`` and
``trace.autograd.backward_passes[0].grad_fn_calls[0]`` hand back a call that
still has its trace when nothing else holds it.
"""

from __future__ import annotations

import copy
from typing import Any

from .._trace_core.record_rows import RecordCellField, install_record_facade, record_state_restore
from ._accessor_base import Accessor
from ._record_views import (
    TRACE_STATE_FIELDS,
    StateDict,
    StateList,
    hold,
    twin,
    view,
)
from .grad_fn_call import GradFnCall

_set = object.__setattr__

#: The ``GradFn`` class, bound when it installs its facade so this module does not import it.
GRAD_FN: list[type] = []

#: The ``BackwardPass`` class, bound the same way.
BACKWARD_PASS: list[type] = []

_CALL_CONTAINERS = ("_dict", "_list")


def _plain_calls(state: dict[str, Any]) -> dict[str, Any]:
    """Replace the call containers in an accessor ``state`` with plain ones of records."""

    for name in _CALL_CONTAINERS:
        held = state.get(name)
        if isinstance(held, (StateDict, StateList)):
            state[name] = held.plain()
    return state


class StateBackedCalls(Accessor[GradFnCall]):
    """The storage of ``GradFnCallAccessor``: its calls are kept as state.

    ``_dict`` and ``_list`` are state containers, so the accessor a trace
    holds in a ``GradFn`` row keeps no ``GradFnCall``; pickle state and the
    state walk carry the plain containers of records they always did. The
    ``_owner`` slot is set only on the accessor a ``GradFn`` hands out, which
    shares this one's instance dictionary.
    """

    __slots__ = ("_owner",)

    def __setattr__(self, name: str, value: Any) -> None:
        """Store an attribute, keeping a plain call container as state."""

        if name == "_dict" and value.__class__ is dict:
            value = StateDict(GradFnCall, value)
        elif name == "_list" and value.__class__ is list:
            value = StateList(GradFnCall, value)
        _set(self, name, value)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state holding plain containers of the calls."""

        return _plain_calls(super().__getstate__())

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state, keeping the calls as state again."""

        super().__setstate__(state)
        restored = self.__dict__
        self.__tl_state_restore__(
            {name: restored[name] for name in _CALL_CONTAINERS if name in restored}
        )

    def __tl_state_items__(self) -> Any:
        """Yield the live state with plain containers of the calls (state-walk hook)."""

        return iter(_plain_calls(dict(self.__dict__)).items())

    def __tl_state_restore__(self, mapping: dict[str, Any]) -> None:
        """Install a state mapping, keeping the calls as state (state-walk hook)."""

        for name, value in mapping.items():
            setattr(self, name, value)


class _CallsField(RecordCellField):
    """``GradFn.calls``: a ``GradFn`` that keeps its trace hands out an accessor that keeps it too."""

    __slots__ = ()

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the stored accessor, or its owned twin when ``record`` has an owner."""

        if record is None:
            return self
        stored = RecordCellField.__get__(self, record, objtype)
        owner = getattr(record, "_owner", None)
        if owner is None or not isinstance(stored, StateBackedCalls):
            return stored
        handed = twin(type(stored), stored.__dict__)
        _set(handed, "_owner", owner)
        return handed


class _PassCallsField(RecordCellField):
    """``BackwardPass.grad_fn_calls``: the calls are kept as state and read as a plain list.

    The list a read returns is built for that read, with each call keeping
    the trace it names; the cell itself holds no ``GradFnCall``.
    """

    __slots__ = ()

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the calls as a plain list of records."""

        if record is None:
            return self
        stored = RecordCellField.__get__(self, record, objtype)
        return stored.plain() if stored.__class__ is StateList else stored

    def __set__(self, record: Any, value: Any) -> None:
        """Write the cell, holding a plain list of calls as a ``StateList``."""

        if value.__class__ is list:
            value = StateList(GradFnCall, value)
        RecordCellField.__set__(self, record, value)


def install_grad_fn_facade(cls: type, stored_names: tuple[str, ...]) -> None:
    """Install the ``GradFn`` row-cell descriptors and bind the class (import time)."""

    layout = install_record_facade(cls, stored_names)
    setattr(cls, "calls", _CallsField("calls", layout.fid_by_name["calls"]))
    GRAD_FN[:] = [cls]
    TRACE_STATE_FIELDS["_grad_fn_logs"] = cls


def install_backward_pass_facade(cls: type, stored_names: tuple[str, ...]) -> None:
    """Install the ``BackwardPass`` row-cell descriptors and bind the class (import time)."""

    layout = install_record_facade(cls, stored_names)
    name = "grad_fn_calls"
    setattr(cls, name, _PassCallsField(name, layout.fid_by_name[name]))
    BACKWARD_PASS[:] = [cls]
    TRACE_STATE_FIELDS["_backward_pass_logs"] = cls


def restore_pass_state(record: Any, mapping: Any) -> None:
    """Install a ``BackwardPass`` state mapping, keeping its calls as state."""

    calls = mapping.get("grad_fn_calls")
    if calls.__class__ is list:
        mapping = {**mapping, "grad_fn_calls": StateList(GradFnCall, calls)}
    record_state_restore(record, mapping)


def pass_state_value(value: Any) -> Any:
    """Return a ``BackwardPass`` state value as the state walk and pickle carry it."""

    return value.plain() if value.__class__ is StateList else value


def grad_fn_store(logs: Any) -> Any:
    """Return ``logs`` (``GradFn`` records by autograd object id) as the container a trace holds."""

    return StateDict(GRAD_FN[0], logs, ordered=True)


def pass_store(logs: Any) -> Any:
    """Return ``logs`` (``BackwardPass`` records by pass index) as the container a trace holds."""

    return StateDict(BACKWARD_PASS[0], logs, ordered=True)


def hold_grad_fn(trace: Any, object_id: int, record: Any) -> None:
    """File a newly discovered ``GradFn`` ``record`` in ``trace`` under ``object_id``, as state."""

    logs = trace._grad_fn_logs
    if logs.__class__ is not StateDict:
        logs = trace._grad_fn_logs = grad_fn_store(logs)
    logs[object_id] = record


class GradFnKey:
    """What an ``Op`` link cell holds for a ``GradFn`` record: the record's state.

    A bare state dictionary in a store cell could be copied or pooled as a
    container, so the cell holds this key. A fork reads its parent's cell
    until it writes its own, so a fork's op names the record its parent
    hands out, as it did when the cell held the record itself.
    """

    __slots__ = ("state",)

    def __init__(self, state: dict[str, Any]) -> None:
        """Name the record with ``state``."""

        self.state = state

    def __deepcopy__(self, memo: dict[int, Any]) -> GradFnKey:
        """Copy the record this key names and key the copy."""

        return GradFnKey(hold(GRAD_FN[0], copy.deepcopy(linked(self), memo)))

    def __reduce__(self) -> Any:
        """Pickle as the record this key names."""

        return _same, (linked(self),)


def _same(value: Any) -> Any:
    """Return ``value`` (the unpickle step of a ``GradFnKey``)."""

    return value


def link_of(value: Any) -> Any:
    """Return what an ``Op`` link cell stores for ``value`` (a ``GradFn`` record becomes a key)."""

    if GRAD_FN and value.__class__ is GRAD_FN[0]:
        return GradFnKey(hold(GRAD_FN[0], value))
    return value


def linked(key: GradFnKey) -> Any:
    """Return the ``GradFn`` record ``key`` names."""

    return view(GRAD_FN[0], key.state)
