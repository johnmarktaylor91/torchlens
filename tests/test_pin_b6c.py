"""A ``Module`` or ``ModuleCall`` read from a trace keeps the trace; so does a facet view.

The trace keeps the state of each module and call and builds the record object
when one is read. A record that was handed out holds its trace, so a chained
read (``tl.trace(model, x).modules["fc"].out``) works. A record copied or
restored on its own has no trace to hold and reads as it did before.
"""

from __future__ import annotations

import copy
import gc
import pickle
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tools.record_ownership.holders import blocking
from torchlens._data_classes import _record_views
from torchlens._data_classes._state_adapter import plain_state
from torchlens._data_classes.module import Module, ModuleCall
from torchlens._trace_core import record_refs
from torchlens._trace_core.record_rows import ROW_KEY, RecordCellField


class _Net(nn.Module):
    """Three modules called once and one (``act``) called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.act = nn.ReLU()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x))).mean((2, 3))
        return self.act(self.fc(x))


_LAST: list[Any] = []


def _trace() -> Any:
    torch.manual_seed(0)
    trace = tl.trace(_Net(), torch.randn(2, 1, 6, 6))
    _LAST[:] = [weakref.ref(trace)]
    return trace


@pytest.fixture
def no_gc() -> Iterator[None]:
    """Reference counting alone must free a trace: keep the cycle collector out."""

    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def _outcome(record: Any, name: str) -> str:
    try:
        value = getattr(record, name)
    except Exception as exc:  # noqa: BLE001 - the outcome is what is compared
        code = getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
        return f"raises {type(exc).__name__}" + (f":{code}" if code else "")
    if callable(value) and not hasattr(value, "__iter__"):
        return "callable"
    if type(value) in (str, int, bool, type(None)):
        return f"ok {type(value).__name__} {value!r}"
    return f"ok {type(value).__name__}"


def _names(record: Any) -> list[str]:
    names = set(dir(type(record)))
    names.update(getattr(record, "__dict__", {}))
    return sorted(name for name in names if not name.startswith("_"))


def _outcomes(record: Any) -> dict[str, str]:
    return {name: _outcome(record, name) for name in _names(record)}


# Every door that hands out a Module or a ModuleCall, each reached in one expression.
_DOORS: dict[str, Callable[[Any], Any]] = {
    "trace.modules[address]": lambda t: t.modules["fc"],
    "trace.modules[position]": lambda t: t.modules[1],
    "trace.modules[pass label]": lambda t: t.modules["act:2"],
    "trace.modules iteration": lambda t: next(iter(t.modules)),
    "trace.modules.values()": lambda t: t.modules.values()[-1],
    "trace.modules.items()": lambda t: t.modules.items()[-1][1],
    "trace.modules.head()": lambda t: t.modules.head(2)[1],
    "trace.module_calls[label]": lambda t: t.module_calls["act:2"],
    "trace.module_calls[position]": lambda t: t.module_calls[0],
    "trace.module_calls iteration": lambda t: next(iter(t.module_calls)),
    "trace.module_calls.values()": lambda t: t.module_calls.values()[-1],
    "module.calls[position]": lambda t: t.modules["act"].calls[1],
    "module.calls.values()": lambda t: t.modules["act"].calls.values()[0],
    "module.calls.get()": lambda t: t.modules["act"].calls.get(2),
    "module.ops[position]": lambda t: t.modules["fc"].ops[0],
    "module_call.module": lambda t: t.module_calls["act:2"].module,
    "module.calls[position].module": lambda t: t.modules["fc"].calls[0].module,
    "trace.root_module": lambda t: t.root_module,
    "trace.call_tree": lambda t: t.call_tree,
    "trace.walk_calls()": lambda t: list(t.walk_calls())[-1],
    "module.call_tree": lambda t: t.modules["fc"].call_tree,
    "module.walk_descendants()": lambda t: next(t.root_module.walk_descendants()),
    "module_call.walk_descendants()": lambda t: list(t.call_tree.walk_descendants())[-1],
}


# An Op keeps the receptive-field view it computed in its own row, and that view holds the
# Op, which holds the trace. A read of either field through any record therefore leaves the
# trace to the cycle collector, as it does before this change. Both names are compared with
# the control; the lifetime check reads every other attribute.
_OP_FIELD_VIEWS = frozenset({"receptive_field", "projective_field"})


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_chained_read_matches_a_held_trace(door: str, no_gc: None) -> None:
    """Every public attribute reads the same with no variable holding the trace."""

    reach = _DOORS[door]
    trace = _trace()
    held = _outcomes(reach(trace))
    del trace
    record = reach(_trace())
    assert isinstance(record, (Module, ModuleCall))
    assert _LAST[0]() is not None, "the record did not keep its trace"
    assert _outcomes(record) == held
    del record
    record = reach(_trace())
    for name in _names(record):
        if name not in _OP_FIELD_VIEWS:
            _outcome(record, name)
    del record
    assert _LAST[0]() is None, "the trace outlived the record that held it"


def test_one_record_object_across_doors() -> None:
    """While a record is held, every door returns that same object."""

    trace = _trace()
    module = trace.modules["act"]
    call = trace.module_calls["act:2"]
    assert trace.modules["act"] is module
    assert trace.modules["act:2"] is module
    assert trace.modules.values()[trace.modules.keys().index("act")] is module
    assert call.module is module
    assert module.calls[1] is call
    assert module.calls.get(2) is call
    assert trace.module_calls.values()[trace.module_calls.keys().index("act:2")] is call
    assert trace.modules is trace.modules
    assert module.calls is module.calls
    assert trace.module_calls is trace.module_calls


def test_state_outlives_the_record_object() -> None:
    """An attribute set on a record is there when the record is built again."""

    trace = _trace()
    trace.modules["fc"].note = "kept"
    trace.module_calls["fc:1"].note = "kept too"
    gc.collect()
    assert trace.modules["fc"].note == "kept"
    assert trace.module_calls["fc:1"].note == "kept too"


def test_trace_with_no_record_held_is_freed_by_reference_counting(no_gc: None) -> None:
    """Reading through every door leaves nothing that holds the trace."""

    trace = _trace()
    for reach in _DOORS.values():
        reach(trace)
    trace.modules["conv"].facets.keys()
    trace.module_calls["fc:1"].facets.keys()
    trace.ops[0].facets.keys()
    trace.modules.to_pandas()
    repr(trace.modules), repr(trace.module_calls), repr(trace.modules["act"])
    del trace
    assert _LAST[0]() is None


@pytest.mark.parametrize("kind", ["module", "module_call"])
@pytest.mark.parametrize("how", ["copy", "deepcopy", "pickle"])
def test_record_copied_or_restored_on_its_own_holds_no_trace(
    kind: str, how: str, no_gc: None
) -> None:
    """A copy has no trace: it does not keep one alive and trace-backed reads refuse."""

    trace = _trace()
    record = trace.modules["fc"] if kind == "module" else trace.module_calls["fc:1"]
    if how == "copy":
        twin = copy.copy(record)
    elif how == "deepcopy":
        twin = copy.deepcopy(record)
    else:
        twin = pickle.loads(pickle.dumps(record))
    del trace, record
    assert _LAST[0]() is None, "a copied record kept the trace alive"
    assert twin._source_trace is None
    outcomes = _outcomes(twin)
    assert any(outcome.startswith("raises RecordBindingError") for outcome in outcomes.values())
    assert outcomes["address"] == "ok str 'fc'"


def _facet_outcomes(view: Any) -> dict[str, str]:
    outcomes = {}
    for key in view.keys():
        try:
            outcomes[str(key)] = f"ok {type(view[key]).__name__}"
        except Exception as exc:  # noqa: BLE001 - the outcome is what is compared
            outcomes[str(key)] = f"raises {type(exc).__name__}"
    return outcomes


_FACET_DOORS: dict[str, Callable[[Any], Any]] = {
    "op.facets": lambda t: t.ops[1].facets,
    "module.facets": lambda t: t.modules["conv"].facets,
    "module_call.facets": lambda t: t.module_calls["conv:1"].facets,
}


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
def test_chained_read_through_a_facet_view(door: str, no_gc: None) -> None:
    """A facet view reached in one expression computes what a held trace computes."""

    reach = _FACET_DOORS[door]
    trace = _trace()
    held = _facet_outcomes(reach(trace))
    assert held, "the control has no facets to compare"
    del trace
    view = reach(_trace())
    assert _LAST[0]() is not None, "the facet view did not keep its trace"
    assert _facet_outcomes(view) == held
    del view
    assert _LAST[0]() is None, "the trace outlived the facet view that held it"


@pytest.mark.parametrize("door", sorted(_FACET_DOORS))
def test_facet_view_is_one_object_and_shares_what_it_computed(door: str) -> None:
    """Two reads of ``facets`` on a held record return one view over one cache."""

    trace = _trace()
    record_door = {
        "op.facets": lambda t: t.ops[1],
        "module.facets": lambda t: t.modules["conv"],
        "module_call.facets": lambda t: t.module_calls["conv:1"],
    }[door]
    record = record_door(trace)
    view = record.facets
    assert record.facets is view
    view.keys()
    del view
    again = record.facets
    assert again._computed, "the facets computed through the first view were lost"


def test_child_door_keys_do_not_depend_on_object_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every record reports the same ``id`` while the cells are keyed: the keys stay right.

    The key of a call in its trace's door was looked up by ``id(call)``. A
    record is a view that lives only while somebody holds it, so an ``id`` can
    name another record later; forcing every ``id`` to collide shows it.
    """

    monkeypatch.setattr(record_refs, "id", lambda obj: 7, raising=False)
    trace = _trace()
    monkeypatch.undo()
    by_row = {
        call.__dict__[ROW_KEY]: label for label, call in trace._module_logs._pass_dict.items()
    }
    keyed = 0
    for module in trace.modules:
        keys = RecordCellField.__get__(Module.__dict__["ops"], module)
        if keys.__class__ is not record_refs.RowKeys:
            continue
        keyed += 1
        assert list(keys.door) == [by_row[row] for row in keys.values()]
    assert keyed >= 4


def test_view_table_does_not_trust_a_reused_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every state reporting one ``id``, each read still returns the right record."""

    trace = _trace()
    monkeypatch.setattr(_record_views, "id", lambda obj: 7, raising=False)
    for address in trace.modules.keys():
        assert trace.modules[address].address == address
    assert [module.address for module in trace.modules] == trace.modules.keys()
    for label in trace.module_calls.keys():
        assert trace.module_calls[label].label == label
    assert [call.label for call in trace.module_calls] == trace.module_calls.keys()


def test_holders_keep_state_and_state_streams_keep_records() -> None:
    """The accessors hold states; pickle, copy and ``plain_state`` see the plain containers."""

    trace = _trace()
    stored = trace._module_logs
    assert stored._dict.__class__ is _record_views.RecordDict
    assert stored._list.__class__ is _record_views.RecordList
    assert stored._pass_dict.__class__ is _record_views.RecordDict
    assert all(state.__class__ is dict for state in stored._dict.states())
    for state in (stored.__getstate__(), plain_state(stored)):
        assert state["_dict"].__class__ is dict
        assert state["_list"].__class__ is list
        assert state["_pass_dict"].__class__ in (dict, OrderedDict)
        assert all(isinstance(module, Module) for module in state["_dict"].values())
        assert all(isinstance(module, Module) for module in state["_list"])
        assert all(isinstance(call, ModuleCall) for call in state["_pass_dict"].values())
    calls = trace.modules["act"].calls
    assert calls.__getstate__()["_dict"].__class__ is dict
    restored = pickle.loads(pickle.dumps(trace))
    assert restored.modules.keys() == trace.modules.keys()
    assert restored.module_calls.keys() == trace.module_calls.keys()
    assert restored._module_logs._dict.__class__ is _record_views.RecordDict
    assert restored.modules["act"].calls[1].module is restored.modules["act"]
    forked = trace.fork()
    assert forked.modules.keys() == trace.modules.keys()
    assert forked.modules["fc"] is not trace.modules["fc"]
    assert forked.modules["fc"]._source_trace is forked


def test_no_pinned_holder_reaches_a_module_or_a_call() -> None:
    """The ownership table has no row, transient or not, that reaches either kind."""

    assert blocking(("Module", "ModuleCall")) == {}


def test_buffer_state_still_carries_its_version_ops() -> None:
    """A buffer's pickle state names its versions as ops, and a round trip keeps their count."""

    trace = _trace()
    buffer = next(iter(trace.buffers))
    assert buffer.versions
    state = buffer.__getstate__()
    assert [type(op).__name__ for op in state["versions"]] == ["Op"] * len(buffer.versions)
    restored = pickle.loads(pickle.dumps(trace))
    assert [len(item.versions) for item in restored.buffers] == [
        len(item.versions) for item in trace.buffers
    ]
    assert all(type(op).__name__ == "Op" for item in restored.buffers for op in item.versions)
