"""The set of library-created strong holders of record objects is pinned.

A record (``Op``, ``Layer``, ``Module``, ``ModuleCall``, ``Param``, ``Buffer``,
``GradFn``, ``GradFnCall``) can only hold its trace once nothing the trace
owns holds the record back; otherwise the two form a cycle and a dropped trace
waits for the cyclic collector. ``tools/record_ownership`` walks every strong
reference from a live trace to a record and names the path. The table in
``tools/record_ownership/holders.py`` lists those paths, and these tests
compare it with the walk on each audit case:

* a path the walk finds that the table lacks is a new holder, and fails;
* a row the walk no longer finds is stale, and fails, so a change that
  converts a holder deletes its rows in the same commit;
* every reference to a record must be accounted for by the walk (a count
  against ``sys.getrefcount``), so a holder the walk cannot reach also fails;
* every path that reaches an ``Op`` or a ``Layer`` belongs to a conversion
  step that lands before those two kinds are activated.

Run ``python -m tools.record_ownership`` to print the walk.
"""

from __future__ import annotations

import functools
import weakref
from typing import Any

import pytest

from tools.record_ownership import audit_roots, run_case, signature, uses
from tools.record_ownership.cases import CASES
from tools.record_ownership.holders import (
    ACTIVATED_FIRST,
    ACTIVATION,
    BEFORE_ACTIVATION,
    HOLDERS,
    KNOWN,
    SIDE_TABLE,
    STEPS,
    blocking,
    expected,
)


@functools.cache
def _result(case: str) -> dict[str, Any]:
    """Run one audit case once per process; the result is plain data, no trace."""

    return run_case(case)


def _found(case: str) -> set[str]:
    """Return every holder path the walk found in ``case``."""

    result = _result(case)
    return set(result["trace_holders"]) | set(result["record_holders"])


@pytest.mark.parametrize("case", sorted(CASES))
def test_holder_set_is_pinned(case: str) -> None:
    """The walk finds exactly the pinned paths: no new holder, no stale row."""

    found = _found(case)
    pinned = expected(case)
    new = sorted(found - pinned)
    stale = sorted(pinned - found)
    assert not new, (
        f"case {case!r}: new strong holder of a record. Hold a row key instead, or add the "
        "path to tools/record_ownership/holders.py with the step that converts it:\n  "
        + "\n  ".join(new)
    )
    assert not stale, (
        f"case {case!r}: pinned holder no longer found. Delete its row from "
        "tools/record_ownership/holders.py (or narrow its cases):\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_reference_to_a_record_is_accounted_for(case: str) -> None:
    """No record has a referrer the walk did not reach."""

    assert _result(case)["unexplained"] == {}


@pytest.mark.parametrize("name", sorted(name for name in KNOWN if KNOWN[name][1] in HOLDERS))
def test_known_holder_is_found(name: str) -> None:
    """Each holder known before the audit is still found while its row is pinned."""

    case, path = KNOWN[name]
    assert path in _found(case)


def test_paths_to_op_and_layer_convert_before_activation() -> None:
    """Every path that reaches an ``Op`` or ``Layer`` has a step that lands before the pin.

    The one exception is the trace core's side table, which activation itself
    removes: only a path that starts there may carry the ``activation`` step.
    """

    late = {
        path: holder.step
        for path, holder in blocking(ACTIVATED_FIRST).items()
        if holder.step not in BEFORE_ACTIVATION
        and not (holder.step == ACTIVATION and path.startswith(SIDE_TABLE))
    }
    assert late == {}


def test_table_is_well_formed() -> None:
    """Rows name real cases and real steps, and no case is left without rows."""

    for path, holder in HOLDERS.items():
        assert holder.step in STEPS, path
        assert set(holder.only) | set(holder.but) <= set(CASES), path
        assert not (holder.only and holder.but), path
    assert all(expected(case) for case in CASES) or not HOLDERS


class _Record:
    """A stand-in record for the walker's own tests."""

    def __init__(self) -> None:
        """Start with no peer and no owner."""

        self.peer: Any = None
        self.owner_ref: Any = None


class _Slotted:
    """A holder that keeps a record in a slot."""

    __slots__ = ("held",)

    def __init__(self, held: Any) -> None:
        """Hold one object."""

        self.held = held


class _Lazy:
    """An object whose missing-attribute hook counts how often it runs."""

    def __init__(self) -> None:
        """Start the count at zero."""

        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        """Count the lookup and refuse it."""

        self.__dict__["calls"] += 1
        raise AttributeError(name)


class _Owner:
    """A stand-in trace: a list, a dictionary and a slotted holder of records."""

    def __init__(self) -> None:
        """Build two records; the first holds the second."""

        first, second = _Record(), _Record()
        first.peer = second
        first.owner_ref = weakref.ref(self)
        self.items = [first, second]
        self.by_name = {"first": first}
        self.cell = _Slotted(second)
        self.lazy = _Lazy()


def _signatures(edges: list[Any]) -> set[str]:
    """Render holder edges as path lines."""

    return {signature(edge) for edge in edges}


def test_walk_names_each_path_and_stops_at_records() -> None:
    """Trace-owned paths and record-owned paths are listed apart, by name."""

    owner = _Owner()
    audit = audit_roots([("Owner", owner)], {_Record: "Record"})
    assert _signatures(audit.trace_edges) == {
        "Owner.items[*] -> Record",
        "Owner.by_name{*} -> Record",
        "Owner.cell:_Slotted.held -> Record",
    }
    assert _signatures(audit.record_edges) == {"Record.peer -> Record"}
    assert audit.alive == {"Record": 2}
    assert audit.unexplained == {}
    assert sum(audit.weak_back.values()) == 1


def test_walk_reports_a_holder_it_cannot_reach() -> None:
    """A referrer outside the walked graph shows up in the reference count."""

    owner = _Owner()
    hidden = _Slotted(owner.items[0])
    audit = audit_roots([("Owner", owner)], {_Record: "Record"})
    names = [name for found in audit.unexplained.values() for name in found]
    assert any("_Slotted" in name for name in names), audit.unexplained
    assert hidden.held is owner.items[0]


def test_walk_does_not_run_attribute_hooks() -> None:
    """The walk reads slots and dictionaries directly; a lookup would build caches."""

    owner = _Owner()
    audit_roots([("Owner", owner)], {_Record: "Record"})
    assert owner.lazy.calls == 0


_USES_SOURCE = """
def touch(trace, other):
    for op in trace._layer_list:
        if op is other:
            return op
    count = len(trace._layer_list)
    trace._layer_list.append(other)
    trace._layer_dict_all_keys["label"] = other
    return trace._layer_logs["label"], count
"""


def test_uses_are_classified() -> None:
    """Each use of the op list and label dictionaries gets one category."""

    found, loops = uses.scan_source("example.py", _USES_SOURCE)
    assert sorted((use.name, use.category) for use in found) == [
        ("_layer_dict_all_keys", "mutate"),
        ("_layer_list", "iterate"),
        ("_layer_list", "mutate"),
        ("_layer_list", "size"),
        ("_layer_logs", "index"),
    ]
    assert [(loop.name, loop.variable, loop.line) for loop in loops] == [("_layer_list", "op", 3)]
