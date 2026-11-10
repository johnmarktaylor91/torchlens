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
  step that lands before those two kinds are activated;
* the state handed to pickle is a transient projection: its rows stay listed
  but block nothing, and a projection that outlives the pickle call fails.

Run ``python -m tools.record_ownership`` to print the walk.
"""

from __future__ import annotations

import functools
import pickle
import weakref
from typing import Any

import pytest

from tools.record_ownership import audit_roots, record_types, run_case, signature, uses
from tools.record_ownership.cases import CASES, _read_doors, _trace
from tools.record_ownership.graph import strong_edges
from tools.record_ownership.holders import (
    ACTIVATED_FIRST,
    ACTIVATED_LATER,
    HOLDERS,
    KNOWN,
    PICKLE_CASE,
    STEPS,
    TRANSIENT,
    blocking,
    expected,
    is_transient,
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


def test_no_trace_holds_an_op_or_layer() -> None:
    """``Op``, ``Layer`` and ``Buffer`` are activated: no pinned path from a trace reaches one.

    The only rows left for them are the transient pickle projection, which no
    trace holds. A path through a buffer, a module record or a gradient record
    counts: it would be a row here like any other.
    """

    held = {
        path: holder.step
        for path, holder in blocking(ACTIVATED_FIRST + ACTIVATED_LATER).items()
        if not is_transient(path, holder)
    }
    assert held == {}


def test_transient_rows_are_only_the_pickle_projection() -> None:
    """A transient row is rooted at a pickle state and appears in that case alone."""

    rows = {path: holder for path, holder in HOLDERS.items() if holder.step in TRANSIENT}
    assert rows, "the pickle projection rows stay listed"
    assert [path for path, holder in rows.items() if not is_transient(path, holder)] == []
    assert rows.keys() <= expected(PICKLE_CASE)
    for case in sorted(set(CASES) - {PICKLE_CASE}):
        assert not rows.keys() & expected(case), case


def _live_paths(trace: Any) -> tuple[set[str], dict[str, Any]]:
    """Walk one live trace; return its holder paths and its unexplained referrers."""

    audit = audit_roots([("Trace", trace)], record_types())
    paths = {signature(edge) for edge in (*audit.trace_edges, *audit.record_edges)}
    return paths, audit.unexplained


def test_pickle_projection_does_not_outlive_the_pickle_call() -> None:
    """After ``pickle.dumps`` returns, the trace holds nothing it did not hold before."""

    trace = _trace()
    _read_doors(trace)
    before, _ = _live_paths(trace)
    payload = pickle.dumps(trace)
    after, unexplained = _live_paths(trace)
    assert payload
    # Pickling may build caches a live trace is allowed to keep (the rows of the
    # other cases); nothing else may appear, and no referrer may be out of the walk's reach.
    live_rows = set().union(*(expected(case) for case in CASES if case != PICKLE_CASE))
    assert sorted(after - before - live_rows) == []
    assert unexplained == {}


def test_a_kept_pickle_projection_is_a_new_holder() -> None:
    """A state that outlives the pickle call is found on the trace and no row excuses it."""

    trace = _trace()
    _read_doors(trace)
    before, _ = _live_paths(trace)
    trace.__dict__["_kept_state"] = trace.__getstate__()
    kept = _live_paths(trace)[0] - before
    assert kept, "the walk must find the records the kept state holds"
    assert all(path.startswith("Trace._kept_state") for path in kept), sorted(kept)
    assert not kept & HOLDERS.keys()
    excused = [
        path for path in kept if any(is_transient(path, holder) for holder in HOLDERS.values())
    ]
    assert excused == []
    for case in sorted(CASES):
        assert kept - expected(case) == kept, case


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


class _ViewingMapping(dict):  # type: ignore[type-arg]
    """A dict subclass whose reads hand out stand-ins, with a second holder in a slot."""

    __slots__ = ("_side",)

    def items(self) -> Any:
        """Hide the entries the mapping really holds."""

        return [(key, None) for key in self]


def test_walk_reads_a_dict_subclass_raw_and_follows_its_slots() -> None:
    """A mapping's own ``items`` may not hide what it holds, and its slots are holders too."""

    entry, side = _Owner(), _Owner()
    mapping = _ViewingMapping(label=entry)
    mapping._side = [side]
    found = dict(strong_edges(mapping))
    assert any(child is entry for child in found.values()), found
    assert found.get("._side") is mapping._side, found


class _ViewingSequence(list):  # type: ignore[type-arg]
    """A list subclass whose iteration hands out stand-ins, with a second holder in a slot."""

    __slots__ = ("_side",)

    def __iter__(self) -> Any:
        """Hide the items the sequence really holds."""

        return iter([None] * len(self))


def test_walk_reads_a_list_subclass_raw_and_follows_its_slots() -> None:
    """A sequence's own iteration may not hide what it holds, and its slots are holders too."""

    item, side = _Owner(), _Owner()
    sequence = _ViewingSequence([item])
    sequence._side = [side]
    found = list(strong_edges(sequence))
    assert any(child is item for _name, child in found), found
    assert any(child is sequence._side for _name, child in found), found


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
