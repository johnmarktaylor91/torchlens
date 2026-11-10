"""Strong-reference walk from a trace to its record objects.

The walk answers one question: which objects keep a record (an ``Op``, a
``Layer``, a ``Module`` ...) alive, and through which attribute. It follows
only strong references (``gc.get_referents`` never crosses a weak reference)
and runs in two phases.

Phase 1 starts at the trace roots and stops at every record it meets. Every
non-record object it visits is *trace-owned storage*; an edge from such an
object to a record is a trace-owned holder.

Phase 2 starts at each record and visits the objects only that record reaches
(its *private state*). An edge from a record, or from its private state, to
another record is a record-owned holder: the path by which a held ``Buffer``
keeps an ``Op`` alive, for example.

A reference-count reconciliation then proves the walk complete: a record
whose reference count exceeds the edges the walk found is held by something
the walk did not visit (a module global, a C-level holder), and the referrers
of that record are named.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

_ATOMS = (str, bytes, bytearray, int, float, complex, bool, type(None), type(Ellipsis))
_OPAQUE = (
    type,
    types.ModuleType,
    types.CodeType,
    types.GetSetDescriptorType,
    types.MemberDescriptorType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
)
_SEQUENCES = (list, tuple, set, frozenset, deque)
_SLOT_NAMES: dict[type, tuple[tuple[str, Any], ...]] = {}

Edge = tuple[str, Any]
Absorb = Callable[[Any], bool]


def _slot_descriptors(cls: type) -> tuple[tuple[str, Any], ...]:
    """Return the ``__slots__`` member descriptors of ``cls`` and its bases."""

    cached = _SLOT_NAMES.get(cls)
    if cached is None:
        found = []
        for klass in cls.__mro__:
            for name, value in vars(klass).items():
                if isinstance(value, types.MemberDescriptorType):
                    found.append((name, value))
        cached = _SLOT_NAMES[cls] = tuple(found)
    return cached


def _slot_value(obj: Any, name: str) -> Any:
    """Read one ``__slots__`` member without running any ``__getattr__``.

    The walk must not change what it measures: a plain ``getattr`` on an
    accessor is a lookup, and a lookup builds caches that hold records.
    """

    for slot_name, descriptor in _slot_descriptors(type(obj)):
        if slot_name == name:
            try:
                return descriptor.__get__(obj)
            except AttributeError:
                return None
    return None


def _layout_of(obj: Any) -> Any:
    """Return the field layout of a row table (a store or a fork's view of one), else None."""

    layout = _slot_value(obj, "layout")
    if layout is None:
        base = _slot_value(obj, "base")
        layout = _slot_value(base, "layout") if base is not None else None
    return layout if isinstance(_slot_value(layout, "names"), tuple) else None


def _dict_edges(mapping: dict[Any, Any], attribute_style: bool) -> Iterator[Edge]:
    """Yield the strong edges of a dict, one per key and one per value.

    Keys are data (labels, addresses, indices) and collapse to ``{*}``, except
    a string key that names a row table, which stays in the path.
    """

    for key, value in list(mapping.items()):
        if attribute_style and isinstance(key, str):
            yield f".{key}", value
            continue
        if not isinstance(key, _ATOMS):
            yield "{key}", key
        named = isinstance(key, str) and _layout_of(value) is not None
        yield ("{" + key + "}" if named else "{*}"), value


_ROW_SLOTS = {"._rows": "rows", "._base_rows": "rows", "._cells": "rows", "._columns": "rows"}
_OVERLAY_SLOTS = {"._overlay": "overlay", "._base_overlay": "overlay"}


def _overlay_edges(value: Any, names: tuple[str, ...], absorb: Absorb) -> Iterator[Edge]:
    """Yield the cells of an overlay, whose integer key is ``row * width + field``."""

    if not isinstance(value, dict) or not absorb(value):
        return
    for key, cell in list(value.items()):
        field_name = names[key % len(names)] if isinstance(key, int) else "*"
        yield f".overlay[{field_name}]", cell


def _row_edges(value: Any, names: tuple[str, ...], absorb: Absorb) -> Iterator[Edge]:
    """Yield the cells of a list of rows, a flat single row, or one column per field."""

    if not isinstance(value, list) or not absorb(value):
        return
    width = len(names)
    for position, row in enumerate(list(value)):
        if not (isinstance(row, list) and len(row) == width):
            # A flat single row (``_cells``) or one column per field (``_columns``).
            yield f".rows[{names[position] if len(value) == width else '*'}]", row
        elif absorb(row):
            for fid, cell in enumerate(row):
                yield f".rows[{names[fid]}]", cell


def _table_edges(name: str, value: Any, names: tuple[str, ...], absorb: Absorb) -> Iterator[Edge]:
    """Yield the cells of one row-table container, named by field instead of by position."""

    if name in _OVERLAY_SLOTS:
        yield from _overlay_edges(value, names, absorb)
    else:
        yield from _row_edges(value, names, absorb)


def _function_edges(function: types.FunctionType) -> Iterator[Edge]:
    """Yield what a function holds besides its globals: closure, defaults, attributes."""

    for cell in function.__closure__ or ():
        try:
            yield "<closure>", cell.cell_contents
        except ValueError:
            continue
    for default in function.__defaults__ or ():
        yield "<default>", default
    for default in (function.__kwdefaults__ or {}).values():
        yield "<default>", default
    yield from _dict_edges(function.__dict__, attribute_style=True)


def _instance_dict(obj: Any) -> dict[str, Any] | None:
    """Return the instance dictionary, read without any attribute hook."""

    try:
        found = object.__getattribute__(obj, "__dict__")
    except Exception:  # noqa: BLE001 - objects without an instance dict
        return None
    return found if isinstance(found, dict) else None


def _slot_names_by_id(obj: Any) -> dict[int, str]:
    """Map the identity of each slot's value to the slot's name."""

    slot_by_id: dict[int, str] = {}
    for name, descriptor in _slot_descriptors(type(obj)):
        try:
            slot_by_id.setdefault(id(descriptor.__get__(obj)), f".{name}")
        except AttributeError:
            continue
    return slot_by_id


def _object_edges(obj: Any, absorb: Absorb) -> Iterator[Edge]:
    """Yield the strong edges of a general object, named by attribute or slot."""

    layout = _layout_of(obj)
    names = _slot_value(layout, "names") if layout is not None else None
    instance_dict = _instance_dict(obj)
    slot_by_id = _slot_names_by_id(obj)
    for referent in gc.get_referents(obj):
        if instance_dict is not None and referent is instance_dict:
            if absorb(referent):
                yield from _dict_edges(referent, attribute_style=True)
            continue
        if referent is type(obj):
            continue
        name = slot_by_id.get(id(referent), "<ref>")
        if names is not None and (name in _ROW_SLOTS or name in _OVERLAY_SLOTS):
            yield from _table_edges(name, referent, names, absorb)
        else:
            yield name, referent


def _sequence_edges(obj: Any) -> Iterator[Edge]:
    """Yield the items of a list, tuple, set or deque."""

    for item in list(obj):
        yield "[*]", item


def _method_edges(obj: types.MethodType) -> Iterator[Edge]:
    """Yield a bound method's receiver and function."""

    yield ".__self__", obj.__self__
    yield ".__func__", obj.__func__


def _traceback_edges(obj: types.TracebackType) -> Iterator[Edge]:
    """Yield a traceback's frame and the next traceback."""

    yield ".tb_frame", obj.tb_frame
    if obj.tb_next is not None:
        yield ".tb_next", obj.tb_next


def _frame_edges(obj: types.FrameType) -> Iterator[Edge]:
    """Yield a frame's locals: a finished frame kept by a traceback still holds them."""

    outside = (obj.f_globals, obj.f_builtins, obj.f_code, obj.f_back)
    for referent in gc.get_referents(obj):
        if not any(referent is item for item in outside):
            yield "<local>", referent


def _builtin_edges(obj: Any) -> Iterator[Edge]:
    """Yield a bound builtin's receiver (``some_dict.get``); a plain builtin holds a module."""

    if not isinstance(obj.__self__, types.ModuleType):
        yield ".__self__", obj.__self__


_EDGE_READERS: tuple[tuple[Any, Callable[[Any], Iterator[Edge]]], ...] = (
    (dict, lambda obj: _dict_edges(obj, attribute_style=False)),
    (_SEQUENCES, _sequence_edges),
    (types.FunctionType, _function_edges),
    (types.MethodType, _method_edges),
    (types.TracebackType, _traceback_edges),
    (types.FrameType, _frame_edges),
    ((types.BuiltinFunctionType, types.MethodWrapperType), _builtin_edges),
)


def strong_edges(obj: Any, absorb: Absorb = lambda _container: True) -> Iterator[Edge]:
    """Yield ``(name, referent)`` for every strong reference ``obj`` holds.

    Classes, modules, code and a function's globals are not followed: they are
    process state, not state of a trace. The reconciliation step catches a
    record held from there. The cells of a row table are named by field;
    ``absorb`` is told about each row container flattened that way and
    answers whether it is new.
    """

    for kinds, reader in _EDGE_READERS:
        if isinstance(obj, kinds):
            return reader(obj)
    return _object_edges(obj, absorb)


def _followable(obj: Any) -> bool:
    """Return whether the walk descends into ``obj``."""

    return not isinstance(obj, _ATOMS) and not isinstance(obj, _OPAQUE)


def _hop(name: str, child: Any) -> str:
    """Render one path hop: the edge name, plus the class of a non-container child."""

    if isinstance(child, (dict, *_SEQUENCES)):
        return name
    if _layout_of(child) is not None:
        return f"{name}:Store"
    return f"{name}:{type(child).__name__}"


@dataclass
class HolderEdge:
    """One strong reference from a holder to a record."""

    path: tuple[str, ...]
    owner: str
    record_kind: str


@dataclass
class Audit:
    """The result of one walk."""

    trace_edges: list[HolderEdge] = field(default_factory=list)
    record_edges: list[HolderEdge] = field(default_factory=list)
    alive: dict[str, int] = field(default_factory=dict)
    weak_back: dict[str, int] = field(default_factory=dict)
    unexplained: dict[str, dict[str, int]] = field(default_factory=dict)
    visited: int = 0


class _Walk:
    """Shared state of the two phases."""

    def __init__(self, record_types: dict[type, str]) -> None:
        """Create an empty walk over ``record_types`` (class to kind name)."""

        self.record_types = record_types
        self.record_classes = tuple(record_types)
        self.seen: set[int] = set()
        self.keep: list[Any] = []
        self.records: list[Any] = []
        self.record_ids: set[int] = set()
        self.explained: dict[int, int] = {}
        self.root_ids: set[int] = set()
        self.weak_back: dict[str, int] = {}

    def absorb(self, container: Any) -> bool:
        """Mark a flattened row container visited; answer whether it was new."""

        if id(container) in self.seen:
            return False
        self.seen.add(id(container))
        self.keep.append(container)
        return True

    def note_weak(self, path: tuple[str, ...], name: str, ref: Any) -> None:
        """Count a weak reference back to a root, by the path of its holder."""

        if id(ref()) in self.root_ids:
            line = "".join(path) + name
            self.weak_back[line] = self.weak_back.get(line, 0) + 1

    def kind(self, obj: Any) -> str:
        """Return the kind name of a record object."""

        for cls in type(obj).__mro__:
            if cls in self.record_types:
                return self.record_types[cls]
        raise TypeError(type(obj))

    def note_record(self, record: Any) -> None:
        """Register one reference to ``record`` found by the walk."""

        ident = id(record)
        self.explained[ident] = self.explained.get(ident, 0) + 1
        if ident not in self.record_ids:
            self.record_ids.add(ident)
            self.records.append(record)

    def expand(self, root: Any, root_path: tuple[str, ...], owner: str) -> Iterator[HolderEdge]:
        """Breadth-first walk from ``root``; yield one edge per reference to a record."""

        queue: deque[tuple[Any, tuple[str, ...]]] = deque([(root, root_path)])
        while queue:
            obj, path = queue.popleft()
            for name, child in strong_edges(obj, self.absorb):
                if isinstance(child, weakref.ReferenceType):
                    self.note_weak(path, name, child)
                if isinstance(child, self.record_classes):
                    self.note_record(child)
                    yield HolderEdge(path + (name,), owner, self.kind(child))
                elif _followable(child) and id(child) not in self.seen:
                    self.seen.add(id(child))
                    self.keep.append(child)
                    queue.append((child, path + (_hop(name, child),)))


def _stack_frame_ids() -> set[int]:
    """Return the ids of the frames now running (their locals are the audit's own)."""

    ids = set()
    frame: types.FrameType | None = sys._getframe()
    while frame is not None:
        ids.add(id(frame))
        frame = frame.f_back
    return ids


def _describe(obj: Any) -> str:
    """Name one object in a referrer chain."""

    if isinstance(obj, types.ModuleType):
        return f"module {obj.__name__}"
    if isinstance(obj, type):
        return f"class {obj.__name__}"
    if isinstance(obj, types.FunctionType):
        return f"function {obj.__qualname__}"
    if isinstance(obj, types.FrameType):
        return f"frame {obj.f_code.co_name}"
    return type(obj).__name__


def _referrer_name(referrer: Any, walk: _Walk, skip: set[int], depth: int = 5) -> str:
    """Name an unvisited referrer of a record by the chain of objects that hold it.

    The chain climbs until it reaches process state (a module, a class, a
    function), an object the walk visited, or ``depth`` levels.
    """

    chain = [_describe(referrer)]
    current = referrer
    local_skip = set(skip)
    stops = (types.ModuleType, type, types.FunctionType, types.FrameType)
    for _ in range(depth):
        if isinstance(current, stops):
            break
        owners = [owner for owner in gc.get_referrers(current) if id(owner) not in local_skip]
        local_skip.add(id(owners))
        if not owners:
            break
        current = owners[0]
        if id(current) in walk.seen or id(current) in walk.record_ids:
            chain.append(f"(visited) {_describe(current)}")
            break
        chain.append(_describe(current))
    return " <- ".join(chain)


def _name_referrers(walk: _Walk, index: int, skip: set[int], bucket: dict[str, int]) -> int:
    """Name each referrer of record ``index`` the walk did not visit; return how many.

    The record is addressed by index so that no local variable of this
    function adds a reference to it.
    """

    referrers = gc.get_referrers(walk.records[index])
    skip.add(id(referrers))
    found = 0
    for referrer in referrers:
        known = id(referrer) in walk.seen or id(referrer) in walk.record_ids
        if known or id(referrer) in skip:
            continue
        name = _referrer_name(referrer, walk, skip)
        bucket[name] = bucket.get(name, 0) + 1
        found += 1
    return found


def _reconcile(walk: _Walk, audit: Audit, own: tuple[Any, ...], limit: int) -> None:
    """Compare each record's reference count with the edges found; name the surplus."""

    own_ids = {id(item) for item in own} | {id(walk.records), id(walk.keep)}
    own_ids |= _stack_frame_ids()
    named = 0
    for index in range(len(walk.records)):
        # Two references belong to this function: the records list and the argument.
        surplus = sys.getrefcount(walk.records[index]) - 2 - walk.explained[id(walk.records[index])]
        if surplus <= 0:
            continue
        bucket = audit.unexplained.setdefault(walk.kind(walk.records[index]), {})
        name = "(not named: limit)"
        if named < limit:
            named += 1
            surplus -= _name_referrers(walk, index, own_ids, bucket)
            name = "(running frame or no collector-visible referrer)"
        if surplus > 0:
            bucket[name] = bucket.get(name, 0) + surplus


def audit_roots(
    roots: Iterable[tuple[str, Any]],
    record_types: dict[type, str],
    *,
    referrer_limit: int = 24,
) -> Audit:
    """Walk from ``roots`` and return every strong holder of a record.

    Parameters
    ----------
    roots:
        ``(name, object)`` pairs, normally one trace; a fork or a loaded trace
        is a second root.
    record_types:
        Record class to kind name.
    referrer_limit:
        How many records with a reference-count surplus get their referrers
        named (each naming scans the whole heap).

    Returns
    -------
    Audit
        Trace-owned holder edges, record-owned holder edges, records alive per
        kind and the unexplained references.
    """

    roots = list(roots)
    walk = _Walk(record_types)
    audit = Audit()
    gc.collect()
    for index in range(len(roots)):
        walk.seen.add(id(roots[index][1]))
        walk.root_ids.add(id(roots[index][1]))
    for index in range(len(roots)):
        name = roots[index][0]
        audit.trace_edges.extend(walk.expand(roots[index][1], (name,), name))
    # Index loops throughout: a loop variable bound to a record would itself
    # be a reference the reconciliation has to explain.
    position = 0
    while position < len(walk.records):
        kind = walk.kind(walk.records[position])
        audit.record_edges.extend(walk.expand(walk.records[position], (kind,), kind))
        position += 1
    for index in range(len(walk.records)):
        kind = walk.kind(walk.records[index])
        audit.alive[kind] = audit.alive.get(kind, 0) + 1
    audit.visited = len(walk.keep)
    audit.weak_back = walk.weak_back
    own = (roots, audit.trace_edges, audit.record_edges)
    _reconcile(walk, audit, own, referrer_limit)
    return audit
