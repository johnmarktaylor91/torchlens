"""The payload eviction walk is cheaper and evicts exactly what it did before.

When the last trace owning an op store dies (or a rerun discards its own capture),
``OpRowStore._evict_tensor_payloads`` replaces every reachable tensor payload with
``None``. On a 585-op GPT-2 capture it made about 831,000 Python and builtin calls,
mostly ``isinstance`` on cells that hold ``None``, labels and counts. The walk now
dispatches on exact type first. These tests pin that the evicted set is unchanged
(against the former walk, kept here verbatim as the reference) and that the call
count stays down.
"""

from __future__ import annotations

import collections
import gc
import sys
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core import op_store
from torchlens.options import CaptureOptions

_CEILING = op_store._EVICT_WALK_DEPTH_CEILING


def _reference_evict(
    value: Any, tensor_cls: type, depth: int = 0, seen: set[int] | None = None
) -> Any:
    """The eviction walk as it was before this change (verbatim body)."""

    if isinstance(value, tensor_cls):
        return None
    if depth >= _CEILING:
        return value
    if isinstance(value, list):
        if seen is None:
            seen = set()
        if id(value) in seen:
            return value
        seen.add(id(value))
        for index, item in enumerate(value):
            value[index] = _reference_evict(item, tensor_cls, depth + 1, seen)
        return value
    if isinstance(value, tuple):
        rebuilt = [_reference_evict(item, tensor_cls, depth + 1, seen) for item in value]
        if all(new is old for new, old in zip(rebuilt, value, strict=True)):
            return value
        if type(value) is tuple:
            return tuple(rebuilt)
        try:
            return type(value)(*rebuilt)
        except Exception:
            try:
                return type(value)(rebuilt)
            except Exception:
                return value
    if isinstance(value, dict):
        if seen is None:
            seen = set()
        if id(value) in seen:
            return value
        seen.add(id(value))
        for key in list(value):
            value[key] = _reference_evict(value[key], tensor_cls, depth + 1, seen)
        return value
    if isinstance(value, (set, frozenset)):
        if not any(isinstance(item, tensor_cls) for item in value):
            return value
        survivors = {item for item in value if not isinstance(item, tensor_cls)}
        survivors.add(None)
        try:
            return type(value)(survivors)
        except Exception:
            return value
    return value


def _reference_store_evict(store: Any) -> None:
    """``OpRowStore._evict_tensor_payloads`` as it was before this change."""

    tensor_cls = torch.Tensor
    evict = _reference_evict
    if store._rows is not None:
        for row_cells in store._rows:
            for fid, value in enumerate(row_cells):
                row_cells[fid] = evict(value, tensor_cls)
    if store._columns is not None:
        for column in store._columns:
            if column.packed:
                continue
            values = column.values
            for index, value in enumerate(values):
                values[index] = evict(value, tensor_cls)
    for key, value in store._overlay.items():
        store._overlay[key] = evict(value, tensor_cls)
    for view_ref in store._payload_views:
        view = view_ref()
        if view is None:
            continue
        if view._base_rows is not None:
            for row_cells in view._base_rows:
                for fid, value in enumerate(row_cells):
                    row_cells[fid] = evict(value, tensor_cls)
        for view_overlay in (view._base_overlay, view._overlay):
            for key, value in view_overlay.items():
                view_overlay[key] = evict(value, tensor_cls)


# --- structural description: what is a tensor, what is the same object as before ---


def _index_objects(value: Any, path: str, out: dict[int, Any], depth: int = 0) -> None:
    """Map ``id`` to ``(path, object)``; holding the object keeps its id from reuse."""

    if id(value) in out or depth > _CEILING + 3:
        return
    out[id(value)] = (path, value)
    if isinstance(value, dict):
        for key, item in list(dict.items(value)):
            _index_objects(item, f"{path}[{key!r}]", out, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        for index, item in enumerate(items):
            _index_objects(item, f"{path}[{index}]", out, depth + 1)


def _describe(
    value: Any, before: dict[int, Any], depth: int = 0, seen: set[int] | None = None
) -> str:
    """Structure of ``value``: tensors, types, and which objects are the originals."""

    origin = before[id(value)][0] if id(value) in before else "new"
    if isinstance(value, torch.Tensor):
        return f"T<{origin}>"
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        # Types, not values: two captures of one model differ in timings, ids
        # and per-trace label tokens.
        return type(value).__name__
    seen = set() if seen is None else seen
    if id(value) in seen or depth > _CEILING + 3:
        return f"<again {origin}>"
    seen.add(id(value))
    name = f"{type(value).__name__}<{origin}>"
    if isinstance(value, dict):
        inner = ",".join(
            f"{type(key).__name__}:{_describe(item, before, depth + 1, seen)}"
            for key, item in dict.items(value)
        )
        return f"{name}{{{inner}}}"
    if isinstance(value, (set, frozenset)):
        return f"{name}{{{','.join(sorted(_describe(i, before, depth + 1, seen) for i in value))}}}"
    if isinstance(value, (list, tuple)):
        return f"{name}[{','.join(_describe(i, before, depth + 1, seen) for i in value)}]"
    return name


# --- adversarial values ---


class _Pair(NamedTuple):
    left: Any
    right: Any


class _Strict(tuple):
    """A tuple subclass neither ``cls(*items)`` nor ``cls(items)`` can rebuild."""

    def __new__(cls, items: Any, tag: str) -> _Strict:
        return super().__new__(cls, items)


class _IterableOnly(tuple):
    """A tuple subclass rebuilt only from one iterable."""

    def __new__(cls, items: Any) -> _IterableOnly:
        return super().__new__(cls, items)


class _LoggedList(list):
    log: list[Any]

    def __setitem__(self, index: Any, value: Any) -> None:
        self.log.append(("set", index, type(value).__name__))
        super().__setitem__(index, value)


class _LoggedDict(dict):
    log: list[Any]

    def __getitem__(self, key: Any) -> Any:
        self.log.append(("get", key))
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.log.append(("set", key, type(value).__name__))
        super().__setitem__(key, value)


class _Holder:
    def __init__(self, payload: Any) -> None:
        self.payload = payload


class _ClaimsTensor:
    """An object whose ``__class__`` claims to be a tensor (isinstance believes it)."""

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return torch.Tensor


class _Bytes(int):
    pass


class _Label(str):
    pass


def _deep(kind: Callable[[Any], Any], levels: int) -> Any:
    value: Any = torch.ones(1)
    for level in range(levels):
        value = kind([value, level, torch.zeros(1)])
    return value


def _adversarial_cells() -> list[Any]:
    """Fresh, equal inputs on every call."""

    tensor = torch.arange(3.0)
    shared = [torch.ones(1), "x"]
    cyclic_list: list[Any] = [torch.ones(1)]
    cyclic_list.append(cyclic_list)
    cyclic_dict: dict[str, Any] = {"t": torch.ones(1)}
    cyclic_dict["self"] = cyclic_dict
    logged_list = _LoggedList([torch.ones(1), 1, "a", None, (torch.ones(1),)])
    logged_list.log = []
    logged_dict = _LoggedDict(a=torch.ones(1), b=2, c=None)
    logged_dict.log = []
    defaults: collections.defaultdict[str, Any] = collections.defaultdict(list)
    defaults["t"].append(torch.ones(1))
    return [
        None,
        True,
        7,
        2.5,
        1j,
        "label",
        b"raw",
        object(),
        _Bytes(5),
        _Label("lbl"),
        tensor,
        nn.Parameter(torch.ones(2)),
        [tensor, "a", 1, None],
        ("a", 1, None),
        ("a", torch.ones(1), 3),
        (["inner", torch.ones(1)], "same tuple kept"),
        {"a": torch.ones(1), "b": "keep", "c": [torch.ones(1), {"d": torch.ones(1)}]},
        {torch.ones(1), "a", None},
        frozenset({torch.ones(1), "a"}),
        frozenset({"a", 1}),
        _Pair(torch.ones(1), "r"),
        _Pair("l", "r"),
        _Strict((torch.ones(1), 1), tag="t"),
        _IterableOnly((torch.ones(1), 2)),
        torch.Size([2, 3]),
        _deep(list, _CEILING + 3),
        _deep(tuple, _CEILING + 3),
        _deep(lambda items: dict(enumerate(items)), _CEILING + 3),
        [shared, shared, (shared,)],
        cyclic_list,
        cyclic_dict,
        [cyclic_list, {"loop": cyclic_dict}],
        logged_list,
        logged_dict,
        defaults,
        collections.OrderedDict(a=torch.ones(1), b=1),
        _Holder(torch.ones(1)),
        [_Holder(torch.ones(1)), _ClaimsTensor()],
        _ClaimsTensor(),
        (torch.float32, torch.device("cpu"), torch.ones(1).dtype),
    ]


def _walk_cells(cells: list[Any], evict: Callable[..., Any]) -> list[str]:
    before: dict[int, Any] = {}
    _index_objects(cells, "cells", before)
    results = [evict(cell, torch.Tensor) for cell in cells]
    described = [_describe(result, before) for result in results]
    for cell in cells:
        log = getattr(cell, "log", None)
        if log is not None:
            described.append(repr(log))
    return described


def test_the_inert_types_are_never_tensors_or_containers() -> None:
    """The exact-type shortcut answers what the isinstance chain would."""

    assert type(torch.Tensor).__instancecheck__ is type.__instancecheck__
    for cls in op_store._EVICT_INERT_TYPES:
        assert not issubclass(cls, (torch.Tensor, *op_store._EVICT_CONTAINER_TYPES)), cls


def test_every_adversarial_value_evicts_as_the_former_walk_did() -> None:
    reference = _walk_cells(_adversarial_cells(), _reference_evict)
    current = _walk_cells(_adversarial_cells(), op_store._evict_payloads_in_cell)
    assert len(current) == len(reference)
    for index, (got, expected) in enumerate(zip(current, reference, strict=True)):
        assert got == expected, f"cell {index}"


def test_the_store_list_and_mapping_sweeps_match_per_cell_eviction() -> None:
    cells = _adversarial_cells()
    reference_cells = _adversarial_cells()
    before: dict[int, Any] = {}
    _index_objects(cells, "cells", before)
    reference_before: dict[int, Any] = {}
    _index_objects(reference_cells, "cells", reference_before)
    op_store._evict_payloads_in_list(cells, torch.Tensor)
    for index, value in enumerate(reference_cells):
        reference_cells[index] = _reference_evict(value, torch.Tensor)
    assert [_describe(c, before) for c in cells] == [
        _describe(c, reference_before) for c in reference_cells
    ]
    mapping = dict(enumerate(_adversarial_cells()))
    reference_mapping = dict(enumerate(_adversarial_cells()))
    before, reference_before = {}, {}
    _index_objects(mapping, "m", before)
    _index_objects(reference_mapping, "m", reference_before)
    op_store._evict_payloads_in_mapping(mapping, torch.Tensor)
    for key, value in reference_mapping.items():
        reference_mapping[key] = _reference_evict(value, torch.Tensor)
    assert _describe(mapping, before) == _describe(reference_mapping, reference_before)


# --- real captures: one row-major store, one columnar store, with a surviving fork view ---


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _Long(nn.Module):
    """About 545 ops: past the store's columnar-transpose threshold (512 rows)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        for _ in range(180):
            h = torch.tanh(h * 1.01 + 0.1)
        return h.sum(dim=-1)


def _store_cells(store: Any) -> Iterator[tuple[str, Any, Any]]:
    """(location, container, key) for every cell the walk visits, in its order."""

    if store._rows is not None:
        for r, row in enumerate(store._rows):
            for fid in range(len(row)):
                yield f"r{r}.{fid}", row, fid
    if store._columns is not None:
        for c, column in enumerate(store._columns):
            if not column.packed:
                for i in range(len(column.values)):
                    yield f"c{c}.{i}", column.values, i
    for key in list(store._overlay):
        yield f"o{key}", store._overlay, key
    for v, view_ref in enumerate(store._payload_views):
        view = view_ref()
        if view is None:
            continue
        if view._base_rows is not None:
            for r, row in enumerate(view._base_rows):
                for fid in range(len(row)):
                    yield f"v{v}r{r}.{fid}", row, fid
        for name in ("_base_overlay", "_overlay"):
            for key in list(getattr(view, name)):
                yield f"v{v}{name}{key}", getattr(view, name), key


def _evicted_capture(model_cls: type[nn.Module], walk: Callable[[Any], None] | None) -> list[str]:
    """Capture, fork with a write, keep one fork Op, drop both traces; describe every cell.

    ``walk`` replaces the store's eviction (``None`` keeps the current one).
    """

    torch.manual_seed(0)
    model = model_cls()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    store = trace._trace_core.ops
    fork = trace.fork()
    fork.do(tl.func("linear") if model_cls is _Long else tl.func("relu"), tl.scale(2.0))
    kept_op = fork.layer_list[2]
    before: dict[int, Any] = {}
    cells_before = {loc: container[key] for loc, container, key in _store_cells(store)}
    for loc, value in cells_before.items():
        _index_objects(value, loc, before)
    original = op_store.OpRowStore._evict_tensor_payloads
    if walk is not None:
        op_store.OpRowStore._evict_tensor_payloads = walk  # type: ignore[method-assign]
    try:
        del trace, fork
        gc.collect()
    finally:
        op_store.OpRowStore._evict_tensor_payloads = original  # type: ignore[method-assign]
    assert store._payload_owner_count == 0
    described = [
        f"{loc}={_describe(container[key], before)}" for loc, container, key in _store_cells(store)
    ]
    assert kept_op.layer_label, "the retained fork Op stays readable"
    del cells_before
    return described


@pytest.mark.parametrize(
    "model_cls",
    [_Small, pytest.param(_Long, marks=pytest.mark.heavy)],  # columnar: 13.6 s measured
    ids=["row_major", "columnar"],
)
def test_a_real_capture_evicts_the_same_cells_as_the_former_walk(
    model_cls: type[nn.Module],
) -> None:
    expected = _evicted_capture(model_cls, _reference_store_evict)
    got = _evicted_capture(model_cls, None)
    assert len(got) == len(expected)
    assert sum("T<" in line for line in expected) == sum("T<" in line for line in got)
    mismatches = [(e, g) for e, g in zip(expected, got, strict=True) if e != g]
    assert not mismatches, mismatches[:3]


def _walk_events(store: Any) -> int:
    """Profiler events (Python and builtin calls) inside one eviction walk."""

    target = op_store.OpRowStore._evict_tensor_payloads.__code__
    state = {"inside": 0, "events": 0}

    def count(frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code is target:
            state["inside"] += 1
        if state["inside"] and event in ("call", "c_call"):
            state["events"] += 1
        if event == "return" and frame.f_code is target:
            state["inside"] -= 1

    sys.setprofile(count)
    try:
        store._evict_tensor_payloads()
    finally:
        sys.setprofile(None)
    return state["events"]


@pytest.mark.parametrize(
    ("model_cls", "bar"),
    [
        # Measured on the base walk: 6,561 events; on this walk: 907.
        (_Small, 1_200),
        # Measured on the base walk: 534,805 events; on this walk: 94,295 (6.2 s measured).
        pytest.param(_Long, 120_000, marks=pytest.mark.heavy),
    ],
    ids=["row_major", "columnar"],
)
def test_one_walk_stays_cheap(model_cls: type[nn.Module], bar: int) -> None:
    """Profiler events in one walk over a fixed capture; the bar leaves about 30 % headroom."""

    torch.manual_seed(0)
    trace = tl.trace(
        model_cls(), torch.randn(2, 4), capture=CaptureOptions(intervention_ready=True)
    )
    store = trace._trace_core.ops
    assert (store._columns is not None) is (model_cls is _Long), "fixture store shape"
    events = _walk_events(store)
    del trace
    gc.collect()
    assert events <= bar, events
