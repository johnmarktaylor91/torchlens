"""The fork copy path takes atomic fast paths and copies exactly what it did before.

A fork copies its trace-side tables through the flat container copy, fills the
detached module records cell by cell, and copies record extras and Layer
shadow dicts through ``cow_copy_value``. All-atomic containers are now copied
without a Python call per member, and atomic cells skip the copy call (which
returns them unchanged). These tests run the former functions, kept here
verbatim as the reference, beside the new ones: on adversarial values they
must build the same structure, share and copy the same objects, and leave the
same fork memo; on real captures a fork built by the former code must equal
one built by the new code field by field. The fork's call count on a fixed
capture is pinned.
"""

from __future__ import annotations

import functools
import math
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._data_classes._trace_fork as trace_fork
from torchlens._data_classes._state_adapter import state_items, state_new
from torchlens._data_classes.layer import OpAccessor
from torchlens._trace_core.op_store import (
    _MISSING,
    DetachedOpStore,
    OpStoreLayout,
    PooledCell,
    cow_copy_value,
)
from torchlens._trace_core.record_rows import CORE_KEY, ROW_KEY
from torchlens.options import CaptureOptions, InterventionOptions

_NO_FLAT_COPY = trace_fork._NO_FLAT_COPY
_FLAT_COPY_DEPTH = trace_fork._FLAT_COPY_DEPTH
_ATOMIC_COPY_TYPES = trace_fork._ATOMIC_COPY_TYPES
_FLAT_COPY_TYPES = trace_fork._FLAT_COPY_TYPES
_ATOMIC_FACTORY_TYPES = trace_fork._ATOMIC_FACTORY_TYPES
_RecordTranslator = trace_fork._RecordTranslator


# The former code, verbatim apart from the reference names its calls resolve to.


def _reference_flat_container_copy(value: Any, memo: dict[Any, Any], depth: int) -> Any:
    cls = type(value)
    if cls not in _FLAT_COPY_TYPES:
        return _NO_FLAT_COPY
    if cls is not dict and getattr(value, "__dict__", None):
        return _NO_FLAT_COPY
    if cls is defaultdict and not isinstance(value.default_factory, _ATOMIC_FACTORY_TYPES):
        return _NO_FLAT_COPY
    known = memo.get(id(value), _NO_FLAT_COPY)
    if known is not _NO_FLAT_COPY:
        return known
    result = _reference_flat_copy_contents(value, cls, memo, depth)
    if result is not _NO_FLAT_COPY and result is not value:
        memo[id(value)] = result
        memo.setdefault(id(memo), []).append(value)  # deepcopy's keep-alive list
    return result


def _reference_flat_copy_contents(value: Any, cls: type, memo: dict[Any, Any], depth: int) -> Any:
    mapping = cls is dict or cls is OrderedDict or cls is defaultdict
    if mapping and any(type(key) not in _ATOMIC_COPY_TYPES for key in value):
        return _NO_FLAT_COPY
    items = [
        _reference_flat_element_copy(item, memo, depth)
        for item in (value.values() if mapping else value)
    ]
    if any(item is _NO_FLAT_COPY for item in items):
        return _NO_FLAT_COPY
    if cls is defaultdict:
        return defaultdict(value.default_factory, zip(value, items, strict=True))
    if mapping:
        return cls(zip(value, items, strict=True))
    if cls is tuple and all(new is old for new, old in zip(items, value, strict=True)):
        return value
    return cls(items)


def _reference_flat_element_copy(item: Any, memo: dict[Any, Any], depth: int) -> Any:
    if type(item) in _ATOMIC_COPY_TYPES:
        return item
    known = memo.get(id(item), _NO_FLAT_COPY)
    if known is not _NO_FLAT_COPY or depth <= 1:
        return known
    return _reference_flat_container_copy(item, memo, depth - 1)


def _reference_fill_detached_record(
    parent_record: Any, shell: Any, translator: _RecordTranslator
) -> None:
    parent_dict = parent_record.__dict__
    store = parent_dict[CORE_KEY]
    row = parent_dict[ROW_KEY]
    layout = type(parent_record)._TL_LAYOUT
    detached = DetachedOpStore(layout)
    for fid in range(layout.n_fields):
        value = store.cell_get(row, fid)
        if value.__class__ is PooledCell:
            value = value.hydrate()
        elif value.__class__ is str and store.compacted_singleton(row, fid, value):
            value = [value]
        detached.cell_set(0, fid, cow_copy_value(value, translator))
    shell.__dict__[CORE_KEY] = detached
    shell.__dict__[ROW_KEY] = 0


def _reference_copy_record_extras(
    parent_record: Any, shell: Any, translator: _RecordTranslator, fork_ref: Any
) -> None:
    declared = getattr(type(parent_record), "_TL_LAYOUT", None)
    declared_names = declared.fid_by_name if declared is not None else ()
    for key, value in parent_record.__dict__.items():
        if key in (CORE_KEY, ROW_KEY) or key == "_facets_cache":
            continue
        if key in declared_names:
            continue
        if key == "_source_trace_ref":
            shell.__dict__[key] = fork_ref
            continue
        shell.__dict__[key] = cow_copy_value(value, translator)


def _reference_fill_layer_shells(
    parent: Any,
    fork: Any,
    layer_shells: dict[int, Any],
    translator: _RecordTranslator,
    fork_ref: Any,
) -> None:
    fork_layer_logs: OrderedDict[str, Any] = OrderedDict()
    consumed_shells: set[int] = set()
    fork_equivalent_ops = fork.op_equivalence_classes
    equivalent_view_memo: dict[Any, Any] = {}
    remap = translator.map
    for label, parent_layer in (parent.__dict__.get("layer_logs") or {}).items():
        parent_layer_id = id(parent_layer)
        if parent_layer_id in consumed_shells:
            shell = state_new(type(parent_layer))
        else:
            shell = layer_shells[parent_layer_id]
            consumed_shells.add(parent_layer_id)
        shell_dict = shell.__dict__
        for key, value in parent_layer.__dict__.items():
            if key == "ops" or key == "_source_trace_ref" or key == "_facets_cache":
                continue
            shell_dict[key] = cow_copy_value(value, translator)
        shell.source_trace = fork
        parent_ops = parent_layer.__dict__.get("ops")
        if parent_ops is not None:
            shell.ops = OpAccessor(
                OrderedDict(
                    (call_index, remap.get(id(layer_pass), layer_pass))
                    for call_index, layer_pass in parent_ops.items()
                )
            )
        equivalence_class = getattr(shell, "equivalence_class", None)
        if equivalence_class in fork_equivalent_ops:
            view = equivalent_view_memo.get(equivalence_class)
            if view is None:
                shell.equivalent_ops = fork_equivalent_ops[equivalence_class]
                view = shell.__dict__.get("equivalent_ops", fork_equivalent_ops[equivalence_class])
                equivalent_view_memo[equivalence_class] = view
            else:
                shell.equivalent_ops = view
        fork_layer_logs[label] = shell
    fork.layer_logs = fork_layer_logs


_REFERENCES = {
    "_flat_container_copy": _reference_flat_container_copy,
    "_fill_detached_record": _reference_fill_detached_record,
    "_copy_record_extras": _reference_copy_record_extras,
    "_fill_layer_shells": _reference_fill_layer_shells,
}


# Structure descriptions: types, atomic values, aliasing, and for every object
# whether it is the record shell, one of the inputs (by walk position), or new.


class _Record:
    """Stands in for a parent record facade the memo or translator maps."""


class _Str(str):
    pass


class _Dict(dict):
    pass


def _input_index(*values: Any) -> dict[int, int]:
    index: dict[int, int] = {}

    def walk(item: Any, depth: int) -> None:
        if type(item) in _ATOMIC_COPY_TYPES or id(item) in index:
            return
        index[id(item)] = len(index)
        if depth > 8:
            return
        if isinstance(item, dict):
            for key, member in item.items():
                walk(key, depth + 1)
                walk(member, depth + 1)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for member in item:
                walk(member, depth + 1)

    for value in values:
        walk(value, 0)
    return index


def _describe(
    item: Any, inputs: dict[int, int], shells: dict[int, str], seen: dict[int, int]
) -> Any:
    if id(item) in shells:
        return shells[id(item)]
    if item is _NO_FLAT_COPY:
        return "no-flat-copy"
    if item is _MISSING:
        return "missing"
    if type(item) in _ATOMIC_COPY_TYPES:
        if isinstance(item, float) and math.isnan(item):
            return ("float", "nan")
        return (type(item).__name__, item)
    key = id(item)
    if key in seen:
        return ("alias", seen[key])
    seen[key] = len(seen)
    origin = ("input", inputs[key]) if key in inputs else "new"
    body: Any = None
    if isinstance(item, dict):
        body = [
            (_describe(k, inputs, shells, seen), _describe(v, inputs, shells, seen))
            for k, v in item.items()
        ]
        factory = getattr(item, "default_factory", None)
        body.append(("factory", repr(factory)))
    elif isinstance(item, (list, tuple, set, frozenset)):
        body = [_describe(member, inputs, shells, seen) for member in item]
    return (type(item).__name__, origin, body)


def _describe_memo(
    memo: dict[Any, Any],
    inputs: dict[int, int],
    shells: dict[int, str],
    seen: dict[int, int],
    parent: Any,
) -> list[Any]:
    rows = []
    for key, value in memo.items():
        if key == id(memo):
            label: Any = "keep-alive"
        elif key == id(parent):
            label = "parent"
        else:
            label = ("input", inputs.get(key, "unknown"))
        rows.append((label, _describe(value, inputs, shells, seen)))
    return rows


def _flat_cases() -> Iterator[Any]:
    yield "atomic_dict", lambda p: {"a": 1, "b": "x", "c": None, "d": 2.5, "e": True, "f": b"z"}
    yield "atomic_exotic", lambda p: {"n": float("nan"), "c": 1j, True: 0, 0: False}
    yield "ordered_with_record", lambda p: OrderedDict([("a", p), ("b", 3)])
    yield "list_mixed", lambda p: [1, "two", p, None, p]
    yield "tuple_atomic", lambda p: ("a", 1, None)
    yield "tuple_record", lambda p: ("a", p)
    yield "tuple_of_atomic_tuples", lambda p: (("a", 1), ("b", 2))
    yield "tuple_nested_record", lambda p: (("a", 1), ("b", p))
    yield "nested", lambda p: {"k": [1, 2, p], "j": ("x", "y"), "m": {"z": p}}
    yield "set_str", lambda p: {"s", "t", "u"}
    yield "set_of_tuples", lambda p: {("a", 1), ("b", 2)}
    yield "frozenset_int", lambda p: frozenset({1, 2, 3})
    yield "frozenset_record", lambda p: frozenset({p, "x"})
    yield "defaultdict_list", lambda p: defaultdict(list, {"a": ["x", p], "b": []})
    yield "defaultdict_atomic", lambda p: defaultdict(int, {"a": 1, "b": 2})
    yield "deep_atomic", lambda p: {"k": ("a", ("b", 1))}
    yield "long_atomic_list", lambda p: list(range(300)) + ["x"] * 50
    yield "empty_dict", lambda p: {}
    yield "empty_list", lambda p: []
    yield "empty_tuple", lambda p: ()
    yield "empty_set", lambda p: set()
    yield "empty_frozenset", lambda p: frozenset()
    yield "empty_ordered", lambda p: OrderedDict()
    yield "empty_defaultdict", lambda p: defaultdict(int)
    yield "list_of_empties", lambda p: [[], {}, (), set()]
    yield "tuple_key", lambda p: {("tuple", "key"): 1}
    yield "unmapped_record", lambda p: {"a": _Record()}
    yield "unmapped_in_tuple", lambda p: ("a", _Record())
    yield "tensor", lambda p: {"t": torch.zeros(1)}
    yield "too_deep", lambda p: [[[["too deep"]]]]
    yield "str_subclass_member", lambda p: ["a", _Str("b")]
    yield "str_subclass_key", lambda p: {_Str("k"): 1}
    yield "dict_subclass_member", lambda p: {"a": _Dict(b=1)}
    yield "partial_factory", lambda p: defaultdict(functools.partial(list), a=[1])
    yield "callable_member", lambda p: [len, "x"]

    def aliased(p: Any) -> Any:
        inner = ["x", p]
        return {"a": inner, "b": inner, "c": (inner,)}

    yield "aliased", aliased

    def order_sensitive(p: Any) -> Any:
        # The first member reaches ``inner`` one level too deep; the second
        # copies it, so the memo ends up holding ``inner``'s copy either way.
        inner = ["x"]
        return [[[inner]], inner, [inner]]

    yield "order_sensitive", order_sensitive

    def failing_after_copies(p: Any) -> Any:
        inner = ["x", p]
        return [inner, {"t": torch.zeros(1)}, inner]

    yield "failing_after_copies", failing_after_copies

    def ordered_with_attribute(p: Any) -> Any:
        value = OrderedDict(a=1)
        value.note = "x"  # type: ignore[attr-defined]
        return value

    yield "ordered_with_attribute", ordered_with_attribute


_FLAT_CASES = list(_flat_cases())


@pytest.mark.parametrize(
    "build", [case for _, case in _FLAT_CASES], ids=[n for n, _ in _FLAT_CASES]
)
def test_flat_copy_matches_the_former_copy(build: Callable[[Any], Any]) -> None:
    """Same structure, same shared objects, same memo, as the former flat copy."""

    parent, shell = _Record(), _Record()
    value = build(parent)
    inputs = _input_index(value, parent)
    results = []
    for copier in (_reference_flat_container_copy, trace_fork._flat_container_copy):
        memo = trace_fork._ForkMemo()
        dict.update(memo, {id(parent): shell})
        first = copier(value, memo, _FLAT_COPY_DEPTH)
        again = copier(value, memo, _FLAT_COPY_DEPTH)
        seen: dict[int, int] = {}
        shells = {id(shell): "shell"}
        results.append(
            (
                _describe(first, inputs, shells, seen),
                _describe(again, inputs, shells, seen),
                _describe_memo(memo, inputs, shells, seen, parent),
            )
        )
    assert results[1] == results[0]


def test_flat_copy_sees_the_atomic_types_the_former_copy_saw() -> None:
    """The fast path's type test is the former per-member test, as a C-level lookup."""

    assert trace_fork._is_atomic_copy_type.__self__ is _ATOMIC_COPY_TYPES
    for item in (1, "a", b"b", 2.5, True, None, 1j, _Str("s"), _Record(), [], ()):
        assert trace_fork._is_atomic_copy_type(type(item)) is (type(item) in _ATOMIC_COPY_TYPES)


class _FakeStore:
    """A record store whose cells, pooled cells and compacted labels the test sets."""

    def __init__(self, cells: list[Any], singletons: dict[int, str]) -> None:
        self.cells = cells
        self.singletons = singletons

    def cell_get(self, row: int, fid: int) -> Any:
        assert row == 3
        return self.cells[fid]

    def compacted_singleton(self, row: int, fid: int, value: Any) -> bool:
        return self.singletons.get(fid) is value


_FAKE_LAYOUT = OpStoreLayout(tuple(f"f{i}" for i in range(16)))


class _FakeRecord:
    _TL_LAYOUT = _FAKE_LAYOUT


def _fake_record(parent: Any) -> tuple[_FakeRecord, list[Any]]:
    label = "layer_3"
    cells = [
        1,
        "plain",
        label,
        None,
        _MISSING,
        PooledCell.from_value(["a", ("b", 1)]),
        PooledCell.from_value({"k": [1, 2]}),
        [1, parent, {"x": parent}],
        {"a": (parent, 1)},
        ("a", 1),
        (parent,),
        frozenset({1, 2}),
        {1, 2},
        torch.zeros(2),
        2.5,
        parent,
    ]
    record = _FakeRecord()
    record.__dict__[CORE_KEY] = _FakeStore(cells, {2: label})
    record.__dict__[ROW_KEY] = 3
    return record, cells


def test_detached_record_fill_matches_the_former_fill() -> None:
    """Every decoded, copied and shared cell equals the former fill's."""

    parent, shell = _Record(), _Record()
    record, cells = _fake_record(parent)
    inputs = _input_index(cells, parent)
    results = []
    for fill in (_reference_fill_detached_record, trace_fork._fill_detached_record):
        translator = _RecordTranslator()
        translator.add(parent, shell)
        fork_record = _FakeRecord()
        fill(record, fork_record, translator)
        detached = fork_record.__dict__[CORE_KEY]
        assert type(detached) is DetachedOpStore and fork_record.__dict__[ROW_KEY] == 0
        assert detached.layout is _FAKE_LAYOUT
        filled = [detached.cell_get(0, fid) for fid in range(_FAKE_LAYOUT.n_fields)]
        results.append(_describe(filled, inputs, {id(shell): "shell"}, {}))
    assert results[1] == results[0]


def test_record_extras_copy_matches_the_former_copy() -> None:
    """Record extras: atomic values shared, containers rebuilt, records translated."""

    parent, shell = _Record(), _Record()
    record = _Record()
    record.__dict__.update(
        {
            "count": 3,
            "name": "fc1",
            "nothing": None,
            "flags": [True, parent],
            "table": {"a": ("b", parent)},
            "tensor": torch.ones(1),
            "_source_trace_ref": "parent-ref",
            "_facets_cache": {"dropped": 1},
            "record": parent,
        }
    )
    inputs = _input_index(list(record.__dict__.values()), parent)
    results = []
    for copy_extras in (_reference_copy_record_extras, trace_fork._copy_record_extras):
        translator = _RecordTranslator()
        translator.add(parent, shell)
        fork_record = _Record()
        copy_extras(record, fork_record, translator, "fork-ref")
        results.append(
            [
                (key, _describe(value, inputs, {id(shell): "shell"}, {}))
                for key, value in fork_record.__dict__.items()
            ]
        )
    assert results[1] == results[0]


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.norm = nn.LayerNorm(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.fc(x)) + x)


class _Stack(nn.Module):
    """Nested modules, a module called twice, and a tuple-returning step."""

    width = 6

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])
        self.head = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.blocks[0](x)
        first, second = x.split(3, dim=-1)
        return self.head(torch.cat([second, first], dim=-1))


class _Small(nn.Module):
    """PERF-7's row-major fixture."""

    width = 4

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _Long(nn.Module):
    """PERF-7's columnar fixture: past the store's columnar-transpose threshold (512 rows)."""

    width = 4

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        for _ in range(180):
            h = torch.tanh(h * 1.01 + 0.1)
        return h.sum(dim=-1)


def _capture(model_cls: type[nn.Module] = _Stack) -> Any:
    torch.manual_seed(0)
    return tl.trace(
        model_cls(),
        torch.randn(2, model_cls.width),  # type: ignore[attr-defined]
        capture=CaptureOptions(replayable=True),
    )


def _labels(trace: Any) -> dict[int, Any]:
    labels: dict[int, Any] = {id(trace): "trace"}
    for op in trace.layer_list:
        labels[id(op)] = ("op", op.layer_label)
    for label, layer in (trace.__dict__.get("layer_logs") or {}).items():
        labels.setdefault(id(layer), ("layer", label))
    modules = trace.__dict__.get("_module_logs")
    for key, module in modules._dict.items() if modules is not None else ():
        labels.setdefault(id(module), ("module", key))
        calls = getattr(getattr(module, "calls", None), "_dict", None) or {}
        for call_key, call in calls.items():
            labels.setdefault(id(call), ("call", key, call_key))
    params = trace.__dict__.get("param_logs")
    for key, param in params._dict.items() if params is not None else ():
        labels.setdefault(id(param), ("param", key))
    return labels


def _collect_ids(value: Any, out: set[int], depth: int = 0) -> None:
    if type(value) in _ATOMIC_COPY_TYPES or id(value) in out or depth > 8:
        return
    out.add(id(value))
    if isinstance(value, dict):
        for key, member in value.items():
            _collect_ids(key, out, depth + 1)
            _collect_ids(member, out, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for member in value:
            _collect_ids(member, out, depth + 1)


def _normalize(value: Any, labels: dict[int, Any], parent_ids: set[int], depth: int = 0) -> Any:
    """Structure with records named, and every other object marked shared or own."""

    if id(value) in labels:
        return labels[id(value)]
    if type(value) in _ATOMIC_COPY_TYPES:
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return value
    if value is _MISSING:
        return "missing"
    shared = "shared" if id(value) in parent_ids else "own"
    if depth > 8:
        return (type(value).__name__, shared)
    if type(value) in (dict, OrderedDict, defaultdict):
        return (
            type(value).__name__,
            shared,
            [(k, _normalize(v, labels, parent_ids, depth + 1)) for k, v in value.items()],
        )
    if type(value) in (list, tuple, set, frozenset):
        members = [_normalize(v, labels, parent_ids, depth + 1) for v in value]
        if type(value) in (set, frozenset):
            members = sorted(map(repr, members))
        return (type(value).__name__, shared, members)
    if isinstance(value, torch.Tensor):
        return ("tensor", shared, tuple(value.shape))
    return (type(value).__name__, shared)


def _fork_picture(parent: Any, fork: Any) -> dict[str, Any]:
    """Trace fields, module record cells and Layer shadow dicts of one fork."""

    labels = _labels(fork)
    parent_ids: set[int] = set()
    for _name, value in state_items(parent):
        _collect_ids(value, parent_ids)
    for layer in (parent.__dict__.get("layer_logs") or {}).values():
        _collect_ids(list(layer.__dict__.values()), parent_ids)
    picture: dict[str, Any] = {}
    for name, value in state_items(fork):
        picture[f"field:{name}"] = _normalize(value, labels, parent_ids)
    modules = fork.__dict__["_module_logs"]
    for key, module in modules._dict.items():
        store = module.__dict__[CORE_KEY]
        if type(store) is DetachedOpStore:
            cells = [store.cell_get(0, fid) for fid in range(store.layout.n_fields)]
            picture[f"module-cells:{key}"] = _normalize(cells, labels, parent_ids)
        extras = {k: v for k, v in module.__dict__.items() if k not in (CORE_KEY, ROW_KEY)}
        picture[f"module-extras:{key}"] = _normalize(extras, labels, parent_ids)
    for label, layer in fork.layer_logs.items():
        shadow = {k: v for k, v in layer.__dict__.items() if k not in ("ops", "source_trace")}
        picture[f"layer:{label}"] = _normalize(shadow, labels, parent_ids)
    return picture


@pytest.mark.parametrize(
    "model_cls",
    [_Stack, _Small, pytest.param(_Long, marks=pytest.mark.heavy)],
    ids=["stack", "row_major", "columnar"],
)
def test_a_fork_equals_one_built_by_the_former_copy_path(
    model_cls: type[nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fields, module cells and Layer shadow dicts: the same copies and the same shares."""

    trace = _capture(model_cls)
    assert (trace._trace_core.ops._columns is not None) is (model_cls is _Long), "store shape"
    current = {name: getattr(trace_fork, name) for name in _REFERENCES}
    for name, reference in _REFERENCES.items():
        monkeypatch.setattr(trace_fork, name, reference)
    control, former = trace.fork(), trace.fork()
    for name, function in current.items():
        monkeypatch.setattr(trace_fork, name, function)
    new = trace.fork()
    pictures = [_fork_picture(trace, fork) for fork in (control, former, new)]
    assert set(pictures[0]) == set(pictures[1]) == set(pictures[2])
    assert any(key.startswith("module-cells:") for key in pictures[0]), "no detached module"
    # Entries that differ between two forks built by the same code (fork names,
    # the parent's growing history) are not this change's doing; every other
    # entry must be identical between the former and the new code.
    stable = [key for key in pictures[0] if pictures[0][key] == pictures[1][key]]
    assert len(stable) > len(pictures[0]) - 5
    mismatches = [key for key in stable if pictures[2][key] != pictures[1][key]]
    assert not mismatches, mismatches[:5]
    # Values: fork outputs equal the parent's, and the same edit on a fork built
    # by each code gives tensor-equal results on every op.
    # The last linear op: a single-pass layer in every fixture.
    label = [op.layer_label for op in trace.layer_list if "linear" in op.layer_label][-1]
    edited = [
        fork.fork().do(label, tl.scale(0.5), intervention=InterventionOptions(engine="replay"))
        for fork in (former, new)
    ]
    for former_op, new_op in zip(edited[0].layer_list, edited[1].layer_list, strict=True):
        assert former_op.layer_label == new_op.layer_label
        if isinstance(former_op.out, torch.Tensor):
            assert torch.equal(former_op.out, new_op.out), former_op.layer_label
    # Isolation: fork tables are their own.
    for fork in (former, new):
        for parent_op, fork_op in zip(trace.layer_list, fork.layer_list, strict=True):
            if isinstance(parent_op.out, torch.Tensor):
                assert torch.equal(parent_op.out, fork_op.out)
        assert fork.layer_dict_all_keys is not trace.layer_dict_all_keys
        fork.layer_dict_all_keys["probe"] = None
        assert "probe" not in trace.layer_dict_all_keys


def _fork_events(trace: Any) -> int:
    trace.fork()  # warm-up, as the benchmark counter does
    calls = 0

    def count(_frame: Any, event: str, _arg: Any) -> None:
        nonlocal calls
        if event in ("call", "c_call"):
            calls += 1

    sys.setprofile(count)
    try:
        member = trace.fork()
    finally:
        sys.setprofile(None)
    del member
    return calls


# Measured for one fork of this capture: 35,641 events on the former copy path,
# 28,274 on this one; the bar leaves about 10 percent headroom above the latter.
_FORK_EVENT_BAR = 31_000


def test_one_fork_stays_cheap() -> None:
    """Profiler events (calls and C calls) of one fork of a fixed capture."""

    events = _fork_events(_capture())
    assert events <= _FORK_EVENT_BAR, events
