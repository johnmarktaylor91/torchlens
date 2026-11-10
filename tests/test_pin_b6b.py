"""Pin ``Param``: a parameter record a user holds keeps its trace; the trace keeps no ``Param``.

Every door that hands out a ``Param`` is read in one chained expression (no
variable holds the trace) and compared with the same read through a held
trace. The trace's own containers hold param rows, so a view nobody keeps is
freed, and the next read builds it again with the same extras. Copied and
restored records stay orphans, as before. The sites that keyed parameters by
``id()`` are tested with an ``id`` that returns one value for every ``Param``.
"""

from __future__ import annotations

import builtins
import copy
import gc
import pickle
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._param_rows import ParamMap, ParamSequence, ParamTuple
from torchlens._data_classes.param import Param
from torchlens._trace_core.record_rows import ROW_KEY, RecordCellField


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv(x))).mean((2, 3))
        for _ in range(2):
            x = self.fc(x)
        return x


def _trace() -> Any:
    torch.manual_seed(0)
    return tl.trace(_Net(), torch.randn(2, 1, 6, 6))


def _first(container: Any) -> Any:
    return next(iter(container))


#: Every door that hands out a Param, as one expression from a trace.
DOORS: dict[str, Callable[[Any], Any]] = {
    "trace.params[address]": lambda t: t.params["fc.weight"],
    "trace.params[index]": lambda t: t.params[2],
    "iter(trace.params)": lambda t: _first(t.params),
    "trace.params.values()": lambda t: t.params.values()[0],
    "trace.params.trainable": lambda t: _first(t.params.trainable),
    "trace.modules[address].params": lambda t: _first(t.modules["fc"].params),
    "trace.modules[address].params.values()": lambda t: t.modules["conv"].params.values()[-1],
    "trace.layers[label].params": lambda t: _first(t.layers["linear_1"].params),
    "trace[label].params": lambda t: _first(t["conv2d_1"].params),
    "trace.layers[label].ops[1].params": lambda t: _first(t.layers["linear_1"].ops[1].params),
    "trace.layers[label].ops[0].params": lambda t: _first(t.layers["conv2d_1"].ops[0].params),
    "param.module.params": lambda t: _first(t.params["bn.weight"].module.params),
}


def _read(param: Any) -> dict[str, Any]:
    """Return what a user reads from a Param, with the trace-backed reads."""

    module = param.module
    trace = param.trace
    return {
        "address": param.address,
        "module_address": param.module_address,
        "shape": tuple(param.shape),
        "num_calls": param.num_calls,
        "used_by_ops": list(param.used_by_ops),
        "used_by_layers": list(param.used_by_layers),
        "index": param.index,
        "module": None if module is None else module.address,
        "trace_alive": trace is not None,
        "trace_params": None if trace is None else len(trace.params),
    }


@pytest.mark.parametrize("door", list(DOORS))
def test_chained_read_matches_held_trace(door: str) -> None:
    reach = DOORS[door]
    trace = _trace()
    held = _read(reach(trace))
    del trace
    chained = _read(reach(_trace()))
    assert chained == held
    assert chained["trace_alive"] and chained["module"] is not None


def test_held_param_keeps_trace_and_frees_it_without_collector() -> None:
    gc.disable()
    try:
        trace = _trace()
        trace_ref = weakref.ref(trace)
        param = trace.params["fc.bias"]
        del trace
        assert trace_ref() is not None
        assert param.trace is trace_ref()
        del param
        assert trace_ref() is None
    finally:
        gc.enable()


def test_trace_keeps_no_param() -> None:
    trace = _trace()
    core = trace.__dict__["_trace_core"]
    fid = core.ops.layout.fid_by_name["_param_logs"]
    cells = [core.ops.cell_get(row, fid, True) for row in range(len(core.ops))]
    assert all(type(cell) in (ParamSequence, ParamTuple) for cell in cells if cell)
    assert any(cells)
    stored = trace.__dict__["_param_logs"]
    assert type(stored.__dict__["_list"]) is ParamSequence
    assert type(stored.__dict__["_dict"]) is ParamMap
    assert all(
        type(value) is ParamSequence for value in trace._param_logs_by_module.values() if value
    )
    view_ref = weakref.ref(trace.params["fc.weight"])
    gc.collect()
    assert view_ref() is None
    assert trace.params["fc.weight"].address == "fc.weight"


def test_identity_across_doors_while_held() -> None:
    trace = _trace()
    param = trace.params["fc.weight"]
    assert trace.params[trace.params.keys().index("fc.weight")] is param
    assert trace.modules["fc"].params.values()[0] is param
    assert trace.layers["linear_1"].ops[0].params["fc.weight"] is param
    assert trace.layers["linear_1"].params["fc.weight"] is param


def test_rebuilt_view_keeps_extras() -> None:
    trace = _trace()
    trace.params["fc.weight"].note = "kept"
    gc.collect()
    assert trace.params["fc.weight"].note == "kept"


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, lambda p: pickle.loads(pickle.dumps(p))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_copied_param_is_an_orphan(duplicate: Callable[[Any], Any]) -> None:
    trace = _trace()
    original = trace.params["fc.weight"]
    twin = duplicate(original)
    assert twin is not original
    assert twin.address == "fc.weight"
    assert twin.trace is None
    assert twin.module is None
    assert twin.index == -1


def test_restored_trace_pins_params() -> None:
    trace = _trace()
    restored = pickle.loads(pickle.dumps(trace))
    assert type(restored.__dict__["_param_logs"].__dict__["_list"]) is ParamSequence
    param = restored.params["fc.weight"]
    assert restored.layers["linear_1"].ops[0].params["fc.weight"] is param
    assert list(restored.params) == list(restored.params.values())
    # As on the base: a restored record's back-reference is not carried by
    # the stream, so ``trace`` and ``module`` read None.
    assert param.trace is None and param.module is None
    chained = pickle.loads(pickle.dumps(_trace())).params["fc.weight"]
    assert chained.address == "fc.weight" and chained.trace is None


def test_state_streams_see_plain_containers() -> None:
    trace = _trace()
    state = trace.__getstate__()
    accessor = state["_param_logs"]
    assert type(accessor.__dict__["_list"]) is list
    assert type(accessor.__dict__["_dict"]) is dict
    op_state = trace.layers["linear_1"].ops[0].__getstate__()
    assert type(op_state["_param_logs"]) in (list, tuple)
    layer_state = trace.layers["linear_1"].__getstate__()
    assert type(layer_state["_param_logs"]) in (list, tuple)
    assert all(type(p) is Param for p in op_state["_param_logs"])
    assert type(trace.params.__getstate__()["_list"]) is list
    assert pickle.dumps(trace) == pickle.dumps(trace)


def test_fork_params_read_the_fork() -> None:
    trace = _trace()
    fork = trace.fork()
    param = fork.params["fc.weight"]
    assert param.trace is fork
    assert param is not trace.params["fc.weight"]
    assert fork.layers["linear_1"].ops[0].params["fc.weight"] is param
    assert _first(fork.modules["fc"].params) is param
    chained = _trace().fork().params["fc.weight"]
    assert chained.trace is not None and chained.module.address == "fc"


def _param_only_id(real: Callable[[Any], int]) -> Callable[[Any], int]:
    """Return an ``id`` that gives every Param the same value (forced reuse)."""

    return lambda obj: 7 if isinstance(obj, Param) else real(obj)


def test_membership_map_survives_param_id_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens._postprocess import finalization

    control = {p.address: _read(p) for p in _trace().params}
    monkeypatch.setattr(finalization, "id", _param_only_id(builtins.id), raising=False)
    forced = {p.address: _read(p) for p in _trace().params}
    assert forced == control
    co_parents = {p.address: list(p.co_parent_params) for p in _trace().params}
    monkeypatch.undo()
    assert co_parents == {p.address: list(p.co_parent_params) for p in _trace().params}


class _FakeParam:
    def __init__(self, barcode: str) -> None:
        self.barcode = barcode
        self.used_by_ops: list[str] = []
        self.used_by_layers: list[str] = []


class _FakeOp:
    def __init__(self, params: list[_FakeParam], label: str, layer_label: str) -> None:
        self._param_logs = params
        self.label = label
        self.layer_label = layer_label


def test_preview_membership_survives_param_id_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens.backends import _finalize

    monkeypatch.setattr(_finalize, "id", lambda obj: 7, raising=False)
    weight, bias = _FakeParam("w"), _FakeParam("b")
    trace = type("T", (), {"layers_with_params": {"w": [], "b": []}})()
    memo: dict[Any, Any] = {}
    _finalize._attach_param_usage(trace, _FakeOp([weight, bias], "linear_1_1", "linear_1"), memo)
    _finalize._attach_param_usage(trace, _FakeOp([weight, bias], "linear_1_2", "linear_1"), memo)
    for param in (weight, bias):
        assert param.used_by_ops == ["linear_1_1", "linear_1_2"]
        assert param.used_by_layers == ["linear_1"]
    assert trace.layers_with_params == {"w": ["linear_1"], "b": ["linear_1"]}


def _raw_cell(record: Any, name: str) -> Any:
    for cls in type(record).__mro__:
        field = cls.__dict__.get(name)
        if field is not None:
            return RecordCellField.__get__(field, record)
    raise AttributeError(name)


def test_module_child_door_keys_survive_param_id_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchlens._trace_core import record_refs

    monkeypatch.setattr(record_refs, "id", _param_only_id(builtins.id), raising=False)
    for trace in (_trace(), pickle.loads(pickle.dumps(_trace()))):
        door = trace.__dict__["_param_logs"]._dict
        for module in trace.modules:
            keys = _raw_cell(module, "params")
            if not isinstance(keys, record_refs.RowKeys):
                continue
            for (key, row), door_key in zip(keys.items(), keys.door, strict=True):
                assert door[door_key].__dict__[ROW_KEY] == row, (module.address, key)


def test_validation_identity_site_reads_held_inventory() -> None:
    """``_check_layer_param_aggregate_dedup`` keys consumed params by ``id``.

    It holds the inventory (``list(ml._param_logs)``) for the whole check, so
    every op's parameter list yields those same objects and no ``id`` can be
    reused while it runs.
    """

    from torchlens.validation._invariants_modules_params import (
        _check_layer_param_aggregate_dedup,
    )

    trace = _trace()
    inventory = list(trace._param_logs)
    held = {builtins.id(pl) for pl in inventory}
    for op in trace._layer_list:
        for pl in op._param_logs or ():
            assert builtins.id(pl) in held
            assert any(pl is kept for kept in inventory)
    del inventory, held
    gc.collect()
    _check_layer_param_aggregate_dedup(trace, "b6b")


def test_retained_state_holds_only_its_trace() -> None:
    """A state one trace handed out and someone kept never holds another trace's params."""

    first = _trace()
    kept = first.__getstate__()
    gc.disable()
    try:
        trace = _trace()
        trace_ref = weakref.ref(trace)
        pickle.dumps(trace)
        copy.deepcopy(trace.params["fc.weight"])
        del trace
        assert trace_ref() is None
    finally:
        gc.enable()
    assert type(kept["_param_logs"].__dict__["_list"]) is list


def test_no_trace_holds_a_param() -> None:
    """The only pinned paths to a ``Param`` are the transient pickle projection."""

    from tools.record_ownership.holders import blocking, is_transient

    rows = blocking(("Param",))
    assert rows
    assert all(is_transient(path, holder) for path, holder in rows.items()), sorted(rows)
