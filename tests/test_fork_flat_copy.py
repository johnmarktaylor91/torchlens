"""The fork's flat-container copy equals ``copy.deepcopy`` under the fork memo.

A fork copies its trace-side fields under one shared deepcopy memo. Most of
those fields are flat label tables (string keys mapping to strings, ints or
record facades), and copying them element by element through deepcopy was
most of a fork's fixed cost. The flat copy builds the same containers
directly; these tests pin that it returns what deepcopy returns (the same
values, the memo's record shells, aliasing kept, an all-atomic tuple returned
as itself), that it hands everything else to deepcopy, and that a fork built
with it equals one built without it, field by field.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.data_classes._trace_fork as trace_fork
from torchlens.data_classes._state_adapter import state_items
from torchlens.options import CaptureOptions


class _Record:
    """Stand-in for a record facade already mapped to its fork shell."""


def _memo_with_shell() -> tuple[dict[Any, Any], _Record, _Record]:
    parent, shell = _Record(), _Record()
    memo = trace_fork._ForkMemo()
    dict.update(memo, {id(parent): shell})
    return memo, parent, shell


def _flat(value: Any, memo: dict[Any, Any]) -> Any:
    return trace_fork._flat_container_copy(value, memo, trace_fork._FLAT_COPY_DEPTH)


@pytest.mark.parametrize(
    "build",
    [
        lambda parent: {"a": 1, "b": "x", "c": None, "d": 2.5, "e": True, "f": b"z"},
        lambda parent: OrderedDict([("a", parent), ("b", 3)]),
        lambda parent: [1, "two", parent, None],
        lambda parent: ("a", parent),
        lambda parent: {"k": [1, 2, parent], "j": ("x", "y")},
        lambda parent: {"s", "t"},
        lambda parent: frozenset({1, 2}),
    ],
)
def test_flat_copy_equals_deepcopy(build: Any) -> None:
    memo, parent, shell = _memo_with_shell()
    deep_memo, deep_parent, deep_shell = _memo_with_shell()
    value, deep_value = build(parent), build(deep_parent)
    flat = _flat(value, memo)
    assert flat is not trace_fork._NO_FLAT_COPY
    expected = copy.deepcopy(deep_value, deep_memo)

    def normalize(item: Any, own_shell: _Record) -> Any:
        if item is own_shell:
            return "<shell>"
        if isinstance(item, dict):
            return (type(item), [(k, normalize(v, own_shell)) for k, v in item.items()])
        if isinstance(item, (list, tuple)):
            return (type(item), [normalize(v, own_shell) for v in item])
        if isinstance(item, (set, frozenset)):
            return (type(item), sorted(map(repr, item)))
        return item

    assert normalize(flat, shell) == normalize(expected, deep_shell)
    assert (flat is value) == (expected is deep_value)
    assert (id(value) in memo) == (id(deep_value) in deep_memo)


def test_nested_shared_containers_keep_their_aliasing() -> None:
    memo, _, _ = _memo_with_shell()
    inner = ["x"]
    outer = {"a": inner, "b": inner}
    flat = _flat(outer, memo)
    assert flat["a"] is flat["b"] and flat["a"] is not inner
    assert _flat(inner, memo) is flat["a"]


def test_everything_else_takes_the_deepcopy_path() -> None:
    memo, _, _ = _memo_with_shell()
    for value in (
        {("tuple", "key"): 1},
        {"a": _Record()},
        defaultdict(list, a=[1]),
        {"t": torch.zeros(1)},
        [[[["too deep"]]]],
    ):
        assert _flat(value, memo) is trace_fork._NO_FLAT_COPY


def test_a_fork_with_the_flat_copy_equals_one_without(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    twin = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    with_flat = trace.fork()
    monkeypatch.setattr(trace_fork, "_flat_container_copy", lambda *_args: trace_fork._NO_FLAT_COPY)
    without = twin.fork()
    shells = {id(op): op.layer_label for op in with_flat.layer_list}
    shells.update({id(op): op.layer_label for op in without.layer_list})

    def normalize(item: Any) -> Any:
        if id(item) in shells:
            return ("op", shells[id(item)])
        if type(item) in (dict, OrderedDict):
            return (type(item), [(k, normalize(v)) for k, v in item.items()])
        if type(item) in (list, tuple):
            return (type(item), [normalize(v) for v in item])
        if type(item) in (set, frozenset):
            return (type(item), sorted(map(repr, map(normalize, item))))
        if isinstance(item, (str, bytes, int, float, bool, type(None))):
            return item
        return type(item)

    first = dict(state_items(with_flat))
    second = dict(state_items(without))
    assert set(first) == set(second)
    for name in sorted(first):
        if name in ("trace_label", "state_history"):
            continue
        assert normalize(first[name]) == normalize(second[name]), name
    # Isolation: the fork's tables are its own.
    assert with_flat.layer_dict_all_keys is not trace.layer_dict_all_keys
    with_flat.layer_dict_all_keys["probe"] = None
    assert "probe" not in trace.layer_dict_all_keys
