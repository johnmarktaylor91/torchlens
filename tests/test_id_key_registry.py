"""``IdKeyRegistry`` answers every read exactly as ``WeakIdKeyDictionary`` does.

Each step runs on both mappings and compares every read; the steps cover the
write paths that must drop a cached read (set, delete, pop, setdefault, update,
popitem, clear), a key that dies, a reused id and tensors whose ``==`` is
elementwise.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

from torchlens._utils import _tensor_label
from torchlens._utils._id_key_registry import IdKeyRegistry

_MISSING = object()


def _reads(mapping: Any, keys: list[Any]) -> list[Any]:
    out = []
    for key in keys:
        out.append(("get", mapping.get(key, _MISSING)))
        out.append(("in", key in mapping))
        try:
            out.append(("item", mapping[key]))
        except KeyError:
            out.append(("item", KeyError))
    out.append(("len", len(mapping)))
    out.append(("values", sorted(map(repr, mapping.values()))))
    return out


def _both(step: Any, keys: list[Any], fast: Any, slow: Any) -> None:
    results = []
    for mapping in (fast, slow):
        try:
            results.append(("ok", step(mapping)))
        except KeyError:
            results.append(("KeyError", None))
    assert results[0] == results[1]
    assert _reads(fast, keys) == _reads(slow, keys)


def test_reads_match_after_every_write_path() -> None:
    fast, slow = IdKeyRegistry(), WeakIdKeyDictionary()
    a, b, c = nn.Linear(2, 2), nn.ReLU(), nn.Parameter(torch.ones(3))
    keys = [a, b, c, nn.Tanh()]
    steps = [
        lambda m: m.__setitem__(a, "a1"),
        lambda m: m.get(a),
        lambda m: m.__setitem__(a, "a2"),
        lambda m: m.setdefault(b, "b1"),
        lambda m: m.setdefault(b, "b2"),
        lambda m: m.__setitem__(c, "c1"),
        lambda m: m.get(c),
        lambda m: m.pop(c),
        lambda m: m.pop(c, "gone"),
        lambda m: m.__delitem__(a),
        lambda m: m.__delitem__(a),
        lambda m: m.update({a: "a3", c: "c2"}),
        lambda m: m.get(a),
        lambda m: m.popitem() and None,
        lambda m: m.clear(),
        lambda m: m.__setitem__(b, "b3"),
    ]
    for step in steps:
        _both(step, keys, fast, slow)
        _both(lambda m: m.get(b), keys, fast, slow)


def test_a_dead_key_and_a_reused_id_miss() -> None:
    fast = IdKeyRegistry()
    module = nn.Linear(2, 2)
    fast[module] = "meta"
    assert fast.get(module) == "meta"  # now cached
    old_id = id(module)
    del module
    gc.collect()
    assert len(fast) == 0 and old_id not in fast._reads
    for _ in range(200):
        other = nn.Linear(2, 2)
        assert fast.get(other) is None and other not in fast
        if id(other) == old_id:
            break


def test_parameters_are_keyed_by_identity() -> None:
    fast = IdKeyRegistry()
    p, q = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
    fast[p] = "p"
    assert fast.get(p) == "p" and fast.get(q) is None
    assert fast.get(p) == "p" and q not in fast and p in fast


@pytest.mark.parametrize("name", ["_MODULE_REGISTRY", "_PARAM_REGISTRY"])
def test_the_registries_use_it(name: str) -> None:
    assert isinstance(getattr(_tensor_label, name), IdKeyRegistry)
