"""``IdKeyRegistry`` answers every read exactly as ``WeakIdKeyDictionary`` does.

Each step runs on both mappings and compares every read; the steps cover the
write paths that must drop a cached read (set, delete, pop, setdefault, update,
popitem, clear), a key that dies, a reused id and tensors whose ``==`` is
elementwise. The capture tests check the real module and parameter registries
against their uncached lookup (``WeakIdKeyDictionary.get``) after entries are
added, removed and replaced between and within captures, and after an id is
reused.
"""

from __future__ import annotations

import copy
import gc
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

import torchlens as tl
from torchlens._utils import _tensor_label
from torchlens._utils._internal_read import IdKeyRegistry

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


def _uncached(registry: Any, key: Any) -> Any:
    return WeakIdKeyDictionary.get(registry, key)


def _assert_registries_match(model: nn.Module) -> None:
    for module in model.modules():
        registry = _tensor_label._MODULE_REGISTRY
        assert registry.get(module) is _uncached(registry, module)
        assert (module in registry) == (_uncached(registry, module) is not None)
    for param in model.parameters():
        registry = _tensor_label._PARAM_REGISTRY
        assert registry.get(param) is _uncached(registry, param)
        assert (param in registry) == (_uncached(registry, param) is not None)


class _Pair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def test_registry_reads_match_uncached_between_captures() -> None:
    model, x = _Pair(), torch.randn(2, 3)
    tl.trace(model, x)
    _assert_registries_match(model)
    # Removed: the module's and a parameter's entries go away between captures.
    _tensor_label.clear_meta(model.a)
    _tensor_label.clear_param_meta(model.a.weight)
    _assert_registries_match(model)
    # Replaced: a new metadata object under the same key.
    meta = _tensor_label.get_module_meta(model.b)
    if meta is not None:
        _tensor_label._MODULE_REGISTRY[model.b] = copy.copy(meta)
    _assert_registries_match(model)
    # Added: a module and parameters the next capture registers for the first time.
    model.c = nn.Linear(3, 3)
    tl.trace(model, x)
    _assert_registries_match(model)
    tl.trace(model, x)
    _assert_registries_match(model)


#: What ``_MutatesRegistry.forward`` saw (module level: the capture may run a copy of the model).
_SEEN: list[tuple[str, bool]] = []


class _MutatesRegistry(nn.Module):
    """Removes, re-adds, replaces and adds registry entries in the middle of its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 3)
        self.strays = [nn.Parameter(torch.ones(1))]  # in a list: not one of the model's parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        modules, params = _tensor_label._MODULE_REGISTRY, _tensor_label._PARAM_REGISTRY
        y = self.a(x)
        meta = modules.get(self.b)
        removed = modules.pop(self.b, None)
        _SEEN.append(("removed", modules.get(self.b) is _uncached(modules, self.b)))
        if removed is not None:
            modules[self.b] = copy.copy(removed)  # replaced by a copy
        _SEEN.append(("replaced", modules.get(self.b) is _uncached(modules, self.b)))
        if meta is not None:
            modules[self.b] = meta  # the capture's own object back
        stray = self.strays[0]
        params[stray] = _tensor_label.ParamMeta()
        _SEEN.append(("added", params.get(stray) is _uncached(params, stray)))
        del params[stray]
        _SEEN.append(("dropped", params.get(stray) is _uncached(params, stray)))
        _SEEN.append(("restored", modules.get(self.b) is _uncached(modules, self.b)))
        return self.b(y)


def test_registry_reads_match_uncached_within_a_capture() -> None:
    _SEEN.clear()
    model, x = _MutatesRegistry(), torch.randn(2, 3)
    tl.trace(model, x)
    tl.trace(model, x)
    assert len(_SEEN) >= 5 and all(ok for _name, ok in _SEEN), _SEEN
    _assert_registries_match(model)


def test_real_registry_misses_on_a_reused_id() -> None:
    registry = _tensor_label._MODULE_REGISTRY
    module = nn.Linear(2, 2)
    _tensor_label._ensure_module_meta(module)
    assert registry.get(module) is _uncached(registry, module) is not None
    old_id = id(module)
    del module
    gc.collect()
    assert old_id not in registry._reads
    for _ in range(200):
        other = nn.Linear(2, 2)
        assert registry.get(other) is _uncached(registry, other) is None
        assert other not in registry
        if id(other) == old_id:
            break
