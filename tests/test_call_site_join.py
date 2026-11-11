"""The call-site table keyed on code, joined with the op's modules (``_call_site_join``).

A site key names code (code objects, files, offsets) and, for the modules the
walk reads, only which of them are equal. Ops in distinct module instances run
from one code path share a key; the cached chain is joined with each op's own
modules. These tests pin that the joined result equals the full walk (code
context and call-site grouping), that equal and distinct modules on one code
path never share a key, that such a model builds its first-time site once,
and that the code-context cache holds what a build of every op writes.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from tests.test_call_site_memo import _BuildEveryOp, _projection, _recurse, _WalkEveryOp
from torchlens.backends.torch import (
    _call_site_join as join_module,
    _ops_call_site_fields as site_module,
)

_CHAIN_KEY = "__torchlens_call_site__"


class _Acts(nn.Module):
    """``count`` distinct stateless modules called from one line (``mod_relu200``'s shape)."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(*[nn.ReLU() for _ in range(count)])

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class _Step(nn.Module):
    """A stateless module that calls ``nxt`` (set without registering it) ``depth`` times."""

    def forward(self, x: Tensor, depth: int) -> Tensor:
        if depth == 0:
            return torch.relu(x)
        return self.nxt(x, depth - 1) + 1


class _SelfAndPair(nn.Module):
    """One code path through one module twice (recursion) and through two distinct modules."""

    def __init__(self) -> None:
        super().__init__()
        self.alone = _Step()
        self.first = _Step()
        self.second = _Step()
        object.__setattr__(self.alone, "nxt", self.alone)
        object.__setattr__(self.first, "nxt", self.second)
        object.__setattr__(self.second, "nxt", self.first)

    def forward(self, x: Tensor) -> Tensor:
        for step in (self.alone, self.first, self.second, self.alone, self.first):
            x = step(x, 2)
        return x


class _Unregistered(nn.Module):
    """Modules the capture does not wrap (a plain list), called from one line inside a stateful one."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.free = [nn.ReLU() for _ in range(4)]

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        for act in self.free:
            x = act(x) - 1
        return x


_MODELS = {
    "acts": lambda: _Acts(12),
    "self_and_pair": _SelfAndPair,
    "unregistered": _Unregistered,
}


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_a_joined_site_equals_the_full_walk(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    served = _projection(_recurse(tl.trace, harness_depth, model, x))
    monkeypatch.setattr(site_module, "_CallSiteMemo", _WalkEveryOp)
    assert served == _projection(_recurse(tl.trace, harness_depth, model, x))


def _cache_view(trace: Any) -> list[Any]:
    """The code-context cache's keys in order, with the call-site chains under their keys."""

    cache = trace._code_context_cache
    assert cache
    return [
        (key, cache[key] if isinstance(key, tuple) and key and key[0] == _CHAIN_KEY else None)
        for key in cache
    ]


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_the_cache_holds_what_building_every_op_writes(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    served = _cache_view(tl.trace(model, x))
    monkeypatch.setattr(site_module, "_CallSiteMemo", _BuildEveryOp)
    assert served == _cache_view(tl.trace(model, x))


def test_distinct_modules_on_one_code_path_build_one_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first op takes the full walk, the second builds the template, the rest join."""

    counts = {"build": 0, "walk": 0, "join": 0}

    def counting(name: str, original: Any) -> Any:
        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return counted

    for name, attr in (("build", "build_from_parts"), ("walk", "_full_walk"), ("join", "join")):
        monkeypatch.setattr(site_module, attr, counting(name, getattr(site_module, attr)))
    trace = tl.trace(_Acts(40), torch.randn(2, 4))
    assert len(trace) >= 40
    assert counts["walk"] == 1
    assert counts["build"] <= 1
    assert counts["join"] >= 38


def test_equal_and_distinct_modules_on_one_code_path_never_share_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``alone`` calls itself, ``first`` calls ``second``: same code, different slot patterns."""

    original = site_module._site_key
    keys: dict[str, set[Any]] = {"alone": set(), "pair": set()}
    model = _SelfAndPair()

    def recording(memo: Any, frame: Any) -> Any:
        key = original(memo, frame)
        stack = memo.module_ws.exhaustive_module_stack
        ids = {frame_record.module_id for frame_record in stack}
        if id(model.alone) in ids and len(stack) >= 4:
            keys["alone"].add(key)
        elif {id(model.first), id(model.second)} <= ids and len(stack) >= 4:
            keys["pair"].add(key)
        return key

    monkeypatch.setattr(site_module, "_site_key", recording)
    tl.trace(model, torch.randn(2, 4))
    assert keys["alone"] and keys["pair"]
    assert not keys["alone"] & keys["pair"]


def test_a_chain_naming_a_module_outside_the_op_is_not_joined() -> None:
    chain = ((("module", 7), 0), (("f", "q", 1, b""), 3))
    raw_key = (_CHAIN_KEY, ((("module", 7), 0),))
    assert join_module.make_template(chain, (8,), raw_key) is None
    template = join_module.make_template(chain, (7,), raw_key)
    assert template is not None
    cache: dict[Any, Any] = {}
    joined = join_module.join(chain, template, (9,), cache)
    assert joined == ((("module", 9), 0), (("f", "q", 1, b""), 3))
    assert cache == {(_CHAIN_KEY, ((("module", 9), 0),)): joined}
    assert join_module.join(chain, template, (9,), cache) is joined


def test_slots_name_equal_modules_alike() -> None:
    assert join_module.slots((5, 6, 5)) == ((5, 6), (0, 1, 0))
    assert join_module.slots(()) == ((), ())
