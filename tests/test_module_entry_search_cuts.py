"""Module entry's cheaper search, functorch check and argument stubs give the same answers.

The tensor search matches its last breadth-first level without expanding it (that
expansion's children were never examined), so it finds the same tensors at the same
addresses while reading no attribute of a last-level object. The functorch check
resolved once per capture run answers as ``is_functorch_wrapped_tensor`` does. The
argument stubber's plain positional-tensor path returns what the general walk returns.
"""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from torchlens._utils.introspection import get_vars_of_type_from_obj
from torchlens._utils.tensor_utils import is_functorch_wrapped_tensor
from torchlens.backends.torch._functorch_check import functorch_wrapped_check
from torchlens.backends.torch._module_arg_stubs import (
    stub_module_arg_payloads,
    stub_module_call_args,
)

_Pair = namedtuple("_Pair", ["left", "right"])


class _Holder:
    """A plain object holding values in attributes, with a property that counts reads."""

    def __init__(self, **values: Any) -> None:
        self.reads = 0
        for name, value in values.items():
            setattr(self, name, value)

    @property
    def counted(self) -> int:
        self.reads += 1
        return self.reads


def _nested_inputs() -> list[Any]:
    a, b, c = torch.ones(1), torch.zeros(2), torch.full((3,), 2.0)
    param = nn.Parameter(torch.ones(2))
    return [
        [a, (b, [c, {"deep": [a]}])],
        {"x": a, "y": _Pair(b, [c, [[torch.ones(4)]]]), "p": param},
        _Holder(t=a, inner=_Holder(t=b, inner=_Holder(t=c, inner=_Holder(t=a)))),
        (a, a, [a, b], {"again": b}),
        [[[[[torch.ones(5)]]]]],
        [[[[[[torch.ones(6)]]]]]],
    ]


@pytest.mark.parametrize("search_depth", [1, 2, 3, 5])
@pytest.mark.parametrize("case", range(len(_nested_inputs())))
def test_search_matches_the_fully_expanding_walk(case: int, search_depth: int) -> None:
    """Same tensors, same order, same addresses as the walk that expands the last level."""

    obj = _nested_inputs()[case]
    kwargs = {"subclass_exceptions": [nn.Parameter], "search_depth": search_depth}
    expanding = get_vars_of_type_from_obj(
        obj, torch.Tensor, return_addresses=True, depth_exceeded_paths=[], **kwargs
    )
    searched = get_vars_of_type_from_obj(obj, torch.Tensor, return_addresses=True, **kwargs)
    assert [(id(t), a, f) for t, a, f in searched] == [(id(t), a, f) for t, a, f in expanding]


def test_search_with_repeats_finds_each_occurrence() -> None:
    a, b = torch.ones(1), torch.zeros(1)
    found = get_vars_of_type_from_obj([a, [a, b], (b,)], torch.Tensor, allow_repeats=True)
    assert [id(t) for t in found] == [id(a), id(a), id(b), id(b)]


def test_search_finds_a_tensor_on_its_last_level_without_reading_that_level() -> None:
    """An object on the last level is matched or skipped, never crawled."""

    tensor = torch.ones(2)
    holder = _Holder(t=torch.zeros(1))
    found = get_vars_of_type_from_obj([holder, tensor], torch.Tensor, search_depth=2)
    assert [id(t) for t in found] == [id(tensor)]
    assert holder.reads == 0


def test_search_still_crawls_objects_above_the_last_level() -> None:
    holder = _Holder(t=torch.zeros(1))
    found = get_vars_of_type_from_obj([holder], torch.Tensor, search_depth=3)
    assert [id(t) for t in found] == [id(holder.t)]
    assert holder.reads >= 1


def test_depth_exceeded_frontier_is_still_reported() -> None:
    """A caller reading the frontier gets the unexplored entries, as before."""

    deep = torch.ones(1)
    frontier: list[str] = []
    found = get_vars_of_type_from_obj(
        [[[deep]]], torch.Tensor, search_depth=2, depth_exceeded_paths=frontier
    )
    assert found == []
    assert len(frontier) == 1
    reached: list[str] = []
    found = get_vars_of_type_from_obj(
        [[[deep]]], torch.Tensor, search_depth=4, depth_exceeded_paths=reached
    )
    assert [id(t) for t in found] == [id(deep)]
    assert reached == []


def test_functorch_check_answers_as_is_functorch_wrapped_tensor() -> None:
    plain = torch.ones(3)
    check = functorch_wrapped_check(run_nonce=-101)
    seen: list[tuple[bool, bool]] = []

    def probe(x: torch.Tensor) -> torch.Tensor:
        answer = False if check is None else check(x)
        seen.append((answer, is_functorch_wrapped_tensor(x)))
        return x * 2

    torch.func.vmap(probe)(torch.ones(2, 3))
    plain_answer = False if check is None else check(plain)
    assert plain_answer is is_functorch_wrapped_tensor(plain) is False
    assert seen and all(mine == reference for mine, reference in seen)


def test_functorch_check_is_reused_within_a_run_and_resolved_per_run() -> None:
    first = functorch_wrapped_check(run_nonce=-202)
    assert functorch_wrapped_check(run_nonce=-202) is first
    second = functorch_wrapped_check(run_nonce=-203)
    if first is not None:
        assert second is not first


@pytest.mark.parametrize(
    "identity_mode", [None, "torch_module", "pytree_module", "function_root", "object_module"]
)
def test_plain_tensor_call_stubs_as_the_general_walk(identity_mode: str | None) -> None:
    a, b = torch.ones(2, 3), torch.zeros(4, dtype=torch.int64)
    param = nn.Parameter(torch.ones(1))
    labels: dict[int, str | None] = {id(a): "linear_1_1"}
    trace = SimpleNamespace(_options_module_identity_mode=identity_mode)
    for args in [(a,), (a, b, param), ()]:
        stub_args, stub_kwargs = stub_module_call_args(args, {}, trace, labels)
        assert stub_args == stub_module_arg_payloads(
            args, labels=labels, identity_mode=identity_mode
        )
        assert type(stub_args) is tuple
        assert stub_kwargs == {}


@pytest.mark.parametrize("identity_mode", [None, "function_root"])
def test_mixed_calls_still_take_the_general_walk(identity_mode: str | None) -> None:
    a = torch.ones(2)
    trace = SimpleNamespace(_options_module_identity_mode=identity_mode)
    cases: list[tuple[Any, Any]] = [
        ((a, 3), {}),
        ((a,), {"mask": None}),
        (_Pair(a, a), {}),
        ((a,), None),
        ([a], {}),
    ]
    for args, kwargs in cases:
        stub_args, stub_kwargs = stub_module_call_args(args, kwargs, trace)
        assert stub_args == stub_module_arg_payloads(args, identity_mode=identity_mode)
        assert type(stub_args) is type(args)
        assert stub_kwargs == stub_module_arg_payloads(kwargs, identity_mode=identity_mode)
