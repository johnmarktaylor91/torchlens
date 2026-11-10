"""The recorded model walk equals torch's own walk, follows every change and pins nothing.

A capture session walks the model's modules, parameters and buffers many
times. ``torchlens._utils._model_walk`` records the module walk once per module
tree and hands it back while the tree is unchanged. These tests hold it to
three promises: each helper returns exactly what the torch call it replaces
returns (shared modules, ``None`` children, tied parameters, classes that walk
their own way); a module, parameter, buffer or hook changed between two traces
is seen by the second trace, one test per kind of change; and the record keeps
no model, module or tensor alive.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _model_walk as walk


class _Scaled(nn.Module):
    """Linear, activation, linear, scaled by a buffer the forward reads."""

    def __init__(self) -> None:
        """Build the submodules and the buffer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act: nn.Module = nn.ReLU()
        self.head = nn.Linear(4, 2)
        self.register_buffer("scale", torch.full((2,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the submodules in order and scale the result."""

        return self.head(self.act(self.fc(x))) * self.scale


class _Tangled(nn.Module):
    """A shared submodule, a ``None`` child, a tied weight and a buffer registered twice."""

    def __init__(self) -> None:
        """Build the tangled tree."""

        super().__init__()
        first = nn.Linear(3, 3)
        self.block = nn.Sequential(first, nn.BatchNorm1d(3))
        self.alias = first
        self.empty: nn.Module | None = None
        self.register_module("nothing", None)
        self.tied = nn.Linear(3, 3)
        self.tied.weight = first.weight
        shared = torch.ones(3)
        self.register_buffer("one", shared)
        self.register_buffer("same", shared)
        self.register_buffer("missing", None)
        self.chain = nn.Sequential(self.alias)
        self.inner = nn.ModuleDict({"a": nn.ReLU(), "b": self.chain})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use every branch once."""

        return self.tied(self.block(x)) + self.one


class _OwnWalk(nn.Sequential):
    """A container that hides its second child from ``named_modules``."""

    def named_modules(
        self, memo: Any = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Any:
        """Yield every module but the second child."""

        for name, module in super().named_modules(memo, prefix, remove_duplicate):
            if name != "1":
                yield name, module


class _EqualsAll(nn.Module):
    """A module that claims to equal every other module of its class."""

    def __eq__(self, other: object) -> bool:
        """Equal to any instance of this class."""

        return isinstance(other, _EqualsAll)

    __hash__ = nn.Module.__hash__


def _expected(model: nn.Module) -> dict[str, Any]:
    """Return torch's own walks of ``model``, as tuples."""

    return {
        "named_modules": tuple(model.named_modules()),
        "every_path": tuple(model.named_modules(remove_duplicate=False)),
        "modules": tuple(model.modules()),
        "named_parameters": tuple(model.named_parameters()),
        "every_parameter": tuple(model.named_parameters(remove_duplicate=False)),
        "parameters": tuple(model.parameters()),
        "named_buffers": tuple(model.named_buffers()),
        "every_buffer": tuple(model.named_buffers(remove_duplicate=False)),
        "own": tuple(
            (tuple(m.named_parameters(recurse=False)), tuple(m.named_buffers(recurse=False)))
            for m in model.modules()
        ),
    }


def _recorded(model: nn.Module) -> dict[str, Any]:
    """Return the helpers' walks of ``model``."""

    return {
        "named_modules": walk.model_named_modules(model),
        "every_path": walk.model_named_modules(model, remove_duplicate=False),
        "modules": walk.model_modules(model),
        "named_parameters": walk.model_named_parameters(model),
        "every_parameter": walk.model_named_parameters(model, remove_duplicate=False),
        "parameters": walk.model_parameters(model),
        "named_buffers": walk.model_named_buffers(model),
        "every_buffer": walk.model_named_buffers(model, remove_duplicate=False),
        "own": tuple(
            (walk.own_named_parameters(m), walk.own_named_buffers(m)) for m in model.modules()
        ),
    }


def _assert_same(model: nn.Module) -> None:
    """Assert every helper matches torch, item by item and object by object."""

    expected, recorded = _expected(model), _recorded(model)
    for name, want in expected.items():
        got = recorded[name]
        assert len(got) == len(want), name
        for got_item, want_item in zip(got, want, strict=True):
            assert _same_objects(got_item, want_item), (name, got_item, want_item)


def _same_objects(left: Any, right: Any) -> bool:
    """Compare nested tuples, strings by value and everything else by identity."""

    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _same_objects(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str):
        return left == right
    return left is right


@pytest.mark.parametrize("build", [_Scaled, _Tangled], ids=["plain", "tangled"])
def test_helpers_equal_torch_walks_and_stay_equal_on_reuse(build: Any) -> None:
    """Every helper equals the torch walk it replaces, on the first read and on reuse."""

    model = build()
    _assert_same(model)
    _assert_same(model)


def test_a_class_that_walks_its_own_way_gets_the_live_walk() -> None:
    """A tree with an overridden walk method or module ``__eq__`` is never served from the record."""

    own = _OwnWalk(nn.ReLU(), nn.Tanh())
    hidden = nn.Sequential(nn.Linear(2, 2), own)
    _assert_same(hidden)
    assert "1" not in [name for name, _module in walk.model_named_modules(hidden)]
    own.append(nn.Sigmoid())
    _assert_same(hidden)

    twins = nn.Sequential(_EqualsAll(), _EqualsAll())
    _assert_same(twins)
    replaced = _EqualsAll()
    twins[0] = replaced
    assert walk.model_modules(twins)[1] is replaced
    _assert_same(twins)


def test_every_tree_change_is_walked_afresh() -> None:
    """Adding, removing, replacing, renaming and moving modules each change the next read."""

    model = _Tangled()
    _assert_same(model)
    model.extra = nn.Linear(3, 3)
    _assert_same(model)
    del model.extra
    _assert_same(model)
    model.block[1] = nn.LayerNorm(3)
    _assert_same(model)
    model.inner["c"] = model.inner.pop("a")
    _assert_same(model)
    model.chain.append(model.tied)
    _assert_same(model)
    model.nothing = nn.ReLU()
    _assert_same(model)


def test_parameters_and_buffers_are_read_live_on_an_unchanged_tree() -> None:
    """A swapped parameter or buffer shows in the next read although the tree is the same."""

    model = _Scaled()
    walk.model_modules(model)
    model.fc.weight = nn.Parameter(torch.zeros(4, 4))
    model.scale = torch.zeros(2)
    model.register_buffer("late", torch.ones(1))
    model.fc.register_parameter("bias", None)
    _assert_same(model)
    assert dict(walk.model_named_parameters(model))["fc.weight"] is model.fc.weight
    assert dict(walk.model_named_buffers(model))["scale"] is model.scale
    assert "fc.bias" not in dict(walk.model_named_parameters(model))


def test_the_record_keeps_nothing_alive() -> None:
    """After its model is dropped, the record pins no model, module or tensor."""

    model = _Scaled()
    walk.model_named_parameters(model)
    walk.model_named_modules(model, remove_duplicate=False)
    refs: list[weakref.ref[Any]] = [weakref.ref(model), weakref.ref(model.fc)]
    refs += [weakref.ref(model.fc.weight), weakref.ref(model.scale)]
    del model
    gc.collect()
    assert all(ref() is None for ref in refs)


def _traced_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Trace ``model`` on ``x`` and return the saved value of its one output."""

    trace = tl.trace(model, x)
    (output_op,) = trace.output_ops
    return output_op.out.detach().clone()


def test_added_and_removed_submodules_are_traced_as_they_are() -> None:
    """A submodule added, then removed, between traces changes each later trace's modules."""

    torch.manual_seed(0)
    model = _Scaled().eval()
    x = torch.randn(3, 4)
    first = tl.trace(model, x)
    assert first.modules["act"].class_name == "ReLU"

    model.act = nn.Sequential(nn.Tanh(), nn.ReLU())
    second = tl.trace(model, x)
    assert second.modules["act.0"].class_name == "Tanh"
    assert second.modules["act.1"].class_name == "ReLU"
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))

    del model.act[1]
    third = tl.trace(model, x)
    assert "act.1" not in {module.address for module in third.modules}
    assert "relu" not in {op.func_name for op in third.ops}
    assert torch.equal(_traced_output(model, x), model(x))


def test_reassigned_buffer_is_traced_with_its_new_value() -> None:
    """A buffer reassigned between traces is the one the second trace uses."""

    torch.manual_seed(0)
    model = _Scaled().eval()
    x = torch.randn(3, 4)
    before = _traced_output(model, x)
    model.scale = torch.full((2,), -3.0)
    after = _traced_output(model, x)
    assert torch.equal(after, model(x))
    assert not torch.equal(after, before)


def test_added_buffer_and_parameter_are_traced() -> None:
    """A parameter and a buffer registered between traces appear in the second trace."""

    torch.manual_seed(0)
    model = _Scaled().eval()
    x = torch.randn(3, 4)
    first = tl.trace(model, x)
    assert "head.extra" not in {param.address for param in first.params}
    model.head.register_parameter("extra", nn.Parameter(torch.ones(1)))
    model.fc.register_buffer("late", torch.ones(1))
    second = tl.trace(model, x)
    assert "head.extra" in {param.address for param in second.params}
    assert torch.equal(_traced_output(model, x), model(x))


def test_swapped_parameter_is_traced_with_its_new_value() -> None:
    """A parameter replaced between traces is the one the second trace records and uses."""

    torch.manual_seed(0)
    model = _Scaled().eval()
    x = torch.randn(3, 4)
    before = _traced_output(model, x)
    model.fc.weight = nn.Parameter(torch.zeros(4, 4))
    second = tl.trace(model, x)
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))
    assert not torch.equal(output_op.out, before)
    assert torch.count_nonzero(second.params["fc.weight"].value) == 0


def test_hook_added_between_traces_runs() -> None:
    """A forward hook added on an unchanged tree between traces runs in the second trace."""

    torch.manual_seed(0)
    model = _Scaled().eval()
    x = torch.randn(3, 4)
    before = _traced_output(model, x)
    calls: list[str] = []

    def plus_one(_module: nn.Module, _inputs: Any, out: torch.Tensor) -> torch.Tensor:
        """Note the call and shift the output."""

        calls.append("hook")
        return out + 1

    model.act.register_forward_hook(plus_one)
    after = _traced_output(model, x)
    assert calls == ["hook"]
    assert not torch.equal(after, before)
    assert torch.equal(after, model(x))
