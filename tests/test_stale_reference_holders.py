"""Stale pre-wrap torch references: every holder shape is disclosed and rescued.

A model built before the process's first capture holds pristine torch
functions (``self.act = F.gelu`` resolved before ``wrap_torch()``). Capture
never rewrites the user's objects; instead an escape signal triggers the
rescue re-run, whose ``TorchFunctionMode`` net redirects ANY stale reference
regardless of where it is held. The signal is the gap: a stale op whose
output a module RETURNS (transformers' ``GELUActivation`` holds ``F.gelu`` and
returns ``self.act(x)``) was tagged by the module-exit boundary before any
consumer saw it, so it became a clean ``internalsource`` node with no warning
and no rescue. These rows pin each holder shape returned from a module, plus
the negatives that must stay quiet.
"""

from __future__ import annotations

import functools
import sys
import types
from collections import namedtuple
from collections.abc import Callable, Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

_PROVENANCE = "no graph/source provenance"
_Acts = namedtuple("_Acts", ["first"])


@pytest.fixture()
def raw() -> Iterator[types.SimpleNamespace]:
    """Pristine pre-wrap torch callables; rewrap and drop temp modules afterwards."""

    unwrap_torch()
    env = types.SimpleNamespace(
        gelu=torch.nn.functional.gelu, tanh=torch.tanh, relu=torch.relu, temp=[]
    )
    assert not is_decorated_function(env.gelu)
    try:
        yield env
    finally:
        for name in env.temp:
            sys.modules.pop(name, None)
        wrap_torch()


class _Holder(nn.Module):
    """Linear, an activation submodule, Linear: the BERT intermediate/output shape."""

    def __init__(self, act: nn.Module) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.act = act
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fc2(act(fc1(x)))."""

        return self.fc2(self.act(self.fc1(x)))


class _GELUActivationLike(nn.Module):
    """Copy of transformers' ``GELUActivation``: stores the function, returns its output."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.act = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the stored activation's output (the module-exit shape)."""

        return self.act(x)


class _CallFn(nn.Module):
    """Return ``self.fn(x)`` for an arbitrary held callable."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the held callable's output."""

        return self.fn(x)


class _NestedContainer(nn.Module):
    """Stale function two container levels deep."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.table = {"acts": [fn]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the nested held function's output."""

        return self.table["acts"][0](x)


class _NamedTupleHolder(nn.Module):
    """Stale function inside a namedtuple attribute."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.acts = _Acts(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the namedtuple-held function's output."""

        return self.acts.first(x)


def _closure(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def apply(v: torch.Tensor) -> torch.Tensor:
        return fn(v)

    return apply


def _default_arg_module(fn: Callable[..., torch.Tensor]) -> nn.Module:
    """A class whose forward default argument was bound to ``fn`` at class creation."""

    class _DefaultArg(nn.Module):
        def forward(self, x: torch.Tensor, act: Callable[..., torch.Tensor] = fn) -> torch.Tensor:
            return act(x)

    return _DefaultArg()


def _module_global_module(env: types.SimpleNamespace) -> nn.Module:
    """``from torch.nn.functional import gelu`` in a defining module, called in forward."""

    module = types.ModuleType("_tl_stale_global_holder")
    module.__dict__.update({"gelu": env.gelu, "nn": nn, "torch": torch})
    exec(  # noqa: S102 - builds a throwaway defining module for the holder class
        "class GlobalGelu(nn.Module):\n    def forward(self, x):\n        return gelu(x)\n",
        module.__dict__,
    )
    sys.modules[module.__name__] = module
    env.temp.append(module.__name__)
    return module.GlobalGelu()


_HOLDERS: dict[str, Callable[[types.SimpleNamespace], nn.Module]] = {
    "gelu_activation_attr": lambda env: _GELUActivationLike(env.gelu),
    "closure_cell": lambda env: _CallFn(_closure(env.gelu)),
    "functools_partial": lambda env: _CallFn(functools.partial(env.gelu, approximate="none")),
    "nested_container": lambda env: _NestedContainer(env.gelu),
    "namedtuple": lambda env: _NamedTupleHolder(env.gelu),
    "default_argument": _default_arg_module,
    "module_global": _module_global_module,
}


@pytest.mark.parametrize("holder", sorted(_HOLDERS))
def test_module_returned_stale_reference_is_rescued(
    raw: types.SimpleNamespace, holder: str
) -> None:
    """Each holder shape, returned from a submodule, is disclosed and recovered."""

    model = _Holder(_HOLDERS[holder](raw))
    wrap_torch()
    with pytest.warns(UserWarning, match=r"adopted at module exit act"):
        trace = tl.trace(model, torch.randn(2, 4))

    assert "gelu" in [op.func_name for op in trace.ops]
    assert "internalsource" not in [layer.layer_type for layer in trace.layer_list]
    assert trace.rescue_rerun is not None and trace.rescue_rerun["recovered"] is True
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "mode_rescue_rerun"


def test_rescue_never_rewrites_the_held_reference(raw: types.SimpleNamespace) -> None:
    """The fix is a signal, not a mutation: the user's stored function keeps its identity."""

    act = _GELUActivationLike(raw.gelu)
    model = _Holder(act)
    wrap_torch()
    with pytest.warns(UserWarning, match=_PROVENANCE):
        tl.trace(model, torch.randn(2, 4))

    assert act.act is raw.gelu


def test_iql_output_activation_returned_by_the_root_is_rescued(
    raw: types.SimpleNamespace,
) -> None:
    """IQL shape: a stale tanh produces the ROOT module's output (root module exit)."""

    class IqlLike(nn.Module):
        def __init__(self, out_act: Callable[..., torch.Tensor]) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)
            self.output_activation = out_act

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.output_activation(self.fc(x))

    model = IqlLike(raw.tanh)
    wrap_torch()
    with pytest.warns(UserWarning, match=r"adopted at module exit"):
        trace = tl.trace(model, torch.randn(2, 4))

    assert "tanh" in [op.func_name for op in trace.ops]
    assert trace.capture_verification_reason == "mode_rescue_rerun"


def test_clean_module_return_stays_quiet() -> None:
    """Negative: a wrapped activation returned from a module raises no signal."""

    wrap_torch()
    model = _Holder(_GELUActivationLike(torch.nn.functional.gelu))
    trace = tl.trace(model, torch.randn(2, 4))

    assert "gelu" in [op.func_name for op in trace.ops]
    assert trace.rescue_rerun is None
    assert trace.capture_verification_reason != "mode_rescue_rerun"


def test_foreign_callable_returned_from_a_module_stays_quiet() -> None:
    """Negative: a user-defined (non-torch) callable composed of live ops is untouched."""

    def my_gelu(v: torch.Tensor) -> torch.Tensor:
        return v * torch.sigmoid(1.702 * v)

    wrap_torch()
    act = _GELUActivationLike(my_gelu)
    trace = tl.trace(_Holder(act), torch.randn(2, 4))

    assert trace.rescue_rerun is None
    assert act.act is my_gelu


def test_model_owned_tensor_returned_by_a_module_is_not_an_escape() -> None:
    """Negative: returning a plain tensor attribute that existed before the forward."""

    class ReturnsCache(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cache = torch.ones(2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.cache

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = ReturnsCache()
            self.fc = nn.Linear(4, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x) + self.inner(x)

    wrap_torch()
    trace = tl.trace(Outer(), torch.randn(2, 4))

    assert trace.rescue_rerun is None
