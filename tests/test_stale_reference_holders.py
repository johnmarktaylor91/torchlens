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

Preparation first rebinds the holder shapes it can reach (attributes,
``functools.partial``, closures, default arguments, exact builtin containers,
namedtuples) to the wrappers for the capture's duration and restores the
user's objects afterwards, so those captures take ONE forward with no warning.
A module global is patched earlier, by the wrap-time module sweep (reversed at
``unwrap_torch()``), so it also takes one forward. Holders neither reaches (a
custom object) keep the disclosure plus the rescue forward.

Holder contract under the pure-observer capture (AGT-2360): the rebind
journal and the rescue never read state the forward left on the module. A
default capture restores every plain attribute afterwards, so forward-time
holder edits are undone and read from the trace's attribute-write record;
only ``CaptureOptions(carry_state=True)`` keeps them, and then the journal
still never clobbers a location the forward changed.
"""

from __future__ import annotations

import functools
import sys
import types
import warnings
from collections import namedtuple
from collections.abc import Callable, Iterator

import pytest
import torch
from _forward_stash import stashed
from torch import nn

import torchlens as tl
from torchlens._errors import CaptureGapWarning
from torchlens._utils._tensor_label import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions

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


class _Box:
    """A plain (non-module, non-container) object holding a callable."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        self.fn = fn


class _BoxHolder(nn.Module):
    """Stale function inside a custom object attribute."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.box = _Box(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the boxed function's output."""

        return self.box.fn(x)


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
    "default_argument": lambda env: _default_arg_module(env.gelu),
    "module_global": _module_global_module,
    "custom_object": lambda env: _BoxHolder(env.gelu),
}
# Holders neither the pre-capture rebind nor the wrap-time module sweep reaches:
# these keep the rescue path.
_UNREBINDABLE = frozenset({"custom_object"})


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_held_refs_capture.py, campaign mutant kills.
def test_capture_leaves_every_held_reference_identical(raw: types.SimpleNamespace) -> None:
    """The rebind is undone at cleanup: the same objects sit in the same slots."""

    act = _GELUActivationLike(raw.gelu)
    partial_act = _CallFn(functools.partial(raw.gelu, approximate="tanh"))
    closure_act = _CallFn(_closure(raw.gelu))
    table = _NestedContainer(raw.gelu)
    acts = _NamedTupleHolder(raw.gelu)
    default_arg = _default_arg_module(raw.gelu)
    before = (
        act.act,
        partial_act.fn,
        partial_act.fn.func,
        closure_act.fn.__closure__[0].cell_contents,
        table.table,
        table.table["acts"],
        table.table["acts"][0],
        acts.acts,
        type(default_arg).forward.__defaults__,
    )
    holders = (act, partial_act, closure_act, table, acts, default_arg)
    model = nn.Sequential(nn.Linear(4, 4), *[_Linear4(_Holder(h)) for h in holders])
    wrap_torch()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        tl.trace(model, torch.randn(2, 4))

    after = (
        act.act,
        partial_act.fn,
        partial_act.fn.func,
        closure_act.fn.__closure__[0].cell_contents,
        table.table,
        table.table["acts"],
        table.table["acts"][0],
        acts.acts,
        type(default_arg).forward.__defaults__,
    )
    assert all(new is old for new, old in zip(after, before, strict=True))
    assert act.act is raw.gelu and partial_act.fn.func is raw.gelu


class _Linear4(nn.Module):
    """Map a ``_Holder``'s 3 outputs back to 4 so holders chain in a Sequential."""

    def __init__(self, holder: nn.Module) -> None:
        super().__init__()
        self.holder = holder
        self.back = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return back(holder(x))."""

        return self.back(self.holder(x))


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_held_refs_capture.py, campaign mutant kills.
def test_aliased_holders_stay_aliased_during_the_capture(raw: types.SimpleNamespace) -> None:
    """One rebuilt replacement per original: ``a is b`` holds inside the forward."""

    class Aliased(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = self.b = functools.partial(raw.gelu)
            self.seen: list[bool] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen.append(self.a is self.b)
            return self.b(self.a(x))

    model = Aliased()
    held = model.a
    wrap_torch()
    trace = tl.trace(model, torch.randn(2, 4))

    assert stashed(trace, "seen") == [True]
    assert [op.func_name for op in trace.ops].count("gelu") == 2
    assert model.a is held and model.b is held


class _Editing(nn.Module):
    """Edits its holders during the forward: clears, deletes and rebinds."""

    def __init__(self, raw: types.SimpleNamespace) -> None:
        super().__init__()
        self.items = [raw.gelu]
        self.table = {"op": raw.tanh, "gone": raw.relu}
        self.act = raw.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.table["op"](self.items[0](x))
        self.items.clear()
        del self.table["gone"]
        self.act = torch.sigmoid
        return y


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_held_refs_capture.py, campaign mutant kills.
def test_forward_time_holder_edits_are_never_clobbered_under_carry_state(
    raw: types.SimpleNamespace,
) -> None:
    """With carry_state=True cleanup restores only locations still holding a wrapper."""

    model = _Editing(raw)
    wrap_torch()
    tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(carry_state=True))

    assert model.items == []
    assert set(model.table) == {"op"} and model.table["op"] is raw.tanh
    assert model.act is torch.sigmoid


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_outputs.py, campaign mutant kills.
def test_iql_output_activation_returned_by_the_root_is_rescued(
    raw: types.SimpleNamespace,
) -> None:
    """IQL shape: a stale tanh produces the ROOT module's output.

    Control row: the root's untagged output already fails output attribution,
    which triggers the rescue without the module-exit record.
    """

    class IqlLike(nn.Module):
        def __init__(self, out_act: Callable[..., torch.Tensor]) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)
            # A custom-object holder: the pre-capture rebind cannot reach it.
            self.output_activation = _Box(out_act)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.output_activation.fn(self.fc(x))

    model = IqlLike(raw.tanh)
    wrap_torch()
    with pytest.warns(CaptureGapWarning, match="output_attribution_failed"):
        trace = tl.trace(model, torch.randn(2, 4))

    assert "tanh" in [op.func_name for op in trace.ops]
    assert trace.provenance.verification_reason == "mode_rescue_rerun"
