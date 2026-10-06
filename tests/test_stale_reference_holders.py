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
Holders it cannot rebind (a module global, a custom object) keep the
disclosure plus the rescue forward.
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
from torch import nn

import torchlens as tl
from torchlens._errors import TorchLensCaptureGapWarning
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


class _Counted(nn.Module):
    """Root wrapper counting how many times the capture ran the forward."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = [0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count, then delegate."""

        self.calls[0] += 1
        return self.inner(x)


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
# Holders the pre-capture rebind cannot reach: these keep the rescue path.
_UNREBINDABLE = frozenset({"module_global", "custom_object"})


@pytest.mark.parametrize("holder", sorted(set(_HOLDERS) - _UNREBINDABLE))
def test_rebindable_stale_reference_captures_in_one_forward(
    raw: types.SimpleNamespace, holder: str
) -> None:
    """Each reachable holder shape is rebound for the capture: one forward, no warning."""

    model = _Counted(_Holder(_HOLDERS[holder](raw)))
    wrap_torch()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace = tl.trace(model, torch.randn(2, 4))

    assert model.calls == [1]
    assert "gelu" in [op.func_name for op in trace.ops]
    assert "internalsource" not in [layer.layer_type for layer in trace.layer_list]
    assert trace.rescue_rerun is None
    assert trace.capture_verified is not False


@pytest.mark.parametrize("holder", sorted(_UNREBINDABLE))
def test_unrebindable_stale_reference_is_disclosed_and_rescued(
    raw: types.SimpleNamespace, holder: str
) -> None:
    """A holder the rebind cannot reach keeps the disclosure and the rescue forward."""

    model = _Counted(_Holder(_HOLDERS[holder](raw)))
    wrap_torch()
    with pytest.warns(UserWarning, match=r"adopted at module exit inner\.act"):
        trace = tl.trace(model, torch.randn(2, 4))

    assert model.calls == [2]
    assert "gelu" in [op.func_name for op in trace.ops]
    assert "internalsource" not in [layer.layer_type for layer in trace.layer_list]
    assert trace.rescue_rerun is not None and trace.rescue_rerun["recovered"] is True
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "mode_rescue_rerun"


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


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("act_name", "class_name"), [("gelu", "GELUActivation"), ("gelu_pytorch_tanh", "GELUTanh")]
)
def test_transformers_activation_built_before_capture_needs_no_rescue(
    raw: types.SimpleNamespace, act_name: str, class_name: str
) -> None:
    """The real transformers activations, built pre-wrap: one forward, no warning, untouched."""

    pytest.importorskip("transformers")
    from transformers import activations

    act = activations.get_activation(act_name)
    assert type(act).__name__ == class_name
    held_before = act.act
    model = _Counted(_Holder(act))
    wrap_torch()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace = tl.trace(model, torch.randn(2, 4))

    assert model.calls == [1]
    assert [op.func_name for op in trace.ops].count("gelu") == 1
    assert trace.rescue_rerun is None
    assert act.act is held_before


def test_held_reference_is_restored_when_the_forward_raises(
    raw: types.SimpleNamespace,
) -> None:
    """The undo runs on a failed capture too.

    Partial and closure holders are used because a failed capture also
    releases the model, and release normalizes direct attributes and builtin
    containers to the live wrappers (a separate, documented behavior).
    """

    gelu = raw.gelu

    def invoke(v: torch.Tensor) -> torch.Tensor:
        return gelu(v)

    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = functools.partial(raw.gelu)
            self.helper = invoke

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.helper(self.act(x))
            raise RuntimeError("boom")

    model = Boom()
    held = model.act
    wrap_torch()
    with pytest.raises(RuntimeError, match="boom"):
        tl.trace(model, torch.randn(2, 4))

    assert model.act is held
    assert held.func is raw.gelu
    assert invoke.__closure__ is not None
    assert invoke.__closure__[0].cell_contents is gelu


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

    assert model.seen == [True]
    assert [op.func_name for op in trace.ops].count("gelu") == 2
    assert model.a is held and model.b is held


def test_forward_time_holder_edits_are_never_clobbered(raw: types.SimpleNamespace) -> None:
    """Cleanup restores only locations that still hold what the rebind installed."""

    class Editing(nn.Module):
        def __init__(self) -> None:
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

    model = Editing()
    wrap_torch()
    tl.trace(model, torch.randn(2, 4))

    assert model.items == []
    assert set(model.table) == {"op"} and model.table["op"] is raw.tanh
    assert model.act is torch.sigmoid


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
    with pytest.warns(TorchLensCaptureGapWarning, match="output_attribution_failed"):
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


class _ReturnsOwnTensor(nn.Module):
    """Return a tensor the module owns (a learned query or a registered buffer)."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind == "parameter":
            self.w = nn.Parameter(torch.randn(2, 3))
        else:
            self.register_buffer("w", torch.randn(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the owned tensor itself, untouched by any op."""

        return self.w


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_module_returning_its_own_parameter_or_buffer_is_not_an_escape(kind: str) -> None:
    """Negative: a returned Parameter or buffer is a known source, never a stale-ref gap.

    No provenance warning (the repo's filter would make it an error), no rescue
    re-run, and the capture is never marked unverified (a clean capture leaves
    ``capture_verified`` unset; only a rescue or gap sets it ``False``).
    """

    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 3)
            self.own = _ReturnsOwnTensor(kind)
            self.fc2 = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.fc1(x) + self.own(x))

    wrap_torch()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace = tl.trace(Outer(), torch.randn(2, 4))

    assert trace.rescue_rerun is None
    assert trace.capture_verified is not False
    assert trace.capture_verification_reason is None


def test_opaque_module_return_is_disclosed_unrecovered_and_still_validates() -> None:
    """Negative: a genuinely opaque producer (direct aten call) is not a stale ref.

    The module-exit record discloses it (provenance warning; the rescue finds
    nothing to recover and settles ``escape_rescue_unrecovered``), and the
    validation contract for an opaque single-dispatch module output is
    unchanged: the boundary credits the dispatch that built it.
    """

    class Opaque(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.tanh.default(x)

    wrap_torch()
    model = _Holder(Opaque())
    x = torch.randn(2, 4)
    with pytest.warns(UserWarning, match=r"adopted at module exit act"):
        trace = tl.trace(model, x)

    assert "tanh" not in [op.func_name for op in trace.ops]
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "escape_rescue_unrecovered"
    with pytest.warns(UserWarning, match=_PROVENANCE):
        assert tl.validate(model, x, scope="forward")


def test_model_preparation_allocates_no_func_call_ids(
    raw: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebinding held refs during model preparation never ticks the call-id counter.

    The model holds a pristine builtin (``F.gelu`` in a partial) and the
    pristine Python ``F.relu`` that ``nn.TransformerEncoderLayer`` binds as
    its default activation; only the forward may allocate ``func_call_id``s.
    """

    from torchlens import _state
    from torchlens.backends.torch.backend import TorchBackend

    allocated: list[int] = []
    real_next = _state.next_func_call_id

    def counting_next() -> int:
        allocated.append(1)
        return real_next()

    during_prep: list[int] = []
    real_prepare = TorchBackend.prepare_model_session

    def counting_prepare(self: TorchBackend, session: object, model: object) -> object:
        before = len(allocated)
        try:
            return real_prepare(self, session, model)
        finally:
            during_prep.append(len(allocated) - before)

    monkeypatch.setattr(_state, "next_func_call_id", counting_next)
    monkeypatch.setattr(TorchBackend, "prepare_model_session", counting_prepare)

    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.act = functools.partial(raw.gelu, approximate="tanh")
            self.encoder = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.encoder(x))

    model = Mixed().eval()
    wrap_torch()
    tl.trace(model, torch.randn(2, 3, 4))

    assert during_prep == [0]
    assert allocated
