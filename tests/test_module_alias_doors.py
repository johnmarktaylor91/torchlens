"""A shared module's alias spelling refuses at every door; its canonical name fires everywhere.

``self.enc = blk; self.dec = self.enc`` registers one module object under two
addresses. ``named_modules()`` reports it once, under the first name, and
TorchLens labels every call of it with that canonical name (``enc:1``,
``enc:2``) while recording both names in ``Module.all_addresses``. No hook can
tell which attribute a forward called the object through, so
``tl.module("dec")`` cannot mean "the dec call": honoring it edits the ``enc``
call too, a number the user did not ask for (AUD-CODE 3.7d).

So every door refuses an alias spelling (``tl.module("dec")``,
``tl.module("dec:2")``, ``tl.in_module("dec")``, ``tl.in_module("dec:2")``)
with ONE typed error: the same code and the same message, naming the
canonical name, which fires at every call site of the shared module. The
doors are ``tl.trace`` and ``tl.record`` (``intervene=``, ``save=`` and
``halt=``), post-hoc ``find_sites`` and ``fork().do``, and
``spec.bind(model)``. Capture and bind doors refuse before any forward runs.

The canonical spelling is held to an eager oracle at every door: a forward
hook on the shared module for ``tl.module`` and a hand-written forward for
``tl.in_module``. The action is an add, which is not idempotent, so an edit
applied at the wrong call, at too many calls, or not at all changes the
numbers.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import site
from torchlens.intervention.errors import MultiMatchWarning

_X = torch.tensor([[1.0, -2.0, 0.5]])
_DELTA = 1.0
_ALIAS_CODE = "bind_static_anchor_unresolved"


class _Block(nn.Module):
    """Two ops (relu, then a scale) so op and module scopes differ."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(x) * 2``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``relu(x) * 2``.
        """

        return torch.relu(x) * 2.0


class _DecAfter(nn.Module):
    """``dec`` is registered after ``enc``; the forward calls both names."""

    def __init__(self) -> None:
        """Register one block under two names."""

        super().__init__()
        self.enc = _Block()
        self.dec = self.enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block through ``enc``, then through ``dec``, then shift.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The shifted second block output.
        """

        return self.dec(self.enc(x) - 0.5) + 0.25


class _DecBefore(nn.Module):
    """``dec`` is registered first, so ``named_modules()`` reports ``dec``."""

    def __init__(self) -> None:
        """Register one block under two names."""

        super().__init__()
        self.dec = _Block()
        self.enc = self.dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block through ``enc``, then through ``dec``, then shift.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The shifted second block output.
        """

        return self.dec(self.enc(x) - 0.5) + 0.25


class _DecTail(nn.Module):
    """The second call's output (through ``dec``) IS the model output."""

    def __init__(self) -> None:
        """Register one block under two names."""

        super().__init__()
        self.enc = _Block()
        self.dec = self.enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block twice and return its second output directly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The second block output.
        """

        return self.dec(self.enc(x) - 0.5)


# model name -> (class, canonical address, alias address)
_MODELS: dict[str, tuple[type[nn.Module], str, str]] = {
    "dec_after": (_DecAfter, "enc", "dec"),
    "dec_before": (_DecBefore, "dec", "enc"),
    "dec_tail": (_DecTail, "enc", "dec"),
}

# selector name -> (selector factory over an address, pass suffix, block calls
# whose OUTPUT receives the add, block calls whose INTERIOR ops each receive it)
_SELECTORS: dict[str, tuple[Callable[[str], Any], str, frozenset[int], frozenset[int]]] = {
    "module": (tl.module, "", frozenset({1, 2}), frozenset()),
    "module_pass_1": (tl.module, ":1", frozenset({1}), frozenset()),
    "module_pass_2": (tl.module, ":2", frozenset({2}), frozenset()),
    "in_module": (tl.in_module, "", frozenset(), frozenset({1, 2})),
    "in_module_pass_2": (tl.in_module, ":2", frozenset(), frozenset({2})),
}

_ALIAS_SPELLINGS = ("module", "module_pass_2", "in_module", "in_module_pass_2")


def _build(model_name: str) -> nn.Module:
    """Return a fresh eval-mode instance of one model.

    Parameters
    ----------
    model_name:
        Key into ``_MODELS``.

    Returns
    -------
    nn.Module
        The model.
    """

    torch.manual_seed(0)
    return _MODELS[model_name][0]().eval()


def _selector(model_name: str, selector_name: str, *, alias: bool) -> Any:
    """Build one selector spelled with the canonical or the alias address.

    Parameters
    ----------
    model_name:
        Key into ``_MODELS``.
    selector_name:
        Key into ``_SELECTORS``.
    alias:
        Whether to spell the address with the alias name.

    Returns
    -------
    Any
        The selector.
    """

    _cls, canonical, alias_name = _MODELS[model_name]
    factory, suffix, _module_calls, _interior_calls = _SELECTORS[selector_name]
    return factory((alias_name if alias else canonical) + suffix)


def _hook_oracle(model: nn.Module, address: str, module_calls: frozenset[int]) -> torch.Tensor:
    """Run the model eagerly with a forward hook adding at the named block calls.

    Parameters
    ----------
    model:
        Model holding the shared module.
    address:
        Attribute name of the shared module (either name is the same object).
    module_calls:
        One-based calls of the shared module whose output receives the add.

    Returns
    -------
    torch.Tensor
        The expected model output.
    """

    calls = 0

    def _hook(module: nn.Module, args: Any, output: torch.Tensor) -> torch.Tensor | None:
        """Add at the selected calls of the shared module."""

        nonlocal calls
        calls += 1
        return output + _DELTA if calls in module_calls else None

    handle = getattr(model, address).register_forward_hook(_hook)
    try:
        with torch.no_grad():
            return model(_X)
    finally:
        handle.remove()


def _interior_oracle(model_name: str, interior_calls: frozenset[int]) -> torch.Tensor:
    """Compute the output by hand with the add after each op inside the named calls.

    Parameters
    ----------
    model_name:
        Key into ``_MODELS``.
    interior_calls:
        One-based block calls whose interior op outputs (relu, scale) each
        receive the add.

    Returns
    -------
    torch.Tensor
        The expected model output.
    """

    calls = 0

    def block(value: torch.Tensor) -> torch.Tensor:
        """One block call with the requested edits applied."""

        nonlocal calls
        calls += 1
        interior = calls in interior_calls
        hidden = torch.relu(value)
        if interior:
            hidden = hidden + _DELTA
        result = hidden * 2.0
        return result + _DELTA if interior else result

    output = block(block(_X) - 0.5)
    return output if model_name == "dec_tail" else output + 0.25


def _expected(model_name: str, selector_name: str) -> torch.Tensor:
    """Return the eager oracle for one model and canonical selector.

    Parameters
    ----------
    model_name:
        Key into ``_MODELS``.
    selector_name:
        Key into ``_SELECTORS``.

    Returns
    -------
    torch.Tensor
        The expected steered model output.
    """

    _factory, _suffix, module_calls, interior_calls = _SELECTORS[selector_name]
    if interior_calls:
        return _interior_oracle(model_name, interior_calls)
    return _hook_oracle(_build(model_name), _MODELS[model_name][1], module_calls)


def _edit_trace(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``tl.trace(intervene=)`` and read the model output."""

    trace = tl.trace(model, _X, intervene=tl.when(selector, tl.add(_DELTA)))
    return trace[trace.output_layers[0]].out


def _edit_record(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``tl.record`` with an opaque callable ``save=``."""

    output, _recording = tl.record(
        model,
        _X,
        save=lambda ctx: True,
        intervene=tl.when(selector, tl.add(_DELTA)),
        return_output=True,
    )
    return output


def _intervention_ready(model: nn.Module) -> Any:
    """Capture an intervention-ready trace of the model."""

    return tl.trace(model, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


def _edit_fork_do(model: nn.Module, selector: Any) -> torch.Tensor:
    """Apply the edit post hoc with ``fork().do`` on an intervention-ready trace."""

    fork = _intervention_ready(model).fork()
    with warnings.catch_warnings():
        # In-module matches inside one call compound on purpose (an op and its successor).
        warnings.simplefilter("ignore", MultiMatchWarning)
        fork.do(selector, tl.add(_DELTA))
    return fork[fork.output_layers[0]].out


def _edit_bind(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``spec.bind(model)(x)``."""

    with torch.no_grad():
        return tl.when(selector, tl.add(_DELTA)).bind(model)(_X)


_EDIT_DOORS: dict[str, Callable[[nn.Module, Any], torch.Tensor]] = {
    "trace": _edit_trace,
    "record": _edit_record,
    "post_hoc_fork_do": _edit_fork_do,
    "bind": _edit_bind,
}


def _find_sites(model: nn.Module, selector: Any) -> Any:
    """Resolve the selector post hoc with ``find_sites``."""

    return tl.trace(model, _X).find_sites(selector, max_fanout=100)


def _trace_save(model: nn.Module, selector: Any) -> Any:
    """Select saved activations with ``tl.trace(save=)``."""

    return tl.trace(model, _X, save=selector)


def _record_save(model: nn.Module, selector: Any) -> Any:
    """Select retained records with ``tl.record(save=)``."""

    return tl.record(model, _X, save=selector, return_output=True)


def _trace_halt(model: nn.Module, selector: Any) -> Any:
    """Halt the capture with ``tl.trace(halt=)``."""

    return tl.trace(model, _X, halt=selector)


def _record_halt(model: nn.Module, selector: Any) -> Any:
    """Halt the recording with ``tl.record(halt=)``."""

    return tl.record(model, _X, save=lambda ctx: True, halt=selector, return_output=True)


#: Every door that reads a module selector. The first five carry an edit; the
#: rest select what to keep or where to stop, and must agree with the edits.
_ALL_DOORS: dict[str, Callable[[nn.Module, Any], Any]] = {
    **_EDIT_DOORS,
    "post_hoc_find_sites": _find_sites,
    "trace_save": _trace_save,
    "record_save": _record_save,
    "trace_halt": _trace_halt,
    "record_halt": _record_halt,
}

#: Doors that must refuse before the model's forward runs at all.
_PRE_FORWARD_DOORS = frozenset(
    {"trace", "record", "bind", "trace_save", "record_save", "trace_halt", "record_halt"}
)


def _refusal(door: str, model_name: str, selector: Any) -> tuple[type[BaseException], str, str]:
    """Run one door on an alias spelling and return its refusal.

    Parameters
    ----------
    door:
        Key into ``_ALL_DOORS``.
    model_name:
        Key into ``_MODELS``.
    selector:
        Alias-spelled selector.

    Returns
    -------
    tuple[type[BaseException], str, str]
        The refusal's type, typed code and message.
    """

    model = _build(model_name)
    forwards = 0

    def _count(module: nn.Module, args: Any) -> None:
        """Count forwards of the model."""

        nonlocal forwards
        forwards += 1

    handle = model.register_forward_pre_hook(_count)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the door's type is compared below
                _ALL_DOORS[door](model, selector)
    finally:
        handle.remove()
    zero_match = [str(w.message) for w in caught if "matched zero sites" in str(w.message)]
    assert not zero_match, f"{door}: warned instead of refusing: {zero_match}"
    if door in _PRE_FORWARD_DOORS:
        assert forwards == 0, f"{door}: the model ran {forwards} forward(s) before the refusal"
    exc = excinfo.value
    fields = getattr(exc, "fields", {})
    return type(exc), str(fields.get("code")), str(exc)


@pytest.mark.parametrize("selector_name", _ALIAS_SPELLINGS)
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_alias_spelling_refuses_with_one_code_and_message_at_every_door(
    model_name: str, selector_name: str
) -> None:
    """trace, record, post hoc and bind refuse the alias with the same code and message."""

    _cls, canonical, alias_name = _MODELS[model_name]
    selector = _selector(model_name, selector_name, alias=True)
    spelled = alias_name + _SELECTORS[selector_name][1]
    refusals = {door: _refusal(door, model_name, selector) for door in _ALL_DOORS}
    bind_type, bind_code, bind_message = refusals["bind"]
    assert bind_code == _ALIAS_CODE
    assert f"{spelled!r} is an alias of {canonical!r}" in bind_message
    assert "EVERY call site" in bind_message
    for door, (exc_type, code, message) in refusals.items():
        assert (exc_type, code) == (bind_type, bind_code), f"{door}: {exc_type.__name__} {code}"
        assert message == bind_message, f"{door}: {message!r} != {bind_message!r}"


@pytest.mark.parametrize("door", ["trace", "post_hoc_find_sites", "bind", "trace_save"])
@pytest.mark.parametrize(
    "build",
    [
        lambda name: tl.func("relu") & tl.in_module(name),
        lambda name: tl.func("relu") | tl.in_module(f"{name}:2"),
        lambda name: ~tl.module(name),
        lambda name: site(module_path=name),
    ],
    ids=["and", "or_pass", "not", "site_module_path"],
)
def test_alias_inside_a_composite_refuses_the_same_way(
    door: str, build: Callable[[str], Any]
) -> None:
    """An alias anywhere in the selector refuses, whatever the other terms would match."""

    _exc_type, code, message = _refusal(door, "dec_after", build("dec"))
    assert code == _ALIAS_CODE
    assert "'dec' is an alias of 'enc'" in message or "'dec:2' is an alias of 'enc'" in message


@pytest.mark.parametrize("selector_name", sorted(_SELECTORS))
@pytest.mark.parametrize("model_name", sorted(_MODELS))
@pytest.mark.parametrize("door", sorted(_EDIT_DOORS))
def test_canonical_spelling_fires_at_every_call_site(
    door: str, model_name: str, selector_name: str
) -> None:
    """The canonical name edits every call the eager oracle edits, at every door."""

    expected = _expected(model_name, selector_name)
    selector = _selector(model_name, selector_name, alias=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = _EDIT_DOORS[door](_build(model_name), selector)
    zero_match = [str(w.message) for w in caught if "matched zero sites" in str(w.message)]
    assert not zero_match, zero_match
    assert torch.equal(value.detach(), expected), f"{value.tolist()} != {expected.tolist()}"


@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_the_alias_map_is_disclosed_at_trace_and_bind(model_name: str) -> None:
    """``Module.all_addresses`` and ``BindReport.module_aliases`` name both addresses."""

    _cls, canonical, alias_name = _MODELS[model_name]
    trace = tl.trace(_build(model_name), _X)
    assert trace.modules[alias_name] is trace.modules[canonical]
    assert set(trace.modules[canonical].all_addresses) == {canonical, alias_name}

    bound = tl.when(tl.module(canonical), tl.add(_DELTA)).bind(_build(model_name))
    with torch.no_grad():
        bound(_X)
    report = bound.last_report
    assert report.module_aliases == {canonical: (alias_name,)}
    assert [fire["target"] for fire in report.fires] == [f"{canonical}:1", f"{canonical}:2"]


def test_oracle_distinguishes_every_scope() -> None:
    """The oracles for whole-module, one-call and interior edits differ from each other."""

    for model_name, (_cls, canonical, _alias) in _MODELS.items():
        values = {
            name: tuple(_expected(model_name, name).flatten().tolist()) for name in _SELECTORS
        }
        unsteered = tuple(_hook_oracle(_build(model_name), canonical, frozenset()).tolist()[0])
        assert len({*values.values(), unsteered}) == len(_SELECTORS) + 1, model_name
