"""One intervention spec gives one answer through every execution door.

Regression pins for two door splits:

* A pass-label module selector (``tl.module("block:2")``, documented as
  "module address or pass label") on a module called twice matched zero sites
  in ``tl.trace`` and ``tl.record`` (warning, unsteered) while
  ``spec.bind(model)`` applied it at the named call.
* ``tl.record(..., intervene=tl.when(tl.module("block"), ...))`` where the
  module's output IS the model output raised ``OutputAttributionError`` while
  ``tl.trace`` and ``bind`` returned the steered value.

Every case runs one spec through ``tl.trace``, ``tl.record`` (with an opaque
callable ``save=`` and with a module-selector ``save=``) and
``spec.bind(model)(x)`` and holds each to the same eager oracle computed by
hand. The action is an add, which is not idempotent, so an edit applied at the
wrong call, at too many calls, or not at all changes the numbers.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

_X = torch.tensor([[1.0, -2.0, 0.5]])
_DELTA = 1.0
_ZERO_MATCH_TEXT = "matched zero sites"


class _Block(nn.Module):
    """Two ops (relu, then a scale) so op and module scopes differ."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(x) * 2``."""

        return torch.relu(x) * 2.0


class _TwiceInner(nn.Module):
    """``block`` is called twice; the model output is a later op."""

    def __init__(self) -> None:
        """Build the shared block."""

        super().__init__()
        self.block = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block twice, then shift."""

        return self.block(self.block(x) - 0.5) + 0.25


class _TwiceTail(nn.Module):
    """``block`` is called twice; its second output IS the model output."""

    def __init__(self) -> None:
        """Build the shared block."""

        super().__init__()
        self.block = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block twice and return its second output directly."""

        return self.block(self.block(x) - 0.5)


_MODELS: dict[str, type[nn.Module]] = {"inner": _TwiceInner, "tail": _TwiceTail}

# selector name -> (selector factory, block calls whose OUTPUT receives the add,
# add after every relu op, block calls whose INTERIOR ops each receive the add)
_SELECTORS: dict[str, tuple[Callable[[], Any], frozenset[int], bool, frozenset[int]]] = {
    "module_bare": (lambda: tl.module("block"), frozenset({1, 2}), False, frozenset()),
    "module_pass_1": (lambda: tl.module("block:1"), frozenset({1}), False, frozenset()),
    "module_pass_2": (lambda: tl.module("block:2"), frozenset({2}), False, frozenset()),
    "func": (lambda: tl.func("relu"), frozenset(), True, frozenset()),
    "in_module_bare": (lambda: tl.in_module("block"), frozenset(), False, frozenset({1, 2})),
    "in_module_pass_2": (lambda: tl.in_module("block:2"), frozenset(), False, frozenset({2})),
}


def _eager_oracle(
    model_name: str,
    module_calls: frozenset[int],
    func_relu: bool,
    interior_calls: frozenset[int] = frozenset(),
) -> torch.Tensor:
    """Compute the steered output by hand, without TorchLens.

    Parameters
    ----------
    model_name:
        Key into ``_MODELS``.
    module_calls:
        One-based block calls whose OUTPUT receives the add.
    func_relu:
        Whether every relu op output receives the add.
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
        if func_relu or interior:
            hidden = hidden + _DELTA
        result = hidden * 2.0
        if interior:
            result = result + _DELTA
        if calls in module_calls:
            result = result + _DELTA
        return result

    output = block(block(_X) - 0.5)
    if model_name == "inner":
        output = output + 0.25
    return output


def _spec(selector_name: str) -> Any:
    """Return the one-clause add spec for a selector."""

    factory = _SELECTORS[selector_name][0]
    return tl.when(factory(), tl.add(_DELTA))


def _door_trace(model: nn.Module, spec: Any) -> torch.Tensor:
    """Run the spec through ``tl.trace`` and read the model output."""

    trace = tl.trace(model, _X, intervene=spec)
    return trace[trace.output_layers[0]].out


def _door_record_callable(model: nn.Module, spec: Any) -> torch.Tensor:
    """Run the spec through ``tl.record`` with an opaque callable ``save=``."""

    output, _recording = tl.record(
        model, _X, save=lambda ctx: True, intervene=spec, return_output=True
    )
    return output


def _door_record_selector(model: nn.Module, spec: Any) -> torch.Tensor:
    """Run the spec through ``tl.record`` with a module-selector ``save=``."""

    output, _recording = tl.record(
        model, _X, save=tl.module("block"), intervene=spec, return_output=True
    )
    return output


def _door_bind(model: nn.Module, spec: Any) -> torch.Tensor:
    """Run the spec through ``spec.bind(model)(x)``."""

    with torch.no_grad():
        return spec.bind(model)(_X)


_DOORS: dict[str, Callable[[nn.Module, Any], torch.Tensor]] = {
    "trace": _door_trace,
    "record_callable_save": _door_record_callable,
    "record_selector_save": _door_record_selector,
    "bind": _door_bind,
}


def _outcome(door: str, model_name: str, selector_name: str) -> tuple[str, Any]:
    """Run one door and return ``("ok", tensor)`` or ``("refused", (type, code))``.

    A zero-match warning counts as a failure of its own: the selector is valid
    for these models, so no door may silently decline it.
    """

    torch.manual_seed(0)
    model = _MODELS[model_name]().eval()
    spec = _spec(selector_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = _DOORS[door](model, spec)
        except Exception as exc:
            return "refused", (type(exc).__name__, getattr(exc, "code", None), str(exc)[:200])
    zero_match = [str(w.message) for w in caught if _ZERO_MATCH_TEXT in str(w.message)]
    assert not zero_match, f"{door} declined a valid selector: {zero_match}"
    return "ok", value.detach().clone()


@pytest.mark.parametrize("selector_name", sorted(_SELECTORS))
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_every_door_matches_the_eager_oracle(model_name: str, selector_name: str) -> None:
    """trace, record and bind return the hand-computed steered output."""

    _factory, module_calls, func_relu, interior_calls = _SELECTORS[selector_name]
    expected = _eager_oracle(model_name, module_calls, func_relu, interior_calls)
    outcomes = {door: _outcome(door, model_name, selector_name) for door in _DOORS}
    refused = {door: detail for door, (kind, detail) in outcomes.items() if kind == "refused"}
    assert not refused, f"doors refused a valid selector: {refused}"
    for door, (_kind, value) in outcomes.items():
        assert torch.equal(value, expected), f"{door}: {value.tolist()} != {expected.tolist()}"


def test_oracle_distinguishes_every_scope() -> None:
    """Each selector's oracle differs from the others and from no edit at all."""

    for model_name in _MODELS:
        values = [
            tuple(_eager_oracle(model_name, calls, func, interior).flatten().tolist())
            for _factory, calls, func, interior in _SELECTORS.values()
        ]
        values.append(tuple(_eager_oracle(model_name, frozenset(), False).flatten().tolist()))
        assert len(set(values)) == len(values), model_name


@pytest.mark.parametrize("selector_name", ["module_bare", "module_pass_2"])
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_record_logs_the_boundary_edit_as_its_own_op(model_name: str, selector_name: str) -> None:
    """A boundary edit is a labeled replacement op in record, as in trace.

    The edited value must have a producer whose parent is the op it replaced,
    so record never attributes the steered value to the unsteered op.
    """

    torch.manual_seed(0)
    model = _MODELS[model_name]().eval()
    output, recording = tl.record(
        model, _X, save=lambda ctx: True, intervene=_spec(selector_name), return_output=True
    )
    replacements = [
        record for record in recording.records if record.ctx.layer_type == "interventionreplacement"
    ]
    module_calls = _SELECTORS[selector_name][1]
    assert len(replacements) == len(module_calls)
    for record in replacements:
        assert record.ctx.parent_labels, "replacement op lost its edge to the replaced op"
        assert record.ram_payload is not None
    if model_name == "tail":
        assert torch.equal(replacements[-1].ram_payload, output.detach())
