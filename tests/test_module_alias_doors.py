"""A second registered name for a module selects that module at every door.

``self.block = blk; self.alias = blk`` registers one module object under two
addresses. ``named_modules()`` reports it once, and TorchLens records both
spellings in ``Module.all_addresses`` (``trace.modules["alias"]`` is
``trace.modules["block"]``). The selector spellings must agree: ``tl.trace``
and ``tl.record`` used to warn "matched zero sites" for whichever spelling the
capture did not label the calls with (leaving the output unsteered), while
``spec.bind(model)`` refused the spelling ``named_modules()`` did not report.

An address names a module object, and a pass label names one call of that
object, whichever attribute the forward called it through. So
``tl.module("alias")`` is ``tl.module("block")``, ``tl.module("alias:2")`` is
the object's second call, and ``tl.in_module`` follows the same rule. Every
door (``tl.trace``, ``tl.record``, post-hoc ``fork().do`` and
``spec.bind(model)(x)``) is held to an eager oracle: a forward hook on the
shared module for ``tl.module`` and a hand-written forward for
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
from torchlens.intervention.errors import MultiMatchWarning

_X = torch.tensor([[1.0, -2.0, 0.5]])
_DELTA = 1.0
_ZERO_MATCH_TEXT = "matched zero sites"


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


class _AliasAfter(nn.Module):
    """``alias`` is registered after ``block``; the forward calls both names."""

    def __init__(self) -> None:
        """Register one block under two names."""

        super().__init__()
        self.block = _Block()
        self.alias = self.block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block through ``block``, then through ``alias``, then shift.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The shifted second block output.
        """

        return self.alias(self.block(x) - 0.5) + 0.25


class _AliasBefore(nn.Module):
    """``alias`` is registered first, so ``named_modules()`` reports ``alias``."""

    def __init__(self) -> None:
        """Register one block under two names."""

        super().__init__()
        self.alias = _Block()
        self.block = self.alias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block through ``block``, then through ``alias``, then shift.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The shifted second block output.
        """

        return self.alias(self.block(x) - 0.5) + 0.25


class _AliasTail(nn.Module):
    """The second call's output (through ``alias``) IS the model output."""

    def __init__(self) -> None:
        """Register one block under two names."""

        super().__init__()
        self.block = _Block()
        self.alias = self.block

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

        return self.alias(self.block(x) - 0.5)


_MODELS: dict[str, type[nn.Module]] = {
    "alias_after": _AliasAfter,
    "alias_before": _AliasBefore,
    "alias_tail": _AliasTail,
}

# selector name -> (selector factory, block calls whose OUTPUT receives the add,
# block calls whose INTERIOR ops each receive the add)
_SELECTORS: dict[str, tuple[Callable[[], Any], frozenset[int], frozenset[int]]] = {
    "module_block": (lambda: tl.module("block"), frozenset({1, 2}), frozenset()),
    "module_alias": (lambda: tl.module("alias"), frozenset({1, 2}), frozenset()),
    "module_block_pass_2": (lambda: tl.module("block:2"), frozenset({2}), frozenset()),
    "module_alias_pass_1": (lambda: tl.module("alias:1"), frozenset({1}), frozenset()),
    "module_alias_pass_2": (lambda: tl.module("alias:2"), frozenset({2}), frozenset()),
    "in_module_block": (lambda: tl.in_module("block"), frozenset(), frozenset({1, 2})),
    "in_module_alias": (lambda: tl.in_module("alias"), frozenset(), frozenset({1, 2})),
    "in_module_alias_pass_2": (lambda: tl.in_module("alias:2"), frozenset(), frozenset({2})),
}


def _hook_oracle(model: nn.Module, module_calls: frozenset[int]) -> torch.Tensor:
    """Run the model eagerly with a forward hook adding at the named block calls.

    Parameters
    ----------
    model:
        Model whose ``block`` attribute is the shared module.
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

    handle = model.block.register_forward_hook(_hook)
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
    return output if model_name == "alias_tail" else output + 0.25


def _expected(model_name: str, selector_name: str) -> torch.Tensor:
    """Return the eager oracle for one model and selector.

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

    _factory, module_calls, interior_calls = _SELECTORS[selector_name]
    if interior_calls:
        return _interior_oracle(model_name, interior_calls)
    return _hook_oracle(_MODELS[model_name]().eval(), module_calls)


def _door_trace(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``tl.trace(intervene=)`` and read the model output."""

    trace = tl.trace(model, _X, intervene=tl.when(selector, tl.add(_DELTA)))
    return trace[trace.output_layers[0]].out


def _door_record_callable(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``tl.record`` with an opaque callable ``save=``."""

    output, _recording = tl.record(
        model,
        _X,
        save=lambda ctx: True,
        intervene=tl.when(selector, tl.add(_DELTA)),
        return_output=True,
    )
    return output


def _door_record_alias_save(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``tl.record`` with an alias-spelled ``save=`` selector."""

    output, _recording = tl.record(
        model,
        _X,
        save=tl.in_module("alias"),
        intervene=tl.when(selector, tl.add(_DELTA)),
        return_output=True,
    )
    return output


def _door_post_hoc(model: nn.Module, selector: Any) -> torch.Tensor:
    """Apply the edit post hoc with ``fork().do`` on an intervention-ready trace."""

    trace = tl.trace(model, _X, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = trace.fork()
    with warnings.catch_warnings():
        # In-module matches inside one call compound on purpose (an op and its successor).
        warnings.simplefilter("ignore", MultiMatchWarning)
        fork.do(selector, tl.add(_DELTA))
    return fork[fork.output_layers[0]].out


def _door_bind(model: nn.Module, selector: Any) -> torch.Tensor:
    """Run the edit through ``spec.bind(model)(x)``."""

    with torch.no_grad():
        return tl.when(selector, tl.add(_DELTA)).bind(model)(_X)


_DOORS: dict[str, Callable[[nn.Module, Any], torch.Tensor]] = {
    "trace": _door_trace,
    "record_callable_save": _door_record_callable,
    "record_alias_save": _door_record_alias_save,
    "post_hoc_fork_do": _door_post_hoc,
    "bind": _door_bind,
}


def _outcome(door: str, model_name: str, selector_name: str) -> tuple[str, Any]:
    """Run one door and return ``("ok", tensor)`` or ``("refused", detail)``.

    Parameters
    ----------
    door:
        Key into ``_DOORS``.
    model_name:
        Key into ``_MODELS``.
    selector_name:
        Key into ``_SELECTORS``.

    Returns
    -------
    tuple[str, Any]
        The outcome kind and the output tensor or the refusal detail. A
        zero-match warning is reported as a refusal: it leaves the output
        unsteered while the user asked for steering.
    """

    torch.manual_seed(0)
    model = _MODELS[model_name]().eval()
    selector = _SELECTORS[selector_name][0]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = _DOORS[door](model, selector)
        except Exception as exc:  # noqa: BLE001 - a refusal of any type is a door outcome the table compares
            return "refused", (type(exc).__name__, getattr(exc, "code", None), str(exc)[:240])
    zero_match = [str(w.message)[:240] for w in caught if _ZERO_MATCH_TEXT in str(w.message)]
    if zero_match:
        return "refused", ("zero-match warning", None, zero_match)
    return "ok", value.detach().clone()


@pytest.mark.parametrize("selector_name", sorted(_SELECTORS))
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_alias_spelling_steers_the_same_module_at_every_door(
    model_name: str, selector_name: str
) -> None:
    """trace, record, post-hoc and bind all match the eager oracle for either name."""

    expected = _expected(model_name, selector_name)
    outcomes = {door: _outcome(door, model_name, selector_name) for door in _DOORS}
    refused = {door: detail for door, (kind, detail) in outcomes.items() if kind == "refused"}
    assert not refused, f"doors declined a registered module address: {refused}"
    for door, (_kind, value) in outcomes.items():
        assert torch.equal(value, expected), f"{door}: {value.tolist()} != {expected.tolist()}"


def test_oracle_distinguishes_every_scope() -> None:
    """The oracles for whole-module, one-call and interior edits differ from each other."""

    for model_name in _MODELS:
        values = {
            name: tuple(_expected(model_name, name).flatten().tolist()) for name in _SELECTORS
        }
        unsteered = tuple(_hook_oracle(_MODELS[model_name]().eval(), frozenset()).tolist()[0])
        distinct = {
            values["module_block"],
            values["module_alias_pass_1"],
            values["module_alias_pass_2"],
            values["in_module_block"],
            values["in_module_alias_pass_2"],
            unsteered,
        }
        assert len(distinct) == 6, model_name


@pytest.mark.parametrize("model_name", sorted(_MODELS))
@pytest.mark.parametrize(
    ("alias_selector", "primary_selector"),
    [
        (lambda: tl.module("alias"), lambda: tl.module("block")),
        (lambda: tl.module("alias:2"), lambda: tl.module("block:2")),
        (lambda: tl.in_module("alias"), lambda: tl.in_module("block")),
        (lambda: tl.in_module("alias:1"), lambda: tl.in_module("block:1")),
    ],
    ids=["module", "module_pass", "in_module", "in_module_pass"],
)
def test_find_sites_resolves_either_name_to_the_same_sites(
    model_name: str,
    alias_selector: Callable[[], Any],
    primary_selector: Callable[[], Any],
) -> None:
    """Post-hoc ``find_sites`` returns the same non-empty sites for both names."""

    torch.manual_seed(0)
    trace = tl.trace(_MODELS[model_name]().eval(), _X)
    assert trace.modules["alias"] is trace.modules["block"]

    def _labels(selector: Any) -> list[str]:
        """Return the matched op labels in order."""

        return [site.layer_label for site in trace.find_sites(selector, max_fanout=100)]

    alias_labels = _labels(alias_selector())
    assert alias_labels, "the alias spelling matched no sites"
    assert alias_labels == _labels(primary_selector())


@pytest.mark.parametrize("model_name", sorted(_MODELS))
@pytest.mark.parametrize(
    ("alias_selector", "primary_selector"),
    [
        (lambda: tl.module("alias"), lambda: tl.module("block")),
        (lambda: tl.in_module("alias:2"), lambda: tl.in_module("block:2")),
    ],
    ids=["module", "in_module_pass"],
)
def test_trace_save_selector_resolves_either_name_to_the_same_ops(
    model_name: str,
    alias_selector: Callable[[], Any],
    primary_selector: Callable[[], Any],
) -> None:
    """``tl.trace(save=...)`` keeps the same activations for both names, without warning."""

    def _saved(selector: Any) -> list[str]:
        """Return the saved op labels of one selective capture."""

        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trace = tl.trace(_MODELS[model_name]().eval(), _X, save=selector)
        zero_match = [str(w.message) for w in caught if _ZERO_MATCH_TEXT in str(w.message)]
        assert not zero_match, zero_match
        return [str(op.label) for op in trace.saved_ops if not op.is_input and not op.is_output]

    alias_saved = _saved(alias_selector())
    assert alias_saved, "the alias spelling saved nothing"
    assert alias_saved == _saved(primary_selector())


@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_record_save_selector_resolves_either_name_to_the_same_records(model_name: str) -> None:
    """``tl.record(save=tl.in_module(...))`` retains the same records for both names."""

    def _retained(selector: Any) -> list[str]:
        """Return the retained record labels of one recording."""

        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _output, recording = tl.record(
                _MODELS[model_name]().eval(), _X, save=selector, return_output=True
            )
        zero_match = [str(w.message) for w in caught if _ZERO_MATCH_TEXT in str(w.message)]
        assert not zero_match, zero_match
        return [str(record.ctx.label) for record in recording.records]

    alias_records = _retained(tl.in_module("alias"))
    assert alias_records, "the alias spelling retained nothing"
    assert alias_records == _retained(tl.in_module("block"))
