"""Tests for Op.var_names source assignment extraction."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import torchlens as tl
from torchlens.data_classes.trace import Trace


def _first_var_names(trace: Trace, func_name: str) -> list[str]:
    """Return var names for the first op with ``func_name``.

    Parameters
    ----------
    trace:
        Captured trace to search.
    func_name:
        Function name to find.

    Returns
    -------
    list[str]
        The matched op's ``var_names``.
    """

    return next(op.var_names for op in trace.layer_list if op.func_name == func_name)


def _all_var_names(trace: Trace, func_name: str) -> list[list[str]]:
    """Return var names for all ops with ``func_name``.

    Parameters
    ----------
    trace:
        Captured trace to search.
    func_name:
        Function name to find.

    Returns
    -------
    list[list[str]]
        ``var_names`` for every matched op, in trace order.
    """

    return [op.var_names for op in trace.layer_list if op.func_name == func_name]


def _trace(model: torch.nn.Module) -> Trace:
    """Capture a small model with source context enabled.

    Parameters
    ----------
    model:
        Module to trace.

    Returns
    -------
    Trace
        Captured trace.
    """

    return tl.trace(model, torch.randn(4), save_code_context=True)


class _SingleAssign(torch.nn.Module):
    """Model with a single named assignment."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a named relu assignment."""

        t = torch.relu(x)
        return t


class _TupleAssign(torch.nn.Module):
    """Model with tuple-unpack assignment from one call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a tuple-unpack chunk assignment."""

        a, b = torch.chunk(x, 2)
        return a + b


class _MultiAssign(torch.nn.Module):
    """Model with chained assignment."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a chained sigmoid assignment."""

        p = q = torch.sigmoid(x)
        return p + q


class _ReturnInline(torch.nn.Module):
    """Model with an inline return call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an inline relu call."""

        return torch.relu(x)


class _NestedCalls(torch.nn.Module):
    """Model with nested call expressions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run nested sigmoid and relu calls."""

        z = torch.relu(torch.sigmoid(x))
        return z


class _TwoCallsOneLine(torch.nn.Module):
    """Model with two calls on one source line."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two same-line assignments."""

        a = torch.relu(x)
        b = torch.sigmoid(x)  # noqa: E702
        return a + b


class _UnsupportedTargets(torch.nn.Module):
    """Model with augmented, attribute, and subscript target forms."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run unsupported assignment target forms."""

        y = x.clone()
        y += torch.relu(x)
        self.h = torch.sigmoid(x)
        values = [x]
        values[0] = torch.tanh(x)
        return y + self.h + values[0]


class _ComprehensionCall(torch.nn.Module):
    """Model with a call inside a comprehension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a comprehension containing a relu call."""

        values = [torch.relu(x) for _ in range(1)]
        return values[0]


def _identity_decorator(
    _value: torch.Tensor,
) -> Callable[[Callable[[], torch.Tensor]], Callable[[], torch.Tensor]]:
    """Return a decorator that leaves the function unchanged.

    Parameters
    ----------
    _value:
        Tensor argument used only to create a decorator-expression call site.

    Returns
    -------
    Callable[[Callable[[], torch.Tensor]], Callable[[], torch.Tensor]]
        Identity decorator.
    """

    def decorate(function: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
        """Return ``function`` unchanged."""

        return function

    return decorate


class _DecoratorCall(torch.nn.Module):
    """Model with a tensor op inside a runtime decorator expression."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define and call a decorated inner function."""

        @_identity_decorator(torch.relu(x))
        def inner() -> torch.Tensor:
            """Return the input tensor."""

            return x

        return inner()


class _WalrusCall(torch.nn.Module):
    """Model with a walrus assignment expression."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a walrus-bound relu call."""

        return (w := torch.relu(x)) + (w * 0)


class _TernaryBoolChain(torch.nn.Module):
    """Model with ternary and boolean-chain inline calls."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inline branch calls that should not be assigned names."""

        a = torch.relu(x) if bool(torch.tensor(True)) else torch.sigmoid(x)
        b = bool(torch.tensor(False)) or torch.tanh(x)
        return a + b


def test_var_names_single_tuple_and_chained_assignments() -> None:
    """Named, tuple, and chained assignments populate ``var_names``."""

    assert _first_var_names(_trace(_SingleAssign()), "relu") == ["t"]
    assert _all_var_names(_trace(_TupleAssign()), "chunk") == [["a", "b"], ["a", "b"]]
    assert _first_var_names(_trace(_MultiAssign()), "sigmoid") == ["p", "q"]


def test_var_names_inline_nested_and_same_line_calls() -> None:
    """Inline calls fail closed while direct same-line assignments resolve by column."""

    nested = _trace(_NestedCalls())
    same_line = _trace(_TwoCallsOneLine())

    assert _first_var_names(_trace(_ReturnInline()), "relu") == []
    assert _first_var_names(nested, "relu") == ["z"]
    assert _first_var_names(nested, "sigmoid") == []
    assert _first_var_names(same_line, "relu") == ["a"]
    assert _first_var_names(same_line, "sigmoid") == ["b"]


def test_var_names_adversarial_source_forms() -> None:
    """Unsupported forms follow the conservative extraction policy."""

    unsupported = _trace(_UnsupportedTargets())
    ternary_bool = _trace(_TernaryBoolChain())

    assert _first_var_names(unsupported, "relu") == []
    assert _first_var_names(unsupported, "sigmoid") == []
    assert _first_var_names(unsupported, "tanh") == []
    assert _first_var_names(_trace(_ComprehensionCall()), "relu") == []
    assert _first_var_names(_trace(_DecoratorCall()), "relu") == []
    assert _first_var_names(_trace(_WalrusCall()), "relu") == ["w"]
    assert _first_var_names(ternary_bool, "relu") == []
    assert _first_var_names(ternary_bool, "tanh") == []


def test_var_names_save_code_context_false() -> None:
    """Disabling source context leaves ``var_names`` empty."""

    trace = tl.trace(_SingleAssign(), torch.randn(4), save_code_context=False)
    assert _first_var_names(trace, "relu") == []


def test_var_names_dynamic_forward_source_unavailable() -> None:
    """Dynamically compiled source leaves ``var_names`` empty."""

    namespace: dict[str, Any] = {"torch": torch}
    exec(
        "def forward(self, x):\n    t = torch.relu(x)\n    return t\n",
        namespace,
    )

    class DynamicForward(torch.nn.Module):
        """Module whose forward method comes from ``exec``."""

    DynamicForward.forward = namespace["forward"]

    assert _first_var_names(_trace(DynamicForward()), "relu") == []


def test_var_names_tlspec_roundtrip(tmp_path: Path) -> None:
    """Portable save/load preserves ``var_names``."""

    trace = _trace(_SingleAssign())
    path = tmp_path / "var_names.tlspec"

    trace.save(path, level="portable")
    loaded = tl.load(path)

    assert _first_var_names(trace, "relu") == ["t"]
    assert _first_var_names(loaded, "relu") == ["t"]
