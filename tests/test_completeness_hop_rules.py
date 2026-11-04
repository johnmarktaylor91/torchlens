"""The completeness witness accounts for higher-order-operator subgraph ops.

``torch.cond``, ``while_loop``, ``map`` and ``scan`` run user callables through a
``HigherOrderOperator``. TorchLens captures the ops those callables run, so the
completeness witness must census the same ops: a correct capture validates True,
and an op inside the subgraph that TorchLens does not capture (a raw
``torch._C._VariableFunctions`` call, the planted drop) validates False. Before the
witness carried a rule per HOP, ``tl.validate`` raised ``NotImplementedError``
("no rule registered for HigherOrderOperator cond") on every such capture.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl

_RAW_MUL = torch._C._VariableFunctions.mul
"""A dispatcher-level multiply TorchLens does not wrap: the planted dropped op."""


def _hop_function(module: str, name: str) -> Callable[..., object]:
    """Return a HOP entry point, skipping the test when this torch lacks it.

    Parameters
    ----------
    module:
        ``torch._higher_order_ops`` submodule name.
    name:
        Attribute holding the user-facing entry point.

    Returns
    -------
    Callable[..., object]
        The entry point.
    """

    try:
        mod = importlib.import_module(f"torch._higher_order_ops.{module}")
    except ImportError:
        pytest.skip(f"torch {torch.__version__} has no torch._higher_order_ops.{module}")
    fn = getattr(mod, name, None)
    if fn is None:
        pytest.skip(f"torch {torch.__version__} has no {module}.{name}")
    return fn


class _Forward(nn.Module):
    """Wrap a plain function as a module forward."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store the forward function.

        Parameters
        ----------
        fn:
            Tensor-to-tensor function the forward calls.
        """

        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stored function.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The function's output.
        """

        return self.fn(x)


class _CondLinear(nn.Module):
    """``torch.cond`` whose taken branch runs a parameterised submodule."""

    def __init__(self) -> None:
        """Build the branch submodule."""

        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the linear branch.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch output plus the input.
        """

        return torch.cond(x.sum() > -1e9, lambda t: self.lin(t), lambda t: t - 1, (x,)) + x


def _cond_true(raw: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward whose ``torch.cond`` takes the true branch.

    Parameters
    ----------
    raw:
        Plant the uncaptured multiply in the taken branch.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    mul = _RAW_MUL if raw else torch.mul
    return lambda x: torch.cond(x.sum() > -1e9, lambda t: mul(t, 2.0), lambda t: t - 1, (x,)) + x


def _cond_false(raw: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward whose ``torch.cond`` takes the false branch.

    Parameters
    ----------
    raw:
        Plant the uncaptured multiply in the taken (false) branch.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    mul = _RAW_MUL if raw else torch.mul
    return lambda x: torch.cond(x.sum() < -1e9, lambda t: t - 1, lambda t: mul(t, 2.0), (x,)) + x


def _while_loop(raw: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running a three-step ``while_loop``.

    Parameters
    ----------
    raw:
        Plant the uncaptured multiply in the loop body.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    while_loop = _hop_function("while_loop", "while_loop")
    mul = _RAW_MUL if raw else torch.mul

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Loop three times doubling the carry.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final carry plus the input.
        """

        start = torch.zeros((), dtype=torch.int64)
        out = while_loop(lambda i, t: i < 3, lambda i, t: (i + 1, mul(t, 2.0)), (start, x))
        return out[1] + x

    return forward


def _map(raw: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward mapping a function over rows with ``map``.

    Parameters
    ----------
    raw:
        Plant the uncaptured multiply in the mapped function.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    hop_map = _hop_function("map", "map")
    mul = _RAW_MUL if raw else torch.mul
    return lambda x: hop_map(lambda r: mul(r.sin(), 2.0), x) + x


def _scan(raw: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running a cumulative ``scan``.

    Parameters
    ----------
    raw:
        Plant the uncaptured multiply in the combine function.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    scan = _hop_function("scan", "scan")
    mul = _RAW_MUL if raw else torch.mul

    def forward(x: torch.Tensor) -> torch.Tensor:
        """Scan rows, carrying a running sum.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Stacked per-step outputs plus the final carry.
        """

        carry, ys = scan(lambda c, r: (c + r, mul(c, r).clone()), torch.zeros(3), x)
        return ys + carry

    return forward


def _associative_scan(raw: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running a generic-mode ``associative_scan`` on CPU.

    Parameters
    ----------
    raw:
        Plant the uncaptured multiply in the combine function.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    associative_scan = _hop_function("associative_scan", "associative_scan")
    mul = _RAW_MUL if raw else torch.mul
    return lambda x: (
        associative_scan(lambda a, b: mul(a + b, 1.0), x, 0, combine_mode="generic") + x
    )


_HOP_FORWARDS: dict[str, Callable[[bool], Callable[[torch.Tensor], torch.Tensor]]] = {
    "cond_true_branch": _cond_true,
    "cond_false_branch": _cond_false,
    "while_loop": _while_loop,
    "map": _map,
    "scan": _scan,
    "associative_scan": _associative_scan,
}


def _validate(model: nn.Module) -> bool:
    """Validate one forward of ``model`` on a fixed input, warnings silenced.

    Parameters
    ----------
    model:
        Model to validate.

    Returns
    -------
    bool
        The public validation verdict.
    """

    torch.manual_seed(0)
    x = torch.randn(3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.validate(model, x, scope="forward", validate_metadata=True)


def _require_plain_run(fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Skip when this torch cannot run the HOP eagerly at all.

    Parameters
    ----------
    fn:
        Forward function to try once without TorchLens.
    """

    try:
        fn(torch.randn(3, 3))
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"torch {torch.__version__} cannot run this HOP eagerly: {exc}")


@pytest.mark.parametrize("hop", sorted(_HOP_FORWARDS))
def test_hop_capture_validates_true(hop: str) -> None:
    """A correct capture through a HOP validates True instead of raising."""

    fn = _HOP_FORWARDS[hop](False)
    _require_plain_run(fn)
    assert _validate(_Forward(fn)) is True


@pytest.mark.parametrize("hop", sorted(_HOP_FORWARDS))
def test_hop_planted_dropped_op_validates_false(hop: str) -> None:
    """An uncaptured op inside the HOP subgraph that ran is caught by the witness."""

    fn = _HOP_FORWARDS[hop](True)
    _require_plain_run(fn)
    assert _validate(_Forward(fn)) is False


def test_cond_branch_with_parameters_validates_true() -> None:
    """A ``torch.cond`` branch that runs a parameterised submodule validates True."""

    assert _validate(_CondLinear()) is True


def test_cond_capture_holds_only_the_taken_branch() -> None:
    """The trace records the branch that ran and nothing from the untaken one."""

    x = torch.randn(3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        taken_true = tl.trace(_Forward(_cond_true(False)), x)
        taken_false = tl.trace(_Forward(_cond_false(False)), x)
    try:
        true_funcs = [op.func_name for op in taken_true.ops]
        false_funcs = [op.func_name for op in taken_false.ops]
    finally:
        taken_true.cleanup()
        taken_false.cleanup()
    assert "mul" in true_funcs and "__sub__" not in true_funcs
    assert "mul" in false_funcs and "__sub__" not in false_funcs
    assert true_funcs.count("mul") == 1 and false_funcs.count("mul") == 1
