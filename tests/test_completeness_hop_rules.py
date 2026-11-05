"""The completeness witness accounts for higher-order-operator subgraph ops.

``torch.cond``, ``while_loop``, ``map`` and ``scan`` run user callables through a
``HigherOrderOperator``. TorchLens captures the ops those callables run, so the
completeness witness must census the same ops: a correct capture validates True,
and an op inside the subgraph that TorchLens does not capture (the original
``torch.mul`` taken while the wrappers are removed, the planted drop) is named by
the witness census. Branch callables use Tensor operators: ``tl.validate`` computes
its ground truth with a plain run that sends eager HOPs through ``torch.compile``,
which cannot inline a TorchLens wrapper captured by closure. Before the witness
carried a rule per HOP, ``tl.validate`` raised ``NotImplementedError``
("no rule registered for HigherOrderOperator cond") on every such capture.
"""

from __future__ import annotations

import importlib
import importlib.util
import operator
import warnings
from collections.abc import Callable, Iterator
from typing import cast

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import rescue
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

_Mul = Callable[[torch.Tensor, object], torch.Tensor]

_CLEAN_MUL: _Mul = operator.mul
"""The Tensor ``*`` operator: captured by TorchLens and inlinable by ``torch.compile``."""


@pytest.fixture
def raw_mul() -> Iterator[_Mul]:
    """Yield the original ``torch.mul``, taken while TorchLens's wrappers are removed.

    The original is a builtin TorchLens never sees. It is held in an
    :class:`OpaqueCallable` because capture preparation rebinds pristine torch
    functions found in closures, so a bare original would be captured after all.
    Calling it inside a HOP subgraph plants an op the capture drops while the
    witness still observes it. The wrappers are re-installed with the session's
    diagnostic modes before the test and again at teardown.

    Yields
    ------
    _Mul
        The unwrapped ``torch.mul`` behind an opaque holder.
    """

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    original = torch.mul
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)
    yield OpaqueCallable(original)
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


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

    qualname = f"torch._higher_order_ops.{module}"
    # Presence is probed with ``find_spec`` (parent package first, so a torch
    # without the package answers None instead of raising), never by catching
    # ImportError: a HOP module that exists but fails to import must fail
    # loudly rather than skip.
    if (
        importlib.util.find_spec("torch._higher_order_ops") is None
        or importlib.util.find_spec(qualname) is None
    ):
        pytest.skip(f"torch {torch.__version__} has no {qualname}")
    mod = importlib.import_module(qualname)
    fn = getattr(mod, name, None)
    if fn is None:
        pytest.skip(f"torch {torch.__version__} has no {module}.{name}")
    return fn


def _torch_cond() -> Callable[..., torch.Tensor]:
    """Return the user-facing ``cond``, skipping when this torch has no cond operator.

    The witness keys its cond rule on the ``cond_op`` HigherOrderOperator (the
    product resolves it through ``get_higher_order_operator("cond", "cond_op")``).
    A torch whose ``torch._higher_order_ops.cond`` predates that operator holds the
    bare operator under ``cond`` and has no user-facing ``torch.cond``, so a cond
    case there is not applicable by capability, through the same probe as an
    absent ``while_loop``.

    Returns
    -------
    Callable[..., torch.Tensor]
        The user-facing ``cond`` entry point.
    """

    _hop_function("cond", "cond_op")
    return cast(Callable[..., torch.Tensor], _hop_function("cond", "cond"))


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
        self.cond = _torch_cond()

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

        return self.cond(x.sum() > -1e9, lambda t: self.lin(t), lambda t: t - 1, (x,)) + x


def _cond_true(mul: _Mul) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward whose ``torch.cond`` takes the true branch.

    Parameters
    ----------
    mul:
        Multiply the taken branch runs.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    cond = _torch_cond()
    return lambda x: cond(x.sum() > -1e9, lambda t: mul(t, 2.0), lambda t: t - 1, (x,)) + x


def _cond_false(mul: _Mul) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward whose ``torch.cond`` takes the false branch.

    Parameters
    ----------
    mul:
        Multiply the taken (false) branch runs.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    cond = _torch_cond()
    return lambda x: cond(x.sum() < -1e9, lambda t: t - 1, lambda t: mul(t, 2.0), (x,)) + x


def _while_loop(mul: _Mul) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running a three-step ``while_loop``.

    Parameters
    ----------
    mul:
        Multiply the loop body runs.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    while_loop = _hop_function("while_loop", "while_loop")

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


def _map(mul: _Mul) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward mapping a function over rows with ``map``.

    Parameters
    ----------
    mul:
        Multiply the mapped function runs.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    hop_map = _hop_function("map", "map")
    return lambda x: hop_map(lambda r: mul(r.sin(), 2.0), x) + x


def _scan(mul: _Mul) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running a cumulative ``scan``.

    Parameters
    ----------
    mul:
        Multiply the combine function runs.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    scan = _hop_function("scan", "scan")

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


def _associative_scan(mul: _Mul) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a forward running a generic-mode ``associative_scan`` on CPU.

    Parameters
    ----------
    mul:
        Multiply the combine function runs.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The forward function.
    """

    associative_scan = _hop_function("associative_scan", "associative_scan")
    return lambda x: (
        associative_scan(lambda a, b: mul(a + b, 2.0), x, 0, combine_mode="generic") + x
    )


_HOP_FORWARDS: dict[str, Callable[[_Mul], Callable[[torch.Tensor], torch.Tensor]]] = {
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


@pytest.mark.parametrize("hop", sorted(_HOP_FORWARDS))
def test_hop_capture_validates_true(hop: str) -> None:
    """A correct capture through a HOP validates True instead of raising."""

    fn = _HOP_FORWARDS[hop](_CLEAN_MUL)
    assert _validate(_Forward(fn)) is True


_HOP_DISPATCHED = sorted(set(_HOP_FORWARDS) - {"associative_scan"})
"""Forwards whose HOP reaches the dispatcher on CPU.

Generic-mode ``associative_scan`` on CPU runs its combine loop in Python without
dispatching ``associative_scan_op`` (its capture validates True with no witness
rule at all), so its planted drop says nothing about the HOP rules.
"""


@pytest.mark.parametrize("hop", _HOP_DISPATCHED)
def test_hop_planted_dropped_op_is_named_by_the_witness(
    hop: str, raw_mul: _Mul, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An op the capture dropped inside the HOP subgraph that ran is named by the census.

    The rescue re-run is disabled so the primary capture keeps the planted drop (it would
    otherwise recover the raw op through aten recording and repair the capture).
    """

    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    wrap_torch(completeness_witness=True)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_Forward(_HOP_FORWARDS[hop](raw_mul)), torch.randn(3, 3))
    try:
        operators = [row["operator"] for row in trace.completeness_diagnostics]
        verified = trace.provenance.verified
    finally:
        trace.cleanup()
    assert any(str(name).startswith("aten.mul.") for name in operators), operators
    assert verified is False


def test_cond_branch_with_parameters_validates_true() -> None:
    """A ``torch.cond`` branch that runs a parameterised submodule validates True."""

    assert _validate(_CondLinear()) is True


def test_cond_capture_holds_only_the_taken_branch() -> None:
    """The trace records the branch that ran and nothing from the untaken one."""

    x = torch.randn(3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        taken_true = tl.trace(_Forward(_cond_true(_CLEAN_MUL)), x)
        taken_false = tl.trace(_Forward(_cond_false(_CLEAN_MUL)), x)
    try:
        true_funcs = [op.func_name for op in taken_true.ops]
        false_funcs = [op.func_name for op in taken_false.ops]
    finally:
        taken_true.cleanup()
        taken_false.cleanup()
    assert "__mul__" in true_funcs and "__sub__" not in true_funcs
    assert "__mul__" in false_funcs and "__sub__" not in false_funcs
    assert true_funcs.count("__mul__") == 1 and false_funcs.count("__mul__") == 1
