"""Capture of ``cond`` whose branch calls a submodule (closure parameters).

torch's ``cond`` entry compiles the call with Dynamo, which inlines every
submodule a branch calls and lifts its parameters into extra operands. The
compiled branch graph never re-enters TorchLens's module or function wrappers,
so its real run would drop the branch's ops and the submodule. During capture
the branch's one real (dense) run calls the user's own branch function
instead; torch's fake passes and the backward still use the compiled graph.

The branch's ops and the submodule are recorded, outputs and every grad equal
the untraced call, the branch runs exactly once (RNG state and a real-call
counter agree with eager), validation holds and still fails on a planted drop,
and a module ``forward`` called directly still logs and is still noted.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import importlib
import importlib.util
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _capture_state_helpers, _state
from torchlens._utils import _torch_compat
from torchlens.backends.torch import _tracing_pass, rescue
from torchlens.backends.torch._tl import get_module_meta
from torchlens.backends.torch.completeness_witness import _CompletenessDispatchMode
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions

_Mul = Callable[[torch.Tensor, object], torch.Tensor]

#: Real (dense) calls of the counting op; its fake kernel never counts.
_REAL_CALLS: list[int] = [0]

#: The counting custom op, defined on first use.
_COUNT_OP: list[Callable[[torch.Tensor], torch.Tensor]] = []


def _cond() -> Callable[..., Any]:
    """Return ``torch._higher_order_ops.cond.cond``, or skip the test.

    Returns
    -------
    Callable[..., Any]
        The ``cond`` entry point.
    """

    qualname = "torch._higher_order_ops.cond"
    if (
        importlib.util.find_spec("torch._higher_order_ops") is None
        or importlib.util.find_spec(qualname) is None
    ):
        pytest.skip(f"torch {torch.__version__} has no {qualname}")
    entry: Callable[..., Any] = importlib.import_module(qualname).cond
    return entry


def _count_real_calls(tensor: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``tensor`` through a custom op that counts its real calls.

    The op's fake kernel (Dynamo's trace and torch's fake passes) does not
    count, so the counter reads how often the branch really ran. The test
    defines the op first (:func:`_define_count_op`): a definition inside
    Dynamo's trace would break the graph.

    Parameters
    ----------
    tensor:
        Branch value to pass through.

    Returns
    -------
    torch.Tensor
        A copy of ``tensor``.
    """

    return _COUNT_OP[0](tensor)


def _define_count_op() -> Callable[[torch.Tensor], torch.Tensor]:
    """Define the counting custom op with a fake kernel and an identity backward.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The op.
    """

    @torch.library.custom_op("torchlens_test_hccp::count_real_calls", mutates_args=())
    def count_real_calls(tensor: torch.Tensor) -> torch.Tensor:
        """Count one real call and return a copy.

        Parameters
        ----------
        tensor:
            Input.

        Returns
        -------
        torch.Tensor
            A copy of ``tensor``.
        """

        _REAL_CALLS[0] += 1
        return tensor.clone()

    @count_real_calls.register_fake
    def _fake(tensor: torch.Tensor) -> torch.Tensor:
        """Return an empty tensor shaped like ``tensor``, without counting.

        Parameters
        ----------
        tensor:
            Input.

        Returns
        -------
        torch.Tensor
            An uninitialized tensor like ``tensor``.
        """

        return torch.empty_like(tensor)

    def _backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        """Pass the gradient through unchanged.

        Parameters
        ----------
        ctx:
            Unused autograd context.
        grad:
            Output gradient.

        Returns
        -------
        torch.Tensor
            ``grad``.
        """

        return grad

    count_real_calls.register_autograd(_backward)
    return count_real_calls


class _ClosureCond(nn.Module):
    """``cond`` whose true branch calls ``self.lin``, so its parameters are closure tensors."""

    def __init__(
        self,
        body: str = "call",
        take_true: bool = True,
        mul: _Mul | None = None,
        nested: bool = False,
    ) -> None:
        """Build the submodule and choose the true branch's body.

        Parameters
        ----------
        body:
            ``"call"`` (``self.lin(t)``), ``"direct"`` (``self.lin.forward(t)``),
            ``"dropout"``, ``"count"`` or ``"mul"`` (``mul(self.lin(t), 2.0)``).
        take_true:
            Whether the predicate holds for positive inputs.
        mul:
            Multiply the ``"mul"`` body calls.
        nested:
            Pass ``cond`` plain functions defined in ``forward`` instead of
            bound methods.
        """

        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)
        self.cond = _cond()
        self.body = body
        self.take_true = take_true
        self.mul = mul
        self.nested = nested

    def _true_branch(self, t: torch.Tensor) -> torch.Tensor:
        """Run the chosen body through ``self.lin``.

        Parameters
        ----------
        t:
            Operand.

        Returns
        -------
        torch.Tensor
            The body's output.
        """

        if self.body == "direct":
            return self.lin.forward(t)
        out = self.lin(t)
        if self.body == "dropout":
            return F.dropout(out, p=0.5, training=True)
        if self.body == "count":
            return _count_real_calls(out)
        if self.body == "mul" and self.mul is not None:
            return self.mul(out, 2.0)
        return out

    def _false_branch(self, t: torch.Tensor) -> torch.Tensor:
        """Decrement the operand.

        Parameters
        ----------
        t:
            Operand.

        Returns
        -------
        torch.Tensor
            ``t - 1``.
        """

        return t - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a branch, then add one.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The taken branch's output plus one; for the ``"dropout"`` body,
            dropped out again after the ``cond`` so the output also reads the
            RNG state the branch left.
        """

        pred = x.sum() > 0 if self.take_true else x.sum() < -1e9
        if not self.nested:
            out = self.cond(pred, self._true_branch, self._false_branch, (x,)) + 1
            return F.dropout(out, p=0.5, training=True) if self.body == "dropout" else out

        def true_fn(t: torch.Tensor) -> torch.Tensor:
            """Call the true branch through ``self``.

            Parameters
            ----------
            t:
                Operand.

            Returns
            -------
            torch.Tensor
                The true branch's output.
            """

            return self._true_branch(t)

        def false_fn(t: torch.Tensor) -> torch.Tensor:
            """Call the false branch through ``self``.

            Parameters
            ----------
            t:
                Operand.

            Returns
            -------
            torch.Tensor
                The false branch's output.
            """

            return self._false_branch(t)

        return self.cond(pred, true_fn, false_fn, (x,)) + 1


def _input(grad: bool = True) -> torch.Tensor:
    """Return the seeded positive input.

    Parameters
    ----------
    grad:
        Whether the input requires grad.

    Returns
    -------
    torch.Tensor
        A ``(3, 4)`` input whose sum is positive.
    """

    torch.manual_seed(0)
    return (torch.rand(3, 4) + 0.1).requires_grad_(grad)


def _func_names(trace: tl.Trace) -> list[str]:
    """Return the recorded op function names in execution order.

    Parameters
    ----------
    trace:
        A finished trace.

    Returns
    -------
    list[str]
        One name per recorded op.
    """

    return [op.func_name for op in trace]


def _module_func_names(trace: tl.Trace, address: str) -> list[str]:
    """Return the function names of the ops recorded inside one module.

    Parameters
    ----------
    trace:
        A finished trace.
    address:
        Module address.

    Returns
    -------
    list[str]
        One name per op the module's calls recorded.
    """

    return [trace[label].func_name for label in trace.modules[address].layer_labels]


def _output(trace: tl.Trace) -> torch.Tensor:
    """Return the trace's single output activation.

    Parameters
    ----------
    trace:
        A trace captured with ``save="all"``.

    Returns
    -------
    torch.Tensor
        The output activation.
    """

    assert len(trace.output_layers) == 1
    return trace[trace.output_layers[0]].out


@pytest.mark.parametrize("grad", [True, False])
@pytest.mark.parametrize("body", ["call", "direct"])
def test_closure_param_cond_records_the_branch_linear_and_module(body: str, grad: bool) -> None:
    """The branch's ``linear`` op is recorded inside ``lin``, and the output equals eager."""

    model, x = _ClosureCond(body), _input(grad)
    untraced = model(x)
    trace = tl.trace(model, x, save="all")
    assert "linear" in _func_names(trace)
    assert _module_func_names(trace, "lin") == ["linear"]
    assert torch.equal(_output(trace).detach(), untraced.detach())


def test_closure_param_cond_false_branch_leaves_the_module_unrun() -> None:
    """With the false branch taken, no ``linear`` op is recorded and the output equals eager."""

    model, x = _ClosureCond(take_true=False), _input()
    untraced = model(x)
    trace = tl.trace(model, x, save="all")
    assert "linear" not in _func_names(trace)
    assert torch.equal(_output(trace).detach(), untraced.detach())


@pytest.mark.parametrize("take_true", [True, False])
def test_closure_param_cond_grads_equal_eager(take_true: bool) -> None:
    """Backward through the trace gives the eager input and parameter grads."""

    model = _ClosureCond(take_true=take_true)
    reference = copy.deepcopy(model)
    reference_input = _input()
    reference(reference_input).sum().backward()
    traced_input = _input()
    trace = tl.trace(model, traced_input, capture=CaptureOptions(save_grads=True))
    output = trace[trace.output_layers[0]]
    assert output.out.requires_grad
    trace.backward(output.out.sum())
    assert traced_input.grad is not None and reference_input.grad is not None
    torch.testing.assert_close(traced_input.grad, reference_input.grad)
    for name, parameter in reference.lin.named_parameters():
        traced_grad = model.lin.get_parameter(name).grad
        if parameter.grad is None:
            assert traced_grad is None or not traced_grad.any(), name
        else:
            assert traced_grad is not None, name
            torch.testing.assert_close(traced_grad, parameter.grad)


def test_closure_param_cond_runs_the_branch_once_with_eager_rng() -> None:
    """A dropout branch draws eager's masks, and the capture leaves the caller's RNG state.

    The capture reseeds torch with its ``seed`` for its own forward and restores
    the caller's state after (:class:`CaptureOptions`), so ``seed=7`` matches
    an eager call after ``torch.manual_seed(7)``. The second dropout, after the
    ``cond``, draws from the state the branch left: a second run of the branch
    would shift both masks.
    """

    model, x = _ClosureCond("dropout"), _input()
    torch.manual_seed(7)
    untraced = model(x)
    caller_state = torch.get_rng_state()
    trace = tl.trace(model, x, save="all", capture=CaptureOptions(seed=7))
    assert torch.equal(_output(trace).detach(), untraced.detach())
    assert torch.equal(torch.get_rng_state(), caller_state)


def test_closure_param_cond_runs_the_branch_once_by_count() -> None:
    """The branch's real-call counter reads one per forward, as in eager."""

    if not _COUNT_OP:
        _COUNT_OP.append(_define_count_op())
    model, x = _ClosureCond("count"), _input()
    _REAL_CALLS[0] = 0
    untraced = model(x)
    assert _REAL_CALLS[0] == 1
    _REAL_CALLS[0] = 0
    trace = tl.trace(model, x, save="all")
    assert _REAL_CALLS[0] == 1
    assert torch.equal(_output(trace).detach(), untraced.detach())


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("grad", [True, False])
@pytest.mark.parametrize("take_true", [True, False])
def test_closure_param_cond_validates_true(take_true: bool, grad: bool, nested: bool) -> None:
    """``tl.validate`` holds on the closure-parameter ``cond``, bound-method or plain branches.

    Validation's completeness witness is a dispatch mode; it steps aside while
    Dynamo compiles the ``cond`` entry and rejoins the compiled run.
    """

    model, x = _ClosureCond(take_true=take_true, nested=nested), _input(grad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, validate_metadata=True) is True


@pytest.fixture
def raw_mul() -> Iterator[_Mul]:
    """Yield the original ``torch.mul``, taken while TorchLens's wrappers are removed.

    The :class:`OpaqueCallable` holder keeps capture preparation from rebinding
    it, so calling it in the branch plants an op the capture drops.

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


@pytest.mark.parametrize("grad", [True, False])
def test_closure_param_cond_with_a_planted_drop_fails_validation(
    raw_mul: _Mul, grad: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw op the capture drops in the branch fails validation.

    The rescue re-run is disabled so the capture keeps the planted drop.
    """

    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    model, x = _ClosureCond("mul", mul=raw_mul), _input(grad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, validate_metadata=True) is False


@contextlib.contextmanager
def _forced_eager_capture() -> Iterator[None]:
    """Capture with every operator entry compile skipped, as before the compile was lifted.

    Yields
    ------
    None
        Nothing; the context is entered for its side effect.
    """

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            _capture_state_helpers,
            "lift_hop_entry_compiles",
            lambda: contextlib.nullcontext(False),
        )
        yield


@pytest.mark.parametrize("take_true", [True, False])
def test_closure_param_cond_records_the_forced_eager_ops(take_true: bool) -> None:
    """Under ``no_grad`` the lifted capture logs exactly the forced-eager op multiset.

    With grad, the forced-eager capture has nothing to compare: torch's
    autograd path then needs the entry compile to lift the closure parameters.
    """

    model, x = _ClosureCond(take_true=take_true), _input(False)
    with torch.no_grad():
        lifted = _func_names(tl.trace(model, x))
        with _forced_eager_capture():
            forced = _func_names(tl.trace(model, x))
    assert collections.Counter(lifted) == collections.Counter(forced)


def test_direct_forward_call_in_a_cond_branch_is_noted() -> None:
    """A branch's ``self.lin.forward(t)`` is captured and marks ``lin`` as directly called."""

    model, x = _ClosureCond("direct"), _input()
    tl.trace(model, x)
    meta = get_module_meta(model.lin)
    assert meta is not None and meta.direct_forward_called


class _DirectForward(nn.Module):
    """Root calling its submodule's ``forward`` directly, with no ``cond``."""

    def __init__(self) -> None:
        """Build the submodule."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call ``self.lin.forward`` directly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The submodule's output.
        """

        return self.lin.forward(x)


def test_direct_forward_call_outside_a_cond_compile_logs_and_is_noted() -> None:
    """Outside a lifted ``cond`` compile the session wrapper logs and notes a direct call."""

    model = _DirectForward()
    trace = tl.trace(model, _input())
    assert _tracing_pass.cond_entry_innermost is False
    assert _module_func_names(trace, "lin") == ["linear"]
    meta = get_module_meta(model.lin)
    assert meta is not None and meta.direct_forward_called


@pytest.fixture
def witness_on() -> Iterator[None]:
    """Run the test with the completeness witness's dispatch mode on every capture.

    Yields
    ------
    None
        While the wrappers are installed with the witness on; they are
        re-installed with the prior setting after.
    """

    saved_escape: Any = _state._escape_detector_mode
    saved_witness: Any = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=True)
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _witness_census(model: nn.Module, x: torch.Tensor) -> collections.Counter[str]:
    """Capture ``model`` and return every op the witness saw inside its ``cond`` rule.

    Parameters
    ----------
    model:
        Model to capture.
    x:
        Its input.

    Returns
    -------
    collections.Counter[str]
        Census by operator overload name.
    """

    cond_op = _torch_compat.get_higher_order_operator("cond", "cond_op")
    assert cond_op is not None
    rule = cond_op.python_key_table[_CompletenessDispatchMode]
    real_dispatch = _CompletenessDispatchMode.__torch_dispatch__
    inside = [False]
    census: collections.Counter[str] = collections.Counter()

    def spy_rule(mode: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the witness's ``cond`` rule, marking its census window.

        Parameters
        ----------
        mode:
            The witness mode.
        *args, **kwargs:
            The rule's arguments.

        Returns
        -------
        Any
            The rule's result.
        """

        inside[0] = True
        try:
            return rule(mode, *args, **kwargs)
        finally:
            inside[0] = False

    def spy_dispatch(mode: Any, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        """Count ``func`` inside the ``cond`` rule, then dispatch it as the witness does.

        Parameters
        ----------
        mode:
            The witness mode.
        func, types, args, kwargs:
            The dispatched call.

        Returns
        -------
        Any
            The witness's result.
        """

        if inside[0]:
            census[str(func)] += 1
        return real_dispatch(mode, func, types, args, kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(cond_op.python_key_table, _CompletenessDispatchMode, spy_rule)
        patch.setattr(_CompletenessDispatchMode, "__torch_dispatch__", spy_dispatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.trace(model, x).cleanup()
    return census


@pytest.mark.usefixtures("witness_on")
@pytest.mark.parametrize("take_true", [True, False])
def test_closure_param_cond_grad_witness_census_equals_no_grad(take_true: bool) -> None:
    """With a grad operand the witness sees the branch's ops exactly as without one."""

    model = _ClosureCond(take_true=take_true)
    with_grad = _witness_census(model, _input(True))
    without_grad = _witness_census(model, _input(False))
    assert with_grad, with_grad
    assert with_grad == without_grad
    assert any("addmm" in name or "linear" in name for name in with_grad) is take_true


@pytest.mark.usefixtures("witness_on")
def test_closure_param_cond_grad_planted_drop_is_named_by_the_witness(
    raw_mul: _Mul, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under grad, the branch's raw ``mul`` is named by the witness and fails the capture's check.

    The rescue re-run is disabled so the capture keeps the planted drop.
    """

    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    model, x = _ClosureCond("mul", mul=raw_mul), _input(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    try:
        operators = [str(row["operator"]) for row in trace.completeness_diagnostics]
        verified = trace.provenance.verified
    finally:
        trace.cleanup()
    assert "aten.mul.Tensor" in operators, operators
    assert verified is False


@pytest.mark.usefixtures("witness_on")
def test_closure_param_cond_grad_branch_real_calls_are_all_witnessed() -> None:
    """Every real call of a branch op runs inside the witness, none while it steps aside.

    The counting op's real kernel runs only on real tensors; the witness counts
    each call it dispatches. A real call while the witness is off the stack
    (Dynamo's trace of the entry) would leave the real count above the
    witness's.
    """

    if not _COUNT_OP:
        _COUNT_OP.append(_define_count_op())
    model = _ClosureCond("count")
    real_dispatch = _CompletenessDispatchMode.__torch_dispatch__
    witnessed = [0]

    def spy_dispatch(mode: Any, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        """Count the witness's dispatches of the counting op, then dispatch as the witness does.

        Parameters
        ----------
        mode:
            The witness mode.
        func, types, args, kwargs:
            The dispatched call.

        Returns
        -------
        Any
            The witness's result.
        """

        if "count_real_calls" in str(func):
            witnessed[0] += 1
        return real_dispatch(mode, func, types, args, kwargs)

    _REAL_CALLS[0] = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_CompletenessDispatchMode, "__torch_dispatch__", spy_dispatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.trace(model, _input(True)).cleanup()
    assert _REAL_CALLS[0] == 1
    assert witnessed[0] == _REAL_CALLS[0]
