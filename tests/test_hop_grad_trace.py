"""Capture of higher-order operators whose operands require grad.

With a grad-requiring operand, torch's Autograd kernel for ``cond``,
``while_loop``, ``map`` and ``scan`` first traces the branch or body with
data-free ``FakeTensor`` inputs, then runs it eagerly. TorchLens must stay out
of that fake-tensor pass and record only the eager run.

``cond`` runs the same program with or without grad, so its grad trace records
exactly the no-grad ops. ``while_loop`` and ``scan`` run a different eager
program on the grad path (stacked loop outputs, the scan's traced combine
graph), and TorchLens records what executed. For them the grad trace must
reproduce the no-grad outputs bit for bit, validate True (and still fail
validation when a capture bug is planted), keep the no-grad ops as an ordered
subsequence, hold only real activations, and carry backward.

Capture runs under the ``force_eager`` stance, but each operator's own entry
compile still runs (as in plain torch), so ``map`` returns its stacked tensor
and ``cond`` carries backward. Those compiles are internal to the operator:
they are not a user ``torch.compile`` region, record exactly the ops the
forced-eager capture records, and leave the user's later operator calls
compiling as before. A user ``torch.compile`` region around an operator is
still detected and refused exactly as before.
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import importlib.util
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _capture_state_helpers, _state
from torchlens._utils import _torch_compat
from torchlens.backends.torch import rescue, wrappers
from torchlens.backends.torch.completeness_witness import _CompletenessDispatchMode
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions

_Mul = Callable[[torch.Tensor, object], torch.Tensor]

_TRACING_TENSOR_NAMES = frozenset({"FakeTensor", "FunctionalTensor"})


def _hop(module: str) -> Callable[..., Any]:
    """Return ``torch._higher_order_ops.<module>.<module>``, or skip the test.

    Parameters
    ----------
    module:
        Submodule name, which is also its entry point's name.

    Returns
    -------
    Callable[..., Any]
        The higher-order operator's entry point.
    """

    qualname = f"torch._higher_order_ops.{module}"
    # Presence is probed with ``find_spec`` (parent package first, so a torch
    # without the package answers None instead of raising), never by catching
    # ImportError: a HOP module that exists but fails to import must fail
    # loudly rather than skip. Only the tests that build this op's model skip.
    if (
        importlib.util.find_spec("torch._higher_order_ops") is None
        or importlib.util.find_spec(qualname) is None
    ):
        pytest.skip(f"torch {torch.__version__} has no {qualname}")
    entry: Callable[..., Any] = getattr(importlib.import_module(qualname), module)
    return entry


def _clean_mul(tensor: torch.Tensor, factor: object) -> torch.Tensor:
    """Multiply through the Tensor ``*`` operator, which TorchLens captures.

    Parameters
    ----------
    tensor:
        Tensor to scale.
    factor:
        Scale factor.

    Returns
    -------
    torch.Tensor
        ``tensor * factor``.
    """

    return tensor * factor


class _CondModel(nn.Module):
    """``cond`` whose predicate takes the true or the false branch."""

    def __init__(self, take_true: bool) -> None:
        """Store which branch the predicate selects.

        Parameters
        ----------
        take_true:
            Whether the predicate holds for the test inputs.
        """

        super().__init__()
        self.take_true = take_true
        self.cond = _hop("cond")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double (true branch) or decrement (false branch), then add one.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The taken branch's output plus one.
        """

        pred = x.sum() > 0 if self.take_true else x.sum() < -1e9
        return self.cond(pred, lambda t: t * 2, lambda t: t - 1, (x,)) + 1


class _WhileModel(nn.Module):
    """``while_loop`` that scales its carry three times."""

    hop_module = "while_loop"

    def __init__(self, mul: _Mul = _clean_mul) -> None:
        """Store the multiply the loop body runs.

        Parameters
        ----------
        mul:
            Multiply called once per iteration.
        """

        super().__init__()
        self.mul = mul
        self.hop = _hop(self.hop_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three doubling iterations, then add one.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final carry plus one.
        """

        start = torch.zeros((), dtype=torch.int64)
        out = self.hop(lambda i, t: i < 3, lambda i, t: (i + 1, self.mul(t, 2.0)), (start, x))
        return out[1] + 1


class _MapModel(nn.Module):
    """``map`` over the leading dimension."""

    hop_module = "map"

    def __init__(self, mul: _Mul = _clean_mul) -> None:
        """Store the multiply the mapped function runs.

        Parameters
        ----------
        mul:
            Multiply called once per row.
        """

        super().__init__()
        self.mul = mul
        self.hop = _hop(self.hop_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map each row to ``2 * row + 1``.

        Parameters
        ----------
        x:
            Input tensor; rows are mapped.

        Returns
        -------
        torch.Tensor
            The stacked mapped rows.
        """

        return self.hop(lambda row: self.mul(row, 2.0) + 1, x)


class _ScanModel(nn.Module):
    """``scan`` computing a scaled running sum."""

    hop_module = "scan"

    def __init__(self, mul: _Mul = _clean_mul) -> None:
        """Store the multiply the combine step runs.

        Parameters
        ----------
        mul:
            Multiply called once per step.
        """

        super().__init__()
        self.mul = mul
        self.hop = _hop(self.hop_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the summed running sums plus the final carry.

        Parameters
        ----------
        x:
            Input tensor; rows are scanned.

        Returns
        -------
        torch.Tensor
            Sum of the step outputs plus the final carry.
        """

        def combine(carry: torch.Tensor, row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Add the row to the carry and emit the scaled new carry.

            Parameters
            ----------
            carry:
                Running sum so far.
            row:
                Current row.

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                New carry and this step's output.
            """

            new = carry + row
            return new, self.mul(new, 2.0)

        carry, ys = self.hop(combine, torch.zeros(4), x)
        return ys.sum(0) + carry


_COND_CASES: dict[str, Callable[[], nn.Module]] = {
    "cond_true": lambda: _CondModel(True),
    "cond_false": lambda: _CondModel(False),
}

_LOOP_CASES: dict[str, Callable[..., nn.Module]] = {
    "map": _MapModel,
    "while_loop": _WhileModel,
    "scan": _ScanModel,
}

_ALL_CASES = sorted([*_COND_CASES, *_LOOP_CASES])
# ``map`` runs its body inside torch's own ``autograd.Function``, so its body ops
# are logged below autograd and carry no grad, as in any custom Function forward.
_DIFFERENTIABLE_BODY_CASES = [name for name in _ALL_CASES if name != "map"]


def _input(name: str) -> torch.Tensor:
    """Return the seeded no-grad input for one case.

    Parameters
    ----------
    name:
        Case name.

    Returns
    -------
    torch.Tensor
        A ``(3, 4)`` input; positive for ``cond_true`` so its predicate holds.
    """

    torch.manual_seed(0)
    if name == "cond_true":
        return torch.rand(3, 4) + 0.1
    return torch.randn(3, 4)


def _model(name: str, mul: _Mul = _clean_mul) -> nn.Module:
    """Build the model for one case.

    Parameters
    ----------
    name:
        Case name.
    mul:
        Multiply the loop-style models run in their body.

    Returns
    -------
    nn.Module
        The model.
    """

    if name in _COND_CASES:
        return _COND_CASES[name]()
    return _LOOP_CASES[name](mul)


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


def _output_values(trace: tl.Trace) -> list[torch.Tensor]:
    """Return the saved activations of the trace's output layers, in order.

    Parameters
    ----------
    trace:
        A trace captured with ``save="all"``.

    Returns
    -------
    list[torch.Tensor]
        Detached output activations.
    """

    return [trace[label].out.detach() for label in trace.output_layers]


def _is_ordered_subsequence(short: list[str], long: list[str]) -> bool:
    """Return whether ``short`` appears in ``long`` in order, gaps allowed.

    Parameters
    ----------
    short:
        Candidate subsequence.
    long:
        Sequence to search.

    Returns
    -------
    bool
        ``True`` when every item of ``short`` is matched left to right.
    """

    remaining = iter(long)
    return all(item in remaining for item in short)


@pytest.mark.parametrize("name", sorted(_COND_CASES))
def test_cond_grad_operands_record_the_no_grad_ops(name: str) -> None:
    """A grad-requiring ``cond`` operand records exactly the no-grad taken-branch ops."""

    model, x = _model(name), _input(name)
    no_grad_names = _func_names(tl.trace(model, x))
    grad_names = _func_names(tl.trace(model, x.clone().requires_grad_(True)))
    assert grad_names == no_grad_names


@pytest.mark.parametrize("name", sorted(_LOOP_CASES))
def test_loop_hop_grad_outputs_are_bit_identical(name: str) -> None:
    """The grad trace's outputs equal the no-grad trace's and the untraced call's."""

    model, x = _model(name), _input(name)
    untraced = model(x)
    no_grad = _output_values(tl.trace(model, x, save="all"))
    grad = _output_values(tl.trace(model, x.clone().requires_grad_(True), save="all"))
    assert len(no_grad) == len(grad) == 1
    assert torch.equal(no_grad[0], untraced)
    assert torch.equal(grad[0], untraced)


@pytest.mark.parametrize("name", sorted(_LOOP_CASES))
def test_loop_hop_no_grad_ops_are_an_ordered_subsequence(name: str) -> None:
    """Every no-grad op appears, in order, on the grad path.

    Names are compared through TorchLens's rescue canonicalisation, which maps
    the dunder spelling (``__add__``) and the function spelling (``add``) of
    one op to a single name; ``scan``'s grad path runs its traced combine graph
    and so records the function spellings.
    """

    model, x = _model(name), _input(name)
    no_grad = [rescue._canonical_op_name(n) for n in _func_names(tl.trace(model, x))]
    grad_trace = tl.trace(model, x.clone().requires_grad_(True))
    grad = [rescue._canonical_op_name(n) for n in _func_names(grad_trace)]
    assert len(grad) >= len(no_grad)
    assert _is_ordered_subsequence(no_grad, grad), (no_grad, grad)


@pytest.mark.parametrize("name", _ALL_CASES)
def test_grad_operand_trace_validates_true(name: str) -> None:
    """``tl.validate`` holds on the grad trace, with the grad path's extra data ops."""

    model, x = _model(name), _input(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x.requires_grad_(True), validate_metadata=True) is True


def test_map_grad_validation_sees_the_one_stacked_output() -> None:
    """Under grad, validating ``map`` holds with no failure recorded.

    The validate capture's completeness dispatch mode steps aside for
    Dynamo's frame evaluation of ``map``'s entry compile and is re-entered
    for the compiled graph's run, so the capture returns the single stacked
    tensor the untraced call returns, and the witness still censuses the body.
    """

    model, x = _model("map"), _input("map")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x.requires_grad_(True), validate_metadata=True) is True
    assert tl.validation.last_validation_failure() is None


@pytest.fixture
def raw_mul() -> Iterator[_Mul]:
    """Yield the original ``torch.mul``, taken while TorchLens's wrappers are removed.

    The original is a builtin TorchLens never sees; the
    :class:`OpaqueCallable` holder keeps capture preparation from rebinding it.
    Calling it in a HOP body plants an op the capture drops. The wrappers are
    re-installed with the session's diagnostic modes before the test and again
    at teardown.

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


def test_while_loop_grad_trace_with_a_planted_drop_fails_validation(
    raw_mul: _Mul, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw op the capture drops in the loop body still fails validation under grad.

    The rescue re-run is disabled so the capture keeps the planted drop. (A raw
    op in a ``scan`` body is not a drop on the grad path: the body runs as the
    graph torch traced from it, whose ops are captured.)
    """

    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    model, x = _model("while_loop", raw_mul), _input("while_loop")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x.requires_grad_(True), validate_metadata=True) is False


def _plant_over_broad_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant a tracing-pass guard that also passes any call scaling by ``2.0`` unlogged.

    The body's ``* 2.0`` then runs on the eager run without being logged, the
    capture bug an over-broad fake-pass test would cause. The rescue re-run is
    disabled so the capture keeps the drop.

    Parameters
    ----------
    monkeypatch:
        The test's monkeypatch; both patches are undone at teardown.
    """

    real_guard = wrappers.inside_tracing_tensor_pass

    def over_broad_guard(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
        """Report the real tracing pass, and also any call scaling by ``2.0``.

        Parameters
        ----------
        args:
            Positional arguments of the decorated call.
        kwargs:
            Keyword arguments of the decorated call.

        Returns
        -------
        bool
            Whether the wrapper should pass the call through unlogged.
        """

        planted = any(isinstance(arg, float) and arg == 2.0 for arg in args)
        return planted or real_guard(args, kwargs)

    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    monkeypatch.setattr(wrappers, "inside_tracing_tensor_pass", over_broad_guard)


def _plant_unpaused_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant the over-broad guard and run the calls it swallows unpaused.

    A real tracing-pass call runs paused, so the witness never counts it. A
    wrapper that skips logging a model op does not pause, so the witness sees
    that op's aten call with nothing logged for it: the drop the census exists
    to name.

    Parameters
    ----------
    monkeypatch:
        The test's monkeypatch; every patch is undone at teardown.
    """

    _plant_over_broad_guard(monkeypatch)
    real_call = wrappers.call_inside_tracing_pass

    def unpaused_call(
        func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Run a planted call unpaused and unlogged, and any other call as before.

        Parameters
        ----------
        func:
            The original torch callable.
        args:
            Positional arguments of the call.
        kwargs:
            Keyword arguments of the call.

        Returns
        -------
        Any
            Whatever ``func`` returns.
        """

        if any(isinstance(arg, float) and arg == 2.0 for arg in args):
            return func(*args, **kwargs)
        return real_call(func, args, kwargs)

    monkeypatch.setattr(wrappers, "call_inside_tracing_pass", unpaused_call)


@pytest.mark.parametrize("name", sorted(_LOOP_CASES))
def test_loop_hop_grad_trace_with_an_over_broad_guard_fails_validation(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracing-pass guard that also swallows an eager body op fails validation.

    The planted guard passes the body's ``* 2.0`` through unlogged on the eager
    run, the capture bug an over-broad fake-pass test would cause.
    """

    _plant_over_broad_guard(monkeypatch)
    model, x = _model(name), _input(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x.requires_grad_(True), validate_metadata=True) is False


@pytest.mark.parametrize("name", _ALL_CASES)
def test_grad_operands_leave_only_real_activations(name: str) -> None:
    """Every activation recorded under grad is a real tensor, never fake or meta."""

    model, x = _model(name), _input(name)
    trace = tl.trace(model, x.requires_grad_(True), save="all")
    saved = [op for op in trace if op.has_saved_activation]
    assert saved
    for op in saved:
        assert isinstance(op.out, torch.Tensor), op.func_name
        assert type(op.out).__name__ not in _TRACING_TENSOR_NAMES, op.func_name
        assert op.out.device.type == x.device.type, op.func_name


@pytest.mark.parametrize("name", _DIFFERENTIABLE_BODY_CASES)
def test_grad_operands_backward_runs_through_the_trace(name: str) -> None:
    """The traced output carries the graph, and backward gives the eager grads."""

    model, x = _model(name), _input(name)
    reference_input = x.clone().requires_grad_(True)
    model(reference_input).sum().backward()
    traced_input = x.clone().requires_grad_(True)
    trace = tl.trace(model, traced_input, capture=CaptureOptions(save_grads=True))
    assert len(trace.output_layers) == 1
    output = trace[trace.output_layers[0]]
    assert output.out.requires_grad
    trace.backward(output.out.sum())
    assert traced_input.grad is not None
    torch.testing.assert_close(traced_input.grad, reference_input.grad)
    producer = trace[output.parents[0]]
    assert producer.has_saved_grad
    assert type(producer.grads.for_pass(1).grad).__name__ not in _TRACING_TENSOR_NAMES


def test_map_grad_operands_backward_runs_through_the_trace() -> None:
    """``map``: the output carries the graph, backward gives eager grads, the input saves its grad.

    The body ops ran inside torch's map ``autograd.Function`` forward, below
    autograd, so the stacked producer has no graph of its own and no grad to save.
    """

    x = _input("map")
    reference_input = x.clone().requires_grad_(True)
    _model("map")(reference_input).sum().backward()
    traced_input = x.clone().requires_grad_(True)
    trace = tl.trace(_model("map"), traced_input, capture=CaptureOptions(save_grads=True))
    assert len(trace.output_layers) == 1
    output = trace[trace.output_layers[0]]
    assert output.out.grad_fn is not None
    trace.backward(output.out.sum())
    assert traced_input.grad is not None
    torch.testing.assert_close(traced_input.grad, reference_input.grad)
    source = trace[trace.input_layers[0]]
    assert source.has_saved_grad
    torch.testing.assert_close(source.grads.for_pass(1).grad, reference_input.grad)
    assert trace[output.parents[0]].out.grad_fn is None


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


@pytest.mark.parametrize(
    ("name", "grad"),
    [(n, False) for n in _ALL_CASES]
    + [(n, True) for n in sorted(_COND_CASES)]
    + [("while_loop", True)],
)
def test_hop_entry_compile_records_the_forced_eager_ops(name: str, grad: bool) -> None:
    """The lifted entry compile logs exactly the op multiset the forced-eager capture logs.

    A grad-requiring ``map`` has no forced-eager capture to compare (it returned
    per-row outputs), and ``scan``'s grad path runs a different program.
    """

    model, x = _model(name), _input(name)
    lifted = _func_names(tl.trace(model, x.clone().requires_grad_(grad)))
    with _forced_eager_capture():
        forced = _func_names(tl.trace(model, x.clone().requires_grad_(grad)))
    assert collections.Counter(lifted) == collections.Counter(forced)


@pytest.fixture
def witness_on() -> Iterator[None]:
    """Run the test with the completeness witness's dispatch mode on every capture.

    The wrappers are re-installed with the witness on, and re-installed with
    the session's diagnostic modes at teardown.

    Yields
    ------
    None
        Nothing; the fixture is used for its side effect.
    """

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=True)
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _map_body_census(x: torch.Tensor) -> tuple[collections.Counter[str], tl.Trace]:
    """Capture the ``map`` model and return the witness's census of ``map``'s kernel.

    The census holds the aten calls the witness mode observed while its ``map``
    rule ran (the body rows, plus the kernel's unstacking and stacking).

    Parameters
    ----------
    x:
        The model input.

    Returns
    -------
    tuple[collections.Counter[str], tl.Trace]
        The census by aten overload name, and the trace.
    """

    map_impl = _torch_compat.get_higher_order_operator("map", "map_impl")
    assert map_impl is not None
    rule = map_impl.python_key_table[_CompletenessDispatchMode]
    real_dispatch = _CompletenessDispatchMode.__torch_dispatch__
    inside = [False]
    census: collections.Counter[str] = collections.Counter()

    def spy_rule(mode: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the witness's ``map`` rule, marking its census window.

        Parameters
        ----------
        mode:
            The witness mode.
        *args, **kwargs:
            The operator's arguments.

        Returns
        -------
        Any
            The rule's output.
        """

        inside[0] = True
        try:
            return rule(mode, *args, **kwargs)
        finally:
            inside[0] = False

    def spy_dispatch(mode: Any, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        """Count ``func`` inside the ``map`` rule, then dispatch it as the witness does.

        Parameters
        ----------
        mode:
            The witness mode.
        func:
            The aten overload.
        types:
            Tensor types of the call.
        args, kwargs:
            The call's arguments.

        Returns
        -------
        Any
            The witness's result.
        """

        if inside[0]:
            census[str(func)] += 1
        return real_dispatch(mode, func, types, args, kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(map_impl.python_key_table, _CompletenessDispatchMode, spy_rule)
        patch.setattr(_CompletenessDispatchMode, "__torch_dispatch__", spy_dispatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = tl.trace(_model("map"), x)
    return census, trace


@pytest.mark.usefixtures("witness_on")
def test_map_grad_witness_census_of_the_body_equals_the_forced_eager_capture() -> None:
    """Under grad, the witness censuses ``map``'s body exactly as the forced-eager capture.

    The lifted entry compile runs the compiled graph with the witness re-entered,
    so the witness sees every aten call of the body rows; the capture returns the
    one stacked output the untraced call returns, and is verified.
    """

    x = _input("map")
    expected = _model("map")(x)
    lifted, trace = _map_body_census(x.clone().requires_grad_(True))
    try:
        outputs = _output_values(trace)
        verified = trace.provenance.verified
    finally:
        trace.cleanup()
    with _forced_eager_capture():
        forced, forced_trace = _map_body_census(x.clone().requires_grad_(True))
    forced_trace.cleanup()
    assert lifted["aten.mul.Tensor"] == 3, lifted
    assert lifted == forced
    assert len(outputs) == 1 and torch.equal(outputs[0], expected)
    assert verified is True


@pytest.mark.usefixtures("witness_on")
def test_map_grad_planted_drop_is_named_by_the_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under grad, a body op the capture leaves unlogged inside ``map`` is named by the census.

    The planted over-broad guard passes the body's ``* 2.0`` through unlogged
    during the compiled graph's run; the witness, re-entered for that run,
    reports the ``aten.mul`` and the capture is not verified.
    """

    _plant_over_broad_guard(monkeypatch)
    x = _input("map").requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_model("map"), x)
    try:
        operators = [str(row["operator"]) for row in trace.completeness_diagnostics]
        verified = trace.provenance.verified
        outputs = len(trace.output_layers)
    finally:
        trace.cleanup()
    assert any(name.startswith("aten.mul.") for name in operators), operators
    assert verified is False
    assert outputs == 1


def test_map_grad_validation_with_a_planted_drop_fails_on_the_census(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under grad, ``tl.validate`` fails ``map`` with a dropped body op, not on output count."""

    _plant_over_broad_guard(monkeypatch)
    model, x = _model("map"), _input("map")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x.requires_grad_(True), validate_metadata=True) is False
    failure = tl.validation.last_validation_failure()
    assert failure is not None
    assert failure.check_name != "ground_truth_missing_out", failure


@pytest.mark.parametrize("name", _ALL_CASES)
def test_hop_entry_compile_is_not_a_user_compile_region(name: str) -> None:
    """The operator's own compile neither raises the Dynamo-region flag nor its warning."""

    model, x = _model(name), _input(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, x.requires_grad_(True))
    assert trace._raw_dynamo_region_detected is False
    assert not [w for w in caught if "torch.compile (Dynamo) region" in str(w.message)]


class _UserCompiledCond(nn.Module):
    """A user ``torch.compile`` region wrapped around a ``cond``."""

    def __init__(self) -> None:
        """Compile the region as a plain attribute, which capture cannot unwrap."""

        super().__init__()
        self.cond = _hop("cond")
        self.region = torch.compile(self._branch, backend="eager")

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ``cond``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``2 * x`` for a positive sum, else ``x - 1``.
        """

        return self.cond(x.sum() > 0, lambda t: t * 2, lambda t: t - 1, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the compiled region.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The region's output plus one.
        """

        return self.region(x) + 1


def test_user_compile_region_around_a_hop_is_still_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the stance, a user region around a ``cond`` is detected and refused as before."""

    monkeypatch.setattr(_torch_compat, "HAS_SET_STANCE", False)
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(_UserCompiledCond(), torch.rand(3, 4) + 0.1)
    assert trace._raw_dynamo_region_detected is True
    assert [w for w in caught if "torch.compile (Dynamo) region" in str(w.message)]
    assert _func_names(trace) == ["none", "__add__", "none"]


@pytest.mark.parametrize("name", _ALL_CASES)
def test_capture_leaves_the_users_hop_compiles_working(name: str) -> None:
    """After grad captures and a validation, the untraced operator call still compiles and matches."""

    model, x = _model(name), _input(name)
    expected = model(x.clone().requires_grad_(True)).detach()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x.clone(), validate_metadata=True) is True
        tl.trace(model, x.clone().requires_grad_(True))
    assert torch.equal(model(x.clone().requires_grad_(True)).detach(), expected)


@pytest.mark.parametrize("name", _ALL_CASES)
def test_untraced_call_after_a_capture_reuses_the_users_own_compile(name: str) -> None:
    """A capture's operator compiles leave the user's Dynamo cache unpoisoned: no new compile."""

    model, x = _model(name), _input(name)
    counters = _torch_compat.get_dynamo_compile_counters(force_probe=True)
    assert counters is not None
    for _ in range(2):
        model(x.clone().requires_grad_(True))
    tl.trace(model, x.clone().requires_grad_(True))
    compiled_before = counters["frames"]["total"]
    model(x.clone().requires_grad_(True))
    assert counters["frames"]["total"] == compiled_before
