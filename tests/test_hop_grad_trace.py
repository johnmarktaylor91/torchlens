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
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions

_HOP_NAMES = ("cond", "while_loop", "map", "scan")
_HOPS_PRESENT = all(
    importlib.util.find_spec(f"torch._higher_order_ops.{name}") is not None for name in _HOP_NAMES
)
pytestmark = pytest.mark.skipif(
    not _HOPS_PRESENT, reason="this torch has no cond, while_loop, map and scan operators"
)


def _hop(name: str) -> Callable[..., Any]:
    """Return torch's user-facing higher-order operator ``name``.

    Parameters
    ----------
    name:
        Operator, also the name of its ``torch._higher_order_ops`` submodule.

    Returns
    -------
    Callable[..., Any]
        The operator, or :func:`_absent_hop` when this torch lacks the
        operators (every test in the module is then skipped).
    """

    if not _HOPS_PRESENT:
        return _absent_hop
    return vars(importlib.import_module(f"torch._higher_order_ops.{name}"))[name]


def _absent_hop(*args: Any, **kwargs: Any) -> Any:
    """Stand in for an operator this torch lacks; never called, the module is skipped.

    Parameters
    ----------
    *args:
        Ignored.
    **kwargs:
        Ignored.

    Returns
    -------
    Any
        Never returns.
    """

    raise AssertionError("higher-order operator called on a torch that lacks it")


_hop_cond, _hop_while, _hop_map, _hop_scan = (_hop(name) for name in _HOP_NAMES)

_Mul = Callable[[torch.Tensor, object], torch.Tensor]

_TRACING_TENSOR_NAMES = frozenset({"FakeTensor", "FunctionalTensor"})


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
        return _hop_cond(pred, lambda t: t * 2, lambda t: t - 1, (x,)) + 1


class _WhileModel(nn.Module):
    """``while_loop`` that scales its carry three times."""

    def __init__(self, mul: _Mul = _clean_mul) -> None:
        """Store the multiply the loop body runs.

        Parameters
        ----------
        mul:
            Multiply called once per iteration.
        """

        super().__init__()
        self.mul = mul

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
        out = _hop_while(lambda i, t: i < 3, lambda i, t: (i + 1, self.mul(t, 2.0)), (start, x))
        return out[1] + 1


class _MapModel(nn.Module):
    """``map`` over the leading dimension."""

    def __init__(self, mul: _Mul = _clean_mul) -> None:
        """Store the multiply the mapped function runs.

        Parameters
        ----------
        mul:
            Multiply called once per row.
        """

        super().__init__()
        self.mul = mul

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

        return _hop_map(lambda row: self.mul(row, 2.0) + 1, x)


class _ScanModel(nn.Module):
    """``scan`` computing a scaled running sum."""

    def __init__(self, mul: _Mul = _clean_mul) -> None:
        """Store the multiply the combine step runs.

        Parameters
        ----------
        mul:
            Multiply called once per step.
        """

        super().__init__()
        self.mul = mul

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

        carry, ys = _hop_scan(combine, torch.zeros(4), x)
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


@pytest.mark.parametrize("name", sorted(_LOOP_CASES))
def test_loop_hop_grad_trace_with_an_over_broad_guard_fails_validation(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracing-pass guard that also swallows an eager body op fails validation.

    The planted guard passes the body's ``* 2.0`` through unlogged on the eager
    run, the capture bug an over-broad fake-pass test would cause.
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


@pytest.mark.parametrize("name", _ALL_CASES)
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
        self.region = torch.compile(_UserCompiledCond._branch, backend="eager")

    @staticmethod
    def _branch(x: torch.Tensor) -> torch.Tensor:
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

        return _hop_cond(x.sum() > 0, lambda t: t * 2, lambda t: t - 1, (x,))

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
