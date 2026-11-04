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

Capture runs under the ``force_eager`` stance. A grad-requiring ``map`` then
returns one tensor per row instead of the stacked tensor (plain torch does the
same under that stance), so its capture keeps failing loudly rather than
returning a different structure. ``cond`` backward under that stance fails in
plain torch too, so backward is checked for ``while_loop`` and ``scan``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import rescue, wrappers
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions

_hop_cond = pytest.importorskip("torch._higher_order_ops.cond").cond
_hop_while = pytest.importorskip("torch._higher_order_ops.while_loop").while_loop
_hop_map = pytest.importorskip("torch._higher_order_ops.map").map
_hop_scan = pytest.importorskip("torch._higher_order_ops.scan").scan

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


def test_map_grad_operand_capture_fails_loudly() -> None:
    """A grad-requiring ``map`` fails capture instead of returning per-row outputs.

    Under the capture stance torch's eager ``map`` autograd returns one tensor
    per row; recording that would silently differ from the untraced call.
    """

    torch.manual_seed(0)
    x = torch.randn(3, 4).requires_grad_(True)
    with pytest.raises(AttributeError, match="fake_mode"):
        tl.trace(_MapModel(), x)


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


@pytest.mark.parametrize("name", sorted(_LOOP_CASES))
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
