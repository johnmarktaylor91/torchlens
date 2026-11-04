"""Capture of higher-order operators whose operands require grad.

With a grad-requiring operand, torch's Autograd kernel for ``cond``,
``while_loop``, ``map`` and ``scan`` first traces the branch or body with
data-free ``FakeTensor`` inputs, then runs it eagerly. TorchLens must stay out
of that fake-tensor pass and record only the eager run: the same ops as the
no-grad trace (the taken branch only), real activations, and a graph that
backward runs through.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

_hop_cond = pytest.importorskip("torch._higher_order_ops.cond").cond
_hop_while = pytest.importorskip("torch._higher_order_ops.while_loop").while_loop
_hop_map = pytest.importorskip("torch._higher_order_ops.map").map
_hop_scan = pytest.importorskip("torch._higher_order_ops.scan").scan

_TRACING_TENSOR_NAMES = frozenset({"FakeTensor", "FunctionalTensor"})


class _CondModel(nn.Module):
    """``cond`` whose predicate takes the true or the false branch."""

    def __init__(self, take_true: bool) -> None:
        """Store which branch the predicate selects."""
        super().__init__()
        self.take_true = take_true

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double (true branch) or decrement (false branch), then add one."""
        pred = x.sum() > 0 if self.take_true else x.sum() < -1e9
        return _hop_cond(pred, lambda t: t * 2, lambda t: t - 1, (x,)) + 1


class _CondLinearModel(nn.Module):
    """``cond`` whose taken branch calls a parameterized submodule."""

    def __init__(self) -> None:
        """Build the branch's linear layer."""
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer when the sum is positive."""
        return _hop_cond(x.sum() > 0, lambda t: self.lin(t), lambda t: t - 1, (x,))


class _WhileModel(nn.Module):
    """``while_loop`` that doubles its carry three times."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three doubling iterations, then add one."""
        start = torch.zeros((), dtype=torch.int64)
        out = _hop_while(lambda i, t: i < 3, lambda i, t: (i + 1, t * 2), (start, x))
        return out[1] + 1


class _MapModel(nn.Module):
    """``map`` over the leading dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map each row to ``2 * row + 1``."""
        return _hop_map(lambda row: row * 2 + 1, x)


class _ScanModel(nn.Module):
    """``scan`` computing a running sum."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the summed running sums plus the final carry."""

        def combine(carry: torch.Tensor, row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Add the row to the carry and emit the new carry."""
            new = carry + row
            return new, new * 1

        carry, ys = _hop_scan(combine, torch.zeros(4), x)
        return ys.sum(0) + carry


_CASES: dict[str, tuple[Callable[[], nn.Module], Callable[[], torch.Tensor]]] = {
    "cond_true": (lambda: _CondModel(True), lambda: torch.rand(3, 4) + 0.1),
    "cond_false": (lambda: _CondModel(False), lambda: torch.randn(3, 4)),
    "cond_linear": (_CondLinearModel, lambda: torch.rand(4) + 0.1),
    "while_loop": (_WhileModel, lambda: torch.randn(3, 4)),
    "map": (_MapModel, lambda: torch.randn(3, 4)),
    "scan": (_ScanModel, lambda: torch.randn(3, 4)),
}


def _case(name: str) -> tuple[nn.Module, torch.Tensor]:
    """Return a seeded model and a fresh no-grad input for one HOP case."""
    torch.manual_seed(0)
    build_model, build_input = _CASES[name]
    return build_model(), build_input()


def _func_names(trace: tl.Trace) -> list[str]:
    """Return the recorded op function names in execution order."""
    return [op.func_name for op in trace]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_grad_operands_record_the_no_grad_ops(name: str) -> None:
    """A grad-requiring operand records the same taken-branch ops as no-grad."""
    model, x = _case(name)
    no_grad_names = _func_names(tl.trace(model, x))
    grad_names = _func_names(tl.trace(model, x.clone().requires_grad_(True)))
    assert grad_names == no_grad_names


@pytest.mark.parametrize("name", sorted(_CASES))
def test_grad_operands_leave_no_fake_tensor_activation(name: str) -> None:
    """No activation recorded under grad comes from the fake-tensor pass."""
    model, x = _case(name)
    trace = tl.trace(model, x.requires_grad_(True), save="all")
    saved = [op for op in trace if op.has_saved_activation]
    assert saved
    for op in saved:
        assert isinstance(op.out, torch.Tensor)
        assert type(op.out).__name__ not in _TRACING_TENSOR_NAMES, op.func_name
        assert op.out.device.type == x.device.type, op.func_name


@pytest.mark.parametrize("name", sorted(_CASES))
def test_grad_operands_backward_runs_through_the_trace(name: str) -> None:
    """The traced output carries the graph, and backward gives the eager grads."""
    model, x = _case(name)
    reference_input = x.clone().requires_grad_(True)
    model(reference_input).sum().backward()
    traced_input = x.clone().requires_grad_(True)
    trace = tl.trace(model, traced_input, capture=CaptureOptions(save_grads=True))
    output = trace[trace.output_layers[0]]
    assert output.out.requires_grad
    trace.backward(output.out.sum())
    assert traced_input.grad is not None
    torch.testing.assert_close(traced_input.grad, reference_input.grad)
    assert output.has_saved_grad
    assert type(output.grads.for_pass(1).grad).__name__ not in _TRACING_TENSOR_NAMES
