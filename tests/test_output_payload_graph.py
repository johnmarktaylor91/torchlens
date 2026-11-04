"""Output rows whose returned tensor gained autograd history after its producer was logged.

A producer logged inside an ``autograd.Function`` forward runs below autograd,
so its retained payload has no ``grad_fn``; the Function sets the history on
the returned tensor afterwards. Unless the trace detaches saved activations,
that output row keeps its own graph-connected copy, the handle ``backward()``
differentiates through, and the byte accounting charges the copy. Every other
output row still rides its producer's one retained payload.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _Double(torch.autograd.Function):
    """Custom autograd function doubling its input."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Return the doubled input.

        Parameters
        ----------
        ctx:
            Autograd context (unused).
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``x * 2``, computed below autograd.
        """

        return x * 2

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return the doubled upstream gradient.

        Parameters
        ----------
        ctx:
            Autograd context (unused).
        grad:
            Upstream gradient.

        Returns
        -------
        torch.Tensor
            ``grad * 2``.
        """

        return grad * 2


class _DirectFunctionOutput(nn.Module):
    """Model returning the custom function's output with no op after it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the function's output directly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``_Double.apply(x)``.
        """

        return _Double.apply(x)


class _PlainOp(nn.Module):
    """Model returning a plain logged op's output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x * 2``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The doubled input.
        """

        return x * 2


def _output_row(trace: tl.Trace) -> tl.Op:
    """Return the trace's single output row.

    Parameters
    ----------
    trace:
        A finished trace with one output.

    Returns
    -------
    tl.Op
        The output row.
    """

    assert len(trace.output_layers) == 1
    return trace[trace.output_layers[0]]


def test_direct_function_output_keeps_a_graph_with_eager_grads() -> None:
    """The output row carries the Function's graph, and backward gives the eager grads."""

    torch.manual_seed(0)
    x = torch.randn(2, 3)
    reference = x.clone().requires_grad_(True)
    _DirectFunctionOutput()(reference).sum().backward()
    traced = x.clone().requires_grad_(True)
    trace = tl.trace(_DirectFunctionOutput(), traced)
    output = _output_row(trace)
    assert output.out.grad_fn is not None
    trace.backward(output.out.sum())
    assert traced.grad is not None
    torch.testing.assert_close(traced.grad, reference.grad)


def test_direct_function_output_copy_is_charged_in_the_byte_accounting() -> None:
    """The graph-connected output copy is its own storage, and the saved bytes count it."""

    trace = tl.trace(_DirectFunctionOutput(), torch.randn(2, 3).requires_grad_(True))
    output = _output_row(trace)
    parent = trace[output.parents[0]]
    assert output.out.untyped_storage().data_ptr() != parent.out.untyped_storage().data_ptr()
    assert int(trace.cost.saved_activation_memory) > int(trace.cost.total_activation_memory)


def test_plain_op_output_still_rides_the_producer_payload() -> None:
    """A plain op's output row shares its producer's one payload: no extra copy."""

    trace = tl.trace(_PlainOp(), torch.randn(2, 3).requires_grad_(True))
    output = _output_row(trace)
    assert output.out is trace[output.parents[0]].out
    assert int(trace.cost.saved_activation_memory) == int(trace.cost.total_activation_memory)


def test_detached_trace_keeps_the_function_output_detached() -> None:
    """With saved activations detached, the output row keeps no graph, as before."""

    trace = tl.trace(
        _DirectFunctionOutput(),
        torch.randn(2, 3).requires_grad_(True),
        capture=CaptureOptions(detach_saved_activations=True),
    )
    assert _output_row(trace).out.grad_fn is None
