"""An in-place write through a view keeps every op's backward node in the recorded graph.

Autograd's ``rebase_history`` gives the base of a written view a new ``CopySlices``
node and drops the view's own node from the graph (for a view of a view, every
intermediate view node too). The op that produced the view recorded that dropped
node, so no backward pass ever reached it and the metadata invariant reported a
missing GradFn backpointer, inside an ``autograd.grad`` region and after a managed
backward alike. Capture now moves such an op onto the ``CopySlices`` node that took
its place. The invariant itself is unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation._invariants_backward_graph import _check_backward_graph_invariants


def _view_used(y: torch.Tensor) -> torch.Tensor:
    """Write a column view in place, then read the view."""

    return y[:, 5:6].add_(0.25)


def _view_of_view(y: torch.Tensor) -> torch.Tensor:
    """Write a view of a view in place."""

    return y[:, 2:8][:, 3:4].mul_(1.5)


def _chained(y: torch.Tensor) -> torch.Tensor:
    """Write the same view twice."""

    col = y[:, 5:6]
    col.add_(0.25)
    return col.mul_(3.0)


_WRITES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "view": _view_used,
    "view_of_view": _view_of_view,
    "chained": _chained,
}


class _GradRegion(nn.Module):
    """Write a column view of an intermediate inside an ``autograd.grad`` region."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Linear(1, 8)
        self.write = write

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an energy and its gradient with respect to ``x``."""

        x = x.detach().clone().requires_grad_(True)
        col = self.write(x * 2.0)
        energy = (self.emb(col) ** 2).sum() + x.sum() * 1e-3
        (grad,) = torch.autograd.grad(energy, x)
        return energy, grad


class _Plain(nn.Module):
    """The same write with no gradient region; the caller runs backward."""

    def __init__(self, write: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)
        self.emb = nn.Linear(1, 4)
        self.write = write

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a projection of the written view."""

        return self.emb(self.write(self.lin(x)))


def _input() -> torch.Tensor:
    return torch.randn(3, 8, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_view_write_inside_autograd_grad_validates(write: str) -> None:
    """Full validation (metadata included) passes for a view write in a grad region."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_GradRegion(_WRITES[write]), _input(), scope="forward")


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_view_write_backward_graph_keeps_view_ops(write: str) -> None:
    """After a managed backward every view op points at a node the pass visited."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_Plain(_WRITES[write]), _input().requires_grad_(True))
        trace.backward(trace[trace.output_layers[0]].out.sum())
    _check_backward_graph_invariants(trace)
    getitems = [layer for layer in trace._layer_list if layer.func_name == "__getitem__"]
    assert getitems
    for layer in getitems:
        assert layer.autograd.grad_fn is not None
        assert layer.autograd.grad_fn.class_name == "CopySlices"


def test_view_write_backward_validates() -> None:
    """Backward validation passes for a model that writes a view in place."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_Plain(_view_used), _input(), scope="backward")


def test_plain_view_read_keeps_its_own_node() -> None:
    """A view that is only read (never written) keeps the node its op produced."""

    class _ReadOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)[:, 5:6] * 3.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_ReadOnly(), _input().requires_grad_(True))
        trace.backward(trace[trace.output_layers[0]].out.sum())
    _check_backward_graph_invariants(trace)
    (getitem,) = [layer for layer in trace._layer_list if layer.func_name == "__getitem__"]
    assert getitem.autograd.grad_fn.class_name == "SliceBackward0"


class _ConsumedBeforeWrite(nn.Module):
    """Read a view, then write it in place; the earlier reader keeps the view's node."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sum that reaches the view's node through the earlier reader."""

        y = self.lin(x)
        v = y[:, 5:6]
        a = v * 3.0
        v.add_(1.0)
        return a.sum() + y.sum()


def _only_layer(trace: tl.Trace, func_names: tuple[str, ...]) -> tl.Layer:
    """Return the one layer whose function name is in ``func_names``."""

    (layer,) = [layer for layer in trace._layer_list if layer.func_name in func_names]
    return layer


def test_view_consumed_before_write_keeps_its_node() -> None:
    """A view read before the write keeps its node; the reader's edge lands on its op.

    The reader's ``MulBackward0`` still points at the view's ``SliceBackward0``, so
    that node stays in the graph and the view op's recorded handle was already
    right. Rebinding it to the base's ``CopySlices`` would leave the reader's edge
    on a node no op owns while the graph still validated.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_ConsumedBeforeWrite(), _input().requires_grad_(True))
        trace.backward(trace[trace.output_layers[0]].out.sum())
    _check_backward_graph_invariants(trace)
    getitem = _only_layer(trace, ("__getitem__",))
    mul = _only_layer(trace, ("__mul__", "mul"))
    assert getitem.autograd.grad_fn.class_name == "SliceBackward0"
    mul_next_ids = mul.autograd.grad_fn.next_grad_fn_ids
    assert getitem.autograd.grad_fn.grad_fn_object_id in mul_next_ids
    assert all(trace._grad_fn_logs[next_id].has_op for next_id in mul_next_ids)


def test_view_consumed_before_write_backward_validates() -> None:
    """Backward validation passes when a view is read before its in-place write."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_ConsumedBeforeWrite(), _input(), scope="backward")
