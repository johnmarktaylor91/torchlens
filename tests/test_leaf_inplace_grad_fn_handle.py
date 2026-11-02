"""A same-object in-place op that returns a leaf records no autograd handle.

``z = x.detach().requires_grad_(True)`` inside a forward returns its receiver, a
leaf with ``grad_fn is None``. The op is logged against a safe copy whose own
``grad_fn`` is TorchLens's ``CloneBackward`` bookkeeping node; the record used to
fall back to that node and carry a phantom ``grad_fn_object_id`` no backward can
visit, which ``backward_graph_invariants`` reported as a missing GradFn
backpointer as soon as an in-forward ``torch.autograd.grad`` gave the trace
backward evidence (stress rows ``bw_double`` and ``bw_input_grad``). The record now
carries ``None``; a same-object in-place op on a non-leaf keeps its real node.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class InputGradientInForward(nn.Module):
    """Return the input-gradient of a scalar beside the forward output."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(torch.relu(self.fc1(z)).sum(), z)
        return self.fc2(x) + g


class InplaceOnNonLeaf(nn.Module):
    """An in-place op whose receiver is a non-leaf activation with a live node."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(x))
        h.add_(1.0)
        return h * 2.0


def _layer(trace: tl.Trace, prefix: str) -> tl.Layer:
    matches = [layer for layer in trace.layers if layer.layer_label.startswith(prefix)]
    assert matches, [layer.layer_label for layer in trace.layers]
    return matches[0]


@pytest.mark.smoke
def test_requires_grad_leaf_records_no_grad_fn_handle() -> None:
    trace = tl.trace(InputGradientInForward(), [torch.randn(2, 4)])
    leaf = _layer(trace, "requiresgrad")
    assert leaf.grad_fn_object_id is None
    assert leaf.grad_fn_class_name is None
    assert leaf.grad_fn_class_qualname is None


def test_in_forward_input_gradient_validates() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(InputGradientInForward(), torch.randn(2, 4), scope="forward")


def test_same_object_inplace_on_non_leaf_keeps_its_real_node() -> None:
    trace = tl.trace(InplaceOnNonLeaf(), [torch.randn(2, 4)])
    add = _layer(trace, "add_")
    assert add.grad_fn_object_id is not None
    assert add.grad_fn_class_name == "AddBackward0"
