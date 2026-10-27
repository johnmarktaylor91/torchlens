"""Validation keeps a grad-tracked leaf input a leaf in its private input copies.

A model that sets ``x.requires_grad = True`` on its own input (force-field models
that differentiate energy by position do this) runs fine eagerly, but validation
copied a grad-tracked leaf input through the graph-connected clone, which is a
NON-leaf, so the ground-truth forward raised "you can only change requires_grad
flags of leaf variables". Validation now copies such a leaf as a new leaf.

A plain ``tl.trace`` keeps the graph-connected clone: its contract is that
gradients from the traced output still reach the caller's input.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.utils.arg_handling import leaf_preserving_input_copies, safe_copy_input_tree


class _SetsRequiresGradOnInput(nn.Module):
    """A model that marks its input as requiring grad, like an energy-to-force model."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, cart: torch.Tensor) -> torch.Tensor:
        cart.requires_grad = True
        return self.lin(cart).sum()


def test_model_setting_requires_grad_on_leaf_input_validates() -> None:
    """The synthetic trigger: a grad-tracked leaf input the model re-marks validates."""

    torch.manual_seed(0)
    x = torch.randn(3, 4).requires_grad_(True)
    assert bool(tl.validate(_SetsRequiresGradOnInput().eval(), x, scope="forward"))
    assert x.is_leaf and x.requires_grad


def test_leaf_preserving_copy_keeps_a_grad_tracked_leaf_a_leaf() -> None:
    """Inside validation's scope a grad-tracked leaf copies as a new leaf."""

    base = torch.randn(4, 3).requires_grad_(True)
    with leaf_preserving_input_copies():
        (copied_base,), _kwargs, gaps = safe_copy_input_tree([base], {})
    assert gaps == ()
    assert copied_base is not base
    assert copied_base.is_leaf and copied_base.requires_grad
    assert torch.equal(copied_base, base)
    assert copied_base.grad_fn is None


def test_plain_copy_of_grad_tracked_leaf_stays_graph_connected() -> None:
    """Outside validation the historical clone keeps gradients reaching the caller."""

    x = torch.randn(3, 4).requires_grad_(True)
    (copied,), _kwargs, gaps = safe_copy_input_tree([x], {})
    assert gaps == ()
    assert not copied.is_leaf
    copied.sum().backward()
    assert x.grad is not None


def test_non_leaf_input_copy_is_unchanged_inside_validation_scope() -> None:
    """A non-leaf input keeps the graph-connected clone even inside validation's scope."""

    leaf = torch.randn(3, 4).requires_grad_(True)
    non_leaf = leaf * 2.0
    with leaf_preserving_input_copies():
        (copied,), _kwargs, gaps = safe_copy_input_tree([non_leaf], {})
    assert gaps == ()
    assert not copied.is_leaf
    assert torch.equal(copied, non_leaf)
    copied.sum().backward()
    assert leaf.grad is not None


def test_trace_output_still_backpropagates_to_caller_leaf_input() -> None:
    """``tl.trace`` is unaffected: the traced output's gradient reaches the caller's input."""

    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    x = torch.randn(3, 4, requires_grad=True)
    trace = tl.trace(model, x)
    trace.output_ops[0].out.sum().backward()
    assert x.grad is not None
