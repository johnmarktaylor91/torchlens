"""Validation keeps a grad-tracked leaf input a leaf in its private input copies.

A model that sets ``x.requires_grad = True`` on its own input (force-field models
that differentiate energy by position do this) runs fine eagerly, but validation
copied a grad-tracked leaf input through the graph-connected clone, which is a
NON-leaf, so the ground-truth forward raised "you can only change requires_grad
flags of leaf variables". Validation now copies such a leaf as a new leaf.

A plain ``tl.trace`` keeps the graph-connected clone: its contract is that
gradients from the traced output still reach the caller's input.

The same holds for an ``nn.Parameter`` passed as an input, and for the copy's
later move to the model's device (``tensor.to(device)`` of a grad-tracked leaf is
a non-leaf): inside validation both keep the leaf a leaf.
"""

from __future__ import annotations

import torch
from torch import nn

from torchlens._capture_state_helpers import _move_tensors_to_device
from torchlens._utils.arg_handling import safe_copy_input_tree
from torchlens._utils.input_leaves import leaf_preserving_input_copies


class _SetsRequiresGradOnInput(nn.Module):
    """A model that marks its input as requiring grad, like an energy-to-force model."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, cart: torch.Tensor) -> torch.Tensor:
        cart.requires_grad = True
        return self.lin(cart).sum()


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


def test_device_move_keeps_a_leaf_inside_validation_scope_only() -> None:
    """A device move keeps a grad-tracked leaf a leaf only inside validation's scope."""

    x = torch.randn(2, 2, requires_grad=True)
    with leaf_preserving_input_copies():
        (moved,) = _move_tensors_to_device([x], "meta")
    assert moved.device.type == "meta"
    assert moved.is_leaf and moved.requires_grad
    (plain_moved,) = _move_tensors_to_device([x], "meta")
    assert plain_moved.device.type == "meta"
    assert not plain_moved.is_leaf
