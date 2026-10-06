"""Held pristine torch functions behave exactly like direct calls under capture.

Capture preparation rebinds a module-held pristine torch function to its
wrapper, so a held ``F.relu`` (the default activation
``nn.TransformerEncoderLayer`` binds at torch import) gets the same outer
``F.relu`` / inner ``torch.relu`` call nesting, the same ``func_call_id``
sequence and the same graph as ``F.relu(x)`` written in ``forward``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch.wrappers import wrap_torch


def _pristine(func: Callable[..., Any]) -> Callable[..., Any]:
    """The original behind an installed wrapper (the object a pre-wrap holder keeps)."""

    wrap_torch()
    return _state._decorated_to_orig.get(id(func), func)


def _call_signature(trace: tl.Trace) -> list[tuple[str, Any, tuple[str, ...]]]:
    """Per op: function name, ``func_call_id`` and parent labels, in capture order."""

    return [(op.func_name, op.func_call_id, tuple(op.parents)) for op in trace.ops]


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(model.eval(), x)


class _HeldRelu(nn.Module):
    """Linear, then a pristine ``F.relu`` held as an attribute."""

    def __init__(self, act: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer and the held activation."""

        return self.act(self.fc(x))


class _DirectRelu(nn.Module):
    """Linear, then ``F.relu`` called directly."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer and ``F.relu``."""

        return F.relu(self.fc(x))


def test_held_python_functional_matches_a_direct_call() -> None:
    """A held pristine ``F.relu`` and a direct ``F.relu(x)`` capture identically.

    Same call nesting (outer ``F.relu`` around inner ``torch.relu``), same
    ``func_call_id`` sequence, same graph.
    """

    x = torch.randn(2, 4)
    torch.manual_seed(0)
    held = _HeldRelu(_pristine(F.relu))
    torch.manual_seed(0)
    direct = _DirectRelu()

    assert _call_signature(_capture(held, x)) == _call_signature(_capture(direct, x))


def test_transformer_layer_built_before_or_after_wrap_captures_identically() -> None:
    """A ``TransformerEncoderLayer`` built pre-wrap and one built post-wrap match.

    Its default ``activation=F.relu`` is bound at torch import, so both hold
    the pristine ``F.relu``; both are rebound for the capture.
    """

    x = torch.randn(2, 3, 4)
    torch.manual_seed(0)
    before = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)
    wrap_torch()
    torch.manual_seed(0)
    after = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)

    assert _call_signature(_capture(before, x)) == _call_signature(_capture(after, x))


def test_transformer_layer_activation_matches_a_direct_relu_call() -> None:
    """The layer's held activation gets the same nesting as a direct ``F.relu`` call.

    Compared on the ``func_call_id`` sequence: the held activation, once
    rebound, consumes exactly the ids a layer whose activation is the live
    (wrapped) ``F.relu`` consumes.
    """

    x = torch.randn(2, 3, 4)
    torch.manual_seed(0)
    held = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)
    wrap_torch()
    torch.manual_seed(0)
    live = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)
    live.activation = F.relu  # the wrapper, exactly what a direct call reaches

    assert held.activation is not live.activation
    assert _call_signature(_capture(held, x)) == _call_signature(_capture(live, x))
