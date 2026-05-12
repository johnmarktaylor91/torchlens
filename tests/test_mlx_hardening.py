"""Hardening tests for the technical-preview MLX backend."""

from __future__ import annotations

from typing import Any

import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402


class TinyMLP(nn.Module):
    """Small MLX MLP used by backend hardening tests."""

    def __init__(self) -> None:
        """Initialize the MLP layers."""

        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 2)

    def __call__(self, x: mx.array) -> mx.array:
        """Run the MLP forward pass."""

        hidden = nn.relu(self.l1(x))
        return self.l2(hidden)


def _tiny_mlp_input() -> mx.array:
    """Return a deterministic-shape MLX input array for hardening tests."""

    return mx.random.normal((2, 4))


@pytest.mark.optional
def test_mlx_intervention_ready_raises() -> None:
    """MLX capture rejects intervention metadata requests explicitly."""

    with pytest.raises(NotImplementedError, match="intervention_ready"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), intervention_ready=True, vis_opt="none")


@pytest.mark.optional
def test_mlx_save_grads_raises() -> None:
    """MLX capture rejects backward-gradient capture explicitly."""

    with pytest.raises(NotImplementedError, match="backward capture"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), save_grads=True, vis_opt="none")


@pytest.mark.optional
def test_mlx_hooks_raise() -> None:
    """MLX capture rejects pre-attached live hook plans explicitly."""

    hooks: list[dict[str, Any]] = [{"target": "linear_1_1", "action": lambda x: x}]
    with pytest.raises(NotImplementedError, match="hooks"):
        tl.trace(TinyMLP(), _tiny_mlp_input(), hooks=hooks, vis_opt="none")
