"""Op-level placement on a real torch model against its real MLX twin (same weights).

The unit is a gated MLP (``down(act(gate(x)) * up(x))``): its three projections are child
modules that pair and agree, so the walk runs over the activation and the product. A
correct twin agrees; a twin whose activation is GELU instead of SiLU is placed on that op,
and a twin that spells SiLU as ``x * sigmoid(x)`` agrees.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = [pytest.mark.backend_mlx, pytest.mark.optional, pytest.mark.rare]

pytest.importorskip("pandas")
mx = pytest.importorskip("mlx.core")
mnn = pytest.importorskip("mlx.nn")

_WIDTH, _HIDDEN = 8, 12
_SINGLE = re.compile(r"op: a (?P<a>\S+) against b (?P<b>\S+);")


class _TorchMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(_WIDTH, _HIDDEN, bias=False)
        self.up_proj = nn.Linear(_WIDTH, _HIDDEN, bias=False)
        self.down_proj = nn.Linear(_HIDDEN, _WIDTH, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TorchNet(nn.Module):
    """``mlp(x) + x``."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TorchMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + x


class _MLXMLP(mnn.Module):
    """The MLX port; ``act`` is ``"silu"``, ``"gelu"`` (the bug) or ``"manual"``."""

    def __init__(self, act: str) -> None:
        super().__init__()
        self.gate_proj = mnn.Linear(_WIDTH, _HIDDEN, bias=False)
        self.up_proj = mnn.Linear(_WIDTH, _HIDDEN, bias=False)
        self.down_proj = mnn.Linear(_HIDDEN, _WIDTH, bias=False)
        self.act = act

    def __call__(self, x: Any) -> Any:
        gate = self.gate_proj(x)
        if self.act == "gelu":
            gate = mnn.gelu(gate)
        elif self.act == "manual":
            gate = gate * mx.sigmoid(gate)
        else:
            gate = mnn.silu(gate)
        return self.down_proj(gate * self.up_proj(x))


class _MLXNet(mnn.Module):
    """``mlp(x) + x``."""

    def __init__(self, act: str) -> None:
        super().__init__()
        self.mlp = _MLXMLP(act)

    def __call__(self, x: Any) -> Any:
        return self.mlp(x) + x


def _pair(act: str) -> tuple[tl.Trace, tl.Trace]:
    """Trace the torch net and its MLX twin on one input, weights copied across."""

    torch.manual_seed(0)
    reference = _TorchNet().eval()
    twin = _MLXNet(act)
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = getattr(reference.mlp, name).weight.detach().numpy()
        getattr(twin.mlp, name).weight = mx.array(weight)
    x = np.linspace(-1.0, 1.0, 2 * _WIDTH, dtype=np.float32).reshape(2, _WIDTH)
    return tl.trace(reference, torch.from_numpy(x)), tl.trace(twin, mx.array(x), backend="mlx")


def test_correct_twin_agrees() -> None:
    assert tl.debug.first_divergence(*_pair("silu")) is None


def test_silu_spelled_by_hand_agrees() -> None:
    assert tl.debug.first_divergence(*_pair("manual")) is None


def test_gelu_for_silu_is_placed_on_the_activation_op() -> None:
    # The first differing call is ``mlp.down_proj`` (its input already differs); the
    # difference is localized inside ``mlp``, whose inputs agree, and placed there.
    row = tl.debug.first_divergence(*_pair("gelu"))
    assert row is not None and str(row["op"]).startswith("mlp.down_proj:")
    assert "op placement inside mlp:" in str(row["reason"]), row["reason"]
    match = _SINGLE.search(str(row["reason"]))
    assert match is not None, row["reason"]
    assert match["a"].startswith("silu_") and match["b"].startswith("gelu_"), row["reason"]
