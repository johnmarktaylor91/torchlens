"""Behavioral tests for the gated (Llama-style) MLP recipe."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl


class LlamaMLP(nn.Module):
    """Tiny Llama-style gated MLP surrogate matching the recipe class name."""

    def __init__(self, hidden: int = 8, intermediate: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _GatedModel(nn.Module):
    """Wrapper exposing one named gated MLP module."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def test_gated_mlp_recipe_populates_all_facets_with_real_values() -> None:
    """The gated MLP recipe exposes gate/up/intermediate/down facets that match math."""

    torch.manual_seed(13)
    model = _GatedModel(LlamaMLP())
    x = torch.randn(2, 3, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))

    view = log.modules["mlp"].facets
    assert view.recipe_source == "gated_mlp"
    assert view.up_out.shape == (2, 3, 16)
    assert view.gated_out.shape == (2, 3, 16)
    assert view.intermediate.shape == (2, 3, 16)
    assert view.down_out.shape == (2, 3, 8)
    assert view.output.shape == (2, 3, 8)
    assert view.input.shape == (2, 3, 8)

    mlp = model.mlp
    with torch.no_grad():
        expected_gated = F.silu(mlp.gate_proj(x))
        expected_intermediate = expected_gated * mlp.up_proj(x)
    assert torch.allclose(view.gated_out, expected_gated, atol=1e-6)
    assert torch.allclose(view.intermediate, expected_intermediate, atol=1e-6)
    assert torch.allclose(view.output, model(x), atol=1e-6)


def test_gated_mlp_recipe_degrades_when_projection_children_missing() -> None:
    """A class-name match without the expected children yields no phantom facets."""

    class _Partial(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)))

    _Partial.__name__ = "LlamaMLP"
    _Partial.__qualname__ = "LlamaMLP"

    torch.manual_seed(14)
    log = tl.trace(
        _GatedModel(_Partial()),
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    view = log.modules["mlp"].facets

    # up_proj is absent, so up_out and the computed intermediate cannot exist.
    assert "up_out" not in view
    assert "intermediate" not in view
    assert view.gated_out.shape == (2, 16)
    assert view.down_out.shape == (2, 8)
