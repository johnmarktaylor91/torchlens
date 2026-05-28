"""Tests for facets and portable serialization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import facets as facets_mod


class _LinearModel(nn.Module):
    """Small model used for serialization tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x)


def test_facets_rerive_after_tlspec_load_when_recipe_registered(tmp_path: Any) -> None:
    """Facet views are not serialized and are rebuilt against the current registry."""

    original = list(facets_mod._REGISTRY)
    try:

        @tl.facets.register(class_name="Linear")
        def linear_recipe(record: Any) -> dict[str, Any]:
            """Return a user facet for Linear modules."""

            return {"linear_out_shape": tuple(record.out.shape)}

        path = tmp_path / "linear.tlspec"
        log = tl.trace(_LinearModel(), torch.randn(1, 3), layers_to_save="all")
        log.save(path)

        facets_mod._REGISTRY[:] = original
        loaded_without_recipe = tl.load(path)
        assert not loaded_without_recipe.modules["linear"].facets.has("linear_out_shape")

        tl.facets.register(class_name="Linear")(linear_recipe)
        loaded_with_recipe = tl.load(path)
        assert loaded_with_recipe.modules["linear"].facets.linear_out_shape == (1, 3)
    finally:
        facets_mod._REGISTRY[:] = original
