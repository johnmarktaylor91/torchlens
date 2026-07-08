"""Tests for menagerie structural digest entry points."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from menagerie.catalog import CatalogRow, load_rows
from menagerie.recipe import build_model_and_input
from menagerie.structural_digest import (
    architecture_distinctness_hash,
    structural_fingerprint,
)


SAMPLE_MODEL_IDS = (3, 23, 24, 26, 27, 79, 80, 82, 92, 182)


class WidthOnlyModel(nn.Module):
    """Single-layer model whose only architectural difference is output width."""

    def __init__(self, output_width: int) -> None:
        """Initialize the width-only model.

        Parameters
        ----------
        output_width:
            Output feature width for the linear layer.
        """

        super().__init__()
        self.proj = nn.Linear(4, output_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the projection.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected tensor.
        """

        return self.proj(x)


def _rows_by_id() -> dict[int, CatalogRow]:
    """Load menagerie rows keyed by model ID.

    Returns
    -------
    dict[int, CatalogRow]
        Catalog rows keyed by model ID.
    """

    return {row.model_id: row for row in load_rows()}


def _build_sample(row: CatalogRow) -> tuple[Any, Any]:
    """Build a deterministic model/input sample for one menagerie row.

    Parameters
    ----------
    row:
        Menagerie catalog row.

    Returns
    -------
    tuple[Any, Any]
        Model and example input.
    """

    torch.manual_seed(0)
    return build_model_and_input(row)


@pytest.mark.parametrize("model_id", SAMPLE_MODEL_IDS)
def test_structural_fingerprint_is_deterministic_for_menagerie_sample(model_id: int) -> None:
    """Structural fingerprints are deterministic across repeated calls.

    Parameters
    ----------
    model_id:
        Menagerie model ID to build and trace.
    """

    row = _rows_by_id()[model_id]
    model, example_input = _build_sample(row)

    first = structural_fingerprint(model, example_input)
    second = structural_fingerprint(model, example_input)

    assert first == second
    assert len(first) == 64


def test_structural_fingerprint_distinguishes_layer_width_change() -> None:
    """Shape-aware structural fingerprints catch layer-width corruption."""

    input_value = torch.randn(2, 4)
    width_8 = structural_fingerprint(WidthOnlyModel(output_width=8), input_value)
    width_16 = structural_fingerprint(WidthOnlyModel(output_width=16), input_value)

    assert width_8 != width_16


def test_architecture_distinctness_hash_matches_for_layer_width_change() -> None:
    """Shape-blind distinctness hash groups same-topology width variants."""

    input_value = torch.randn(2, 4)
    width_8 = architecture_distinctness_hash(WidthOnlyModel(output_width=8), input_value)
    width_16 = architecture_distinctness_hash(WidthOnlyModel(output_width=16), input_value)

    assert width_8 == width_16
