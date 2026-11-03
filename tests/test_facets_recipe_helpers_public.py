"""Custom facet recipes need no private import: ``tl.facets`` helpers and ``heads(n)``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.facets import FacetSpec, child_output_spec, config_value, parameter_spec


class MyAttention(nn.Module):
    """Fused-projection attention whose head count is a plain attribute."""

    def __init__(self, d_model: int = 16, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        return self.out(q * k.sigmoid() + v)


def _recipe(module: Any) -> dict[str, Any]:
    n_heads = config_value(module, "num_heads", "n_heads")
    qkv = child_output_spec(module, "qkv", "my_attention_test")
    return {
        "q": qkv.split(3, dim=-1)[0].heads(n_heads),
        "q_explicit": qkv.split(3, dim=-1)[0].heads(n_heads, 4),
        "q_bad": qkv.split(3, dim=-1)[0].heads(5),
        "out_weight": parameter_spec(module, "out.weight", "my_attention_test"),
    }


@pytest.fixture(autouse=True)
def _register() -> Iterator[None]:
    from torchlens._semantic import facets as _facets

    saved = list(_facets._REGISTRY)
    tl.facets.register(
        class_name="MyAttention",
        target_scope="module",
        facets=("q", "q_explicit", "q_bad", "out_weight"),
    )(_recipe)
    try:
        yield
    finally:
        _facets._REGISTRY[:] = saved
        _facets._REGISTRY_VERSION += 1


def test_helpers_are_public_and_identical_to_the_canonical_ones() -> None:
    """The three helpers live on ``tl.facets`` and are the recipe helpers themselves."""

    from torchlens._semantic.recipes import _helpers

    assert tl.facets.config_value is _helpers.config_value
    assert tl.facets.child_output_spec is _helpers.child_output_spec
    assert tl.facets.parameter_spec is _helpers.parameter_spec
    assert {"child_output_spec", "config_value", "parameter_spec"} <= set(tl.facets.__all__)
    assert isinstance(FacetSpec, type)


def test_heads_infers_the_head_dimension_and_refuses_a_bad_split() -> None:
    """``heads(n)`` reads ``d_head`` from the last axis; a non-divisor refuses."""

    torch.manual_seed(0)
    model = nn.Sequential(MyAttention()).eval()
    trace = tl.trace(model, torch.randn(1, 5, 16))
    facets = trace.modules["0"].facets

    q = facets["q"].value
    assert tuple(q.shape) == (1, 5, 4, 4)
    assert torch.equal(q, facets["q_explicit"].value)
    with pytest.raises(ValueError, match="not divisible by 5"):
        _ = facets["q_bad"].value
    assert tuple(facets["out_weight"].value.shape) == (16, 16)


def test_heads_rejects_a_non_positive_count() -> None:
    """A bogus head count is refused when the facet is read."""

    from torchlens._semantic.facets import _heads_geometry

    with pytest.raises(ValueError, match="positive integer head count"):
        _heads_geometry(torch.zeros(2, 8), 0, None)
    with pytest.raises(ValueError, match="positive integer head count"):
        _heads_geometry(torch.zeros(2, 8), True, None)
    assert _heads_geometry(torch.zeros(2, 8), 2, None) == (2, 4)
    assert _heads_geometry(torch.zeros(2, 8), 2, 3) == (2, 3)
