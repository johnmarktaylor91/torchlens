"""A01 semantic-attention slice: detection + merge semantics units.

The fused-class gate is dead: fusedness comes from the captured graph with
config corroboration. These units pin the F2 merge rule (a produced value
beats an equal-tier absence), the primary-tuple-output resolution, the
config-based head-geometry fallback, the flash-attention teaching absence,
and the projection-orientation refusal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import facets as facets_mod
from torchlens.semantic.recipes._helpers import (
    attention_implementation,
    config_object_value,
    module_output_spec,
)
from torchlens.semantic.recipes.attention import (
    _no_attention_evidence,
    _projection_orientation,
)

pytestmark = [pytest.mark.smoke]


class _TupleOut(nn.Module):
    """Module returning (primary, secondary) with distinct shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        main = self.lin(x)
        aux = main.sum(dim=-1, keepdim=True) * torch.ones(1, 8)
        return main, aux


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _TupleOut()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(x)


def test_module_output_spec_resolves_the_primary_tuple_leaf():
    """F2: a multi-output module resolves attn_out-style reads to leaf 0."""

    torch.manual_seed(0)
    model = _Wrapper()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x)
    module = trace.modules["block"]
    spec = module_output_spec(module, "unit")
    assert not isinstance(spec, facets_mod.AbsenceReason), spec
    value = facets_mod.Facet(spec).value
    direct_main = model.block(x)[0]
    assert value.shape == direct_main.shape
    assert value.shape == (2, 4)


def test_value_beats_equal_tier_absence():
    """F2 merge rule: an equal-tier absence never pops a produced value."""

    values = {"foo": 1}
    missing: dict = {}
    menu: dict = {}
    tiers = {"foo": (2, 0)}
    recipe = SimpleNamespace(public=SimpleNamespace(recipe_name="unit_recipe"))
    reason = facets_mod.AbsenceReason(status="structurally_absent", detail="claimed absent")
    facets_mod._merge_missing(
        "foo",
        reason,
        recipe=recipe,  # type: ignore[arg-type]
        tier=(2, 0),
        values=values,
        missing=missing,
        menu=menu,
        facet_tiers=tiers,
    )
    assert values == {"foo": 1}, "equal-tier absence popped the produced value (F2)"
    assert "foo" not in missing
    # A strictly higher-tier absence still displaces the value.
    facets_mod._merge_missing(
        "foo",
        reason,
        recipe=recipe,  # type: ignore[arg-type]
        tier=(3, 0),
        values=values,
        missing=missing,
        menu=menu,
        facet_tiers=tiers,
    )
    assert "foo" not in values
    assert "foo" in missing


def test_config_object_fallback_serves_head_geometry():
    """Head counts resolve from the captured HF config snapshot (third tier)."""

    config = SimpleNamespace(num_attention_heads=12, num_key_value_heads=4, head_dim=None)
    module = SimpleNamespace(custom_attributes={"config": config})
    assert config_object_value(module, "num_attention_heads") == 12
    assert config_object_value(module, "num_key_value_heads") == 4
    # None config fields fall through instead of being served.
    assert config_object_value(module, "head_dim") is None


def test_attention_implementation_reads_module_then_config():
    module = SimpleNamespace(
        custom_attributes={"config": SimpleNamespace(_attn_implementation="sdpa")}
    )
    assert attention_implementation(module) == "sdpa"
    direct = SimpleNamespace(custom_attributes={"_attn_implementation": "eager"})
    assert attention_implementation(direct) == "eager"
    assert attention_implementation(SimpleNamespace(custom_attributes={})) is None


def test_flash_attention_absence_teaches_both_remedies():
    """An external fused kernel yields needs_capture naming read AND write remedies."""

    module = SimpleNamespace(
        address="model.layers.0.self_attn",
        custom_attributes={"config": SimpleNamespace(_attn_implementation="flash_attention_2")},
    )
    reason = _no_attention_evidence(module)
    assert reason.status == "needs_capture"
    assert "flash_attention_2" in reason.detail
    assert "reconstruction_ready=True" in reason.detail
    assert "eager" in reason.detail


def test_no_evidence_absence_is_structural_and_names_the_gap():
    module = SimpleNamespace(address="blk.attn", custom_attributes={})
    reason = _no_attention_evidence(module)
    assert reason.status == "structurally_absent"
    assert "no attention score computation was captured" in reason.detail


def test_projection_orientation_class_table_and_shape_proof():
    """Conv1D/Linear key the table; unknown classes need a shape proof; square refuses."""

    conv_weight = torch.zeros(8, 16)  # [in=n*d, out]
    lin_weight = torch.zeros(16, 8)  # [out, in=n*d]
    square = torch.zeros(8, 8)
    assert _projection_orientation("Conv1D", conv_weight, 2, 4) == "in_out"
    assert _projection_orientation("Linear", lin_weight, 2, 4) == "out_in"
    # Unknown class, non-square weight: the shape proves the orientation.
    assert _projection_orientation("MyProj", conv_weight, 2, 4) == "in_out"
    assert _projection_orientation("MyProj", lin_weight, 2, 4) == "out_in"
    # Unknown class, square weight: refuse -- never guess (mikit F5 class).
    assert _projection_orientation("MyProj", square, 2, 4) is None
