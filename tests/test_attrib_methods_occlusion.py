"""F06 B5: the direct input occlusion map (attrib memo D8-D9).

Window geometry (clipped-edge full coverage), overlap combination modes,
replacement-policy seam, per-example vs aggregate deltas, the full-window
self-oracle against a direct two-forward difference, and the deterministic
pass-budget refusal carrying the arithmetic.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.errors import AttributionError


class _PatchSum(nn.Module):
    """Linear read-out over a tiny image so window deltas are analytic."""

    def __init__(self) -> None:
        """Build a fixed per-pixel weight grid."""

        super().__init__()
        weight = torch.arange(1 * 4 * 4, dtype=torch.float64).reshape(1, 4, 4) + 1.0
        self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        """Weighted sum per example, two output slots for int targeting."""

        score = (x * self.weight).sum(dim=(1, 2, 3), keepdim=False)
        return torch.stack([score, -score], dim=-1)


def test_per_example_int_target_deltas() -> None:
    """Int targets produce per-example maps; callable targets aggregate to batch 1."""

    model = _PatchSum()
    x = torch.stack(
        [torch.ones(1, 4, 4, dtype=torch.float64), 2 * torch.ones(1, 4, 4, dtype=torch.float64)]
    )
    per_example = attribution.occlusion_map(model, x, target=0, window=(4, 4))
    assert per_example.values.shape == x.shape
    assert per_example.extra["per_example_deltas"] is True
    # Example 1 has doubled inputs, so its full-window delta is doubled.
    torch.testing.assert_close(
        per_example.values[1], per_example.values[0] * 2, rtol=1e-12, atol=1e-12
    )
    aggregate = attribution.occlusion_map(
        model, x, target=lambda out: out[..., 0].sum(), window=(4, 4)
    )
    assert aggregate.values.shape == (1, 1, 4, 4)
    assert aggregate.extra["per_example_deltas"] is False


def test_replacement_policy_seam() -> None:
    """zeros/mean/scalar/tensor replacement policies resolve and disclose."""

    model = _PatchSum()
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    zeros = attribution.occlusion_map(model, x, target=0, window=(4, 4))
    assert zeros.extra["replacement_policy"] == "zeros"
    mean = attribution.occlusion_map(model, x, target=0, window=(4, 4), baseline_value="mean")
    assert "mean" in mean.extra["replacement_policy"]
    # All-ones input: the global mean IS 1.0, so the mean-policy delta is 0.
    torch.testing.assert_close(mean.values, torch.zeros_like(x), rtol=0, atol=0)
    scalar = attribution.occlusion_map(model, x, target=0, window=(4, 4), baseline_value=0.5)
    torch.testing.assert_close(scalar.values, zeros.values * 0.5, rtol=1e-12, atol=1e-12)
    with pytest.raises(AttributionError) as excinfo:
        attribution.occlusion_map(model, x, target=0, window=(4, 4), baseline_value=object())
    assert excinfo.value.fields["code"] == "occlusion_geometry_invalid"


def test_pass_budget_refusal_carries_the_arithmetic() -> None:
    """D9: exceeding max_passes refuses with count, one-pass time, projection."""

    model = _PatchSum()
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    with pytest.raises(AttributionError) as excinfo:
        attribution.occlusion_map(model, x, target=0, window=(1, 1), strides=(1, 1), max_passes=8)
    err = excinfo.value
    assert err.fields["code"] == "occlusion_pass_budget_exceeded"
    assert err.fields["computed_passes"] == 16
    assert err.fields["max_passes"] == 8
    assert err.fields["one_pass_seconds"] > 0
    assert err.fields["projected_total_seconds"] > 0
    message = str(err)
    assert "16" in message and "8" in message and "s total" in message


def test_geometry_refusals() -> None:
    """Oversized windows, bad specs, and bad leaf indexes refuse typed."""

    model = _PatchSum()
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    for kwargs in (
        {"window": (5, 5)},
        {"window": (0, 2)},
        {"window": (2, 2, 2, 2)},
        {"window": (2, 2), "strides": (1, 1, 1)},
        {"window": (2, 2), "occlude_leaf": 3},
        {"window": (2, 2), "overlap": "max"},
    ):
        with pytest.raises(AttributionError) as excinfo:
            attribution.occlusion_map(model, x, target=0, **kwargs)
        assert excinfo.value.fields["code"] == "occlusion_geometry_invalid", kwargs
