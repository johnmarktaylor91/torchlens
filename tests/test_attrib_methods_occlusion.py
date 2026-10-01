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
from torchlens.attribution import AttributionError
from torchlens.attribution._occlusion import _axis_origins, _window_slices

pytestmark = pytest.mark.smoke


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


def test_axis_origins_clipped_edge_full_coverage() -> None:
    """The final window clips at the boundary so every element is covered."""

    assert _axis_origins(10, 4, 4) == [0, 4, 8]
    assert _axis_origins(10, 4, 3) == [0, 3, 6]
    assert _axis_origins(4, 4, 2) == [0]
    assert _axis_origins(5, 4, 4) == [0, 4]


def test_window_slices_cover_everything() -> None:
    """Union of window regions covers the full swept extent."""

    slices, per_axis = _window_slices((1, 2, 6, 6), (3, 3), (3, 3))
    assert per_axis == [2, 2]
    covered = torch.zeros(2, 6, 6)
    for region in slices:
        covered[region] += 1
    assert bool((covered > 0).all())


def test_full_window_self_oracle() -> None:
    """One full-size window equals the direct two-forward difference."""

    model = _PatchSum()
    x = torch.rand(1, 1, 4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    result = attribution.occlusion_map(model, x, target=0, window=(4, 4))
    with torch.no_grad():
        expected = model(x)[..., 0] - model(torch.zeros_like(x))[..., 0]
    assert result.extra["perturbation_count"] == 1
    # Every element is covered exactly once by the one full window.
    torch.testing.assert_close(
        result.values, expected.reshape(1, 1, 1, 1).expand_as(x), rtol=1e-12, atol=1e-12
    )


def test_map_values_match_analytic_deltas_average_and_sum() -> None:
    """Non-overlapping windows: each element's delta is its window's weight mass."""

    model = _PatchSum()
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    result = attribution.occlusion_map(model, x, target=0, window=(2, 2), strides=(2, 2))
    weight = model.weight
    for row0 in (0, 2):
        for col0 in (0, 2):
            block_mass = weight[0, row0 : row0 + 2, col0 : col0 + 2].sum()
            torch.testing.assert_close(
                result.values[0, 0, row0 : row0 + 2, col0 : col0 + 2],
                block_mass.expand(2, 2),
                rtol=1e-12,
                atol=1e-12,
            )
    # Overlapping windows: 'sum' accumulates, 'average' divides by coverage.
    summed = attribution.occlusion_map(
        model, x, target=0, window=(2, 2), strides=(1, 1), overlap="sum"
    )
    averaged = attribution.occlusion_map(
        model, x, target=0, window=(2, 2), strides=(1, 1), overlap="average"
    )
    assert summed.extra["perturbation_count"] == 9
    # The corner element is covered by exactly one 2x2 window at stride 1.
    torch.testing.assert_close(summed.values[0, 0, 0, 0], averaged.values[0, 0, 0, 0])
    # The center element (1,1) is covered by four windows.
    torch.testing.assert_close(
        summed.values[0, 0, 1, 1], averaged.values[0, 0, 1, 1] * 4, rtol=1e-12, atol=1e-12
    )


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


def test_only_selected_leaf_is_occluded() -> None:
    """Composition row: one leaf swept, all other inputs held fixed."""

    class _TwoInput(nn.Module):
        def forward(self, image: Tensor, bias: Tensor) -> Tensor:
            score = image.sum(dim=(1, 2, 3)) + bias.sum(dim=-1)
            return torch.stack([score, -score], dim=-1)

    image = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    bias = torch.full((1, 3), 5.0, dtype=torch.float64)
    result = attribution.occlusion_map(
        _TwoInput(), (image, bias), target=0, window=(2, 2), occlude_leaf=0
    )
    # bias is held fixed: the full-window delta equals the image mass alone.
    torch.testing.assert_close(result.values, torch.full_like(image, 4.0), rtol=1e-12, atol=1e-12)
