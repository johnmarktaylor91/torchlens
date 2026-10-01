"""F06 B1/B3/B4: binder contract, noise tunnel, GradientShap.

Attrib memo D1-D7 and D10: the two wrapping routes and their typed refusals,
NT semantics (zero-noise degenerate exactness, seeded determinism, stored
noise banks, aggregation vocabulary, repeated-reference identity, integer
leaves unchanged, composition rulings), SmoothGrad-as-thin-alias equivalence,
and the GradientShap estimator pinned by a stored-draw oracle.
"""

from __future__ import annotations

import functools

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution import AttributionError

pytestmark = pytest.mark.smoke


class _TwoLeafModel(nn.Module):
    """Model with two float inputs and one integer mask input."""

    def __init__(self) -> None:
        """Build fixed weights."""

        super().__init__()
        self.head_a = nn.Linear(3, 2, dtype=torch.float64)
        self.head_b = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            self.head_a.weight.copy_(torch.arange(6, dtype=torch.float64).reshape(2, 3) / 4.0)
            self.head_a.bias.zero_()
            self.head_b.weight.copy_(torch.ones(2, 3, dtype=torch.float64) * 0.5)
            self.head_b.bias.zero_()

    def forward(self, a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
        """Combine the two float paths gated by the integer mask."""

        assert mask.dtype == torch.long, "integer mask must pass through unnoised"
        return self.head_a(a) + self.head_b(b) * mask.to(a.dtype).unsqueeze(-1)


class _LinearOne(nn.Module):
    """One-layer linear model for analytic rows."""

    def __init__(self) -> None:
        """Build fixed bias-free weights."""

        super().__init__()
        self.head = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=torch.float64)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Linear forward."""

        return self.head(x)


def _simple_inputs() -> tuple[Tensor, Tensor, Tensor]:
    """Return deterministic (a, b, mask) inputs for _TwoLeafModel."""

    a = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
    b = torch.tensor([[0.1, 0.4, -0.6]], dtype=torch.float64)
    mask = torch.ones(1, dtype=torch.long)
    return a, b, mask


def test_binder_route_refusals() -> None:
    """D1: exactly one route; method_kwargs only with method=; sugar needs model+target."""

    model = _LinearOne()
    x = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(AttributionError) as neither:
        attribution.noise_tunnel(x, stdevs=0.1)
    assert neither.value.fields["code"] == "attribution_method_binding_invalid"
    with pytest.raises(AttributionError) as both:
        attribution.noise_tunnel(
            x,
            stdevs=0.1,
            attribute=lambda inputs, kwargs: None,
            method=attribution.saliency,
            model=model,
            target=0,
        )
    assert both.value.fields["code"] == "attribution_method_binding_invalid"
    with pytest.raises(AttributionError) as closed_plus:
        attribution.noise_tunnel(
            x,
            stdevs=0.1,
            attribute=lambda inputs, kwargs: None,
            method_kwargs={"n_steps": 4},
        )
    assert closed_plus.value.fields["code"] == "attribution_method_binding_invalid"
    with pytest.raises(AttributionError) as missing_target:
        attribution.noise_tunnel(x, stdevs=0.1, method=attribution.saliency, model=model)
    assert missing_target.value.fields["code"] == "attribution_method_binding_invalid"


def test_noise_tunnel_zero_noise_is_degenerate_exact() -> None:
    """D4: with stdevs=0 every sample equals the direct child call exactly."""

    model = _LinearOne()
    x = torch.tensor([[0.4, -0.1, 0.7]], dtype=torch.float64)
    direct = attribution.saliency(model, x, target=1)
    tunneled = attribution.noise_tunnel(
        x, method=attribution.saliency, model=model, target=1, n_samples=3, stdevs=0.0
    )
    torch.testing.assert_close(tunneled.values, direct.values, rtol=0, atol=0)
    assert tunneled.method == "noise_tunnel"
    assert tunneled.extra["child_method"] == "saliency"
    assert tunneled.extra["stdevs_resolved"] == [0.0]
    assert tunneled.extra["planned_logical_calls"] == 3
    assert tunneled.extra["completed_logical_calls"] == 3


def test_noise_tunnel_seeded_determinism_and_aggregations() -> None:
    """Seeded runs reproduce; aggregations match manual math on one stored bank."""

    model = _LinearOne()
    x = torch.tensor([[0.2, 0.6, -0.3]], dtype=torch.float64)
    first = attribution.noise_tunnel(
        x, method=attribution.saliency, model=model, target=0, n_samples=4, stdevs=0.2, seed=7
    )
    second = attribution.noise_tunnel(
        x, method=attribution.saliency, model=model, target=0, n_samples=4, stdevs=0.2, seed=7
    )
    torch.testing.assert_close(first.values, second.values, rtol=0, atol=0)

    # One stored bank feeds all three aggregations; manual math is the oracle.
    bank = [[torch.randn(1, 3, dtype=torch.float64)] for _ in range(4)]
    samples = [attribution.saliency(model, x + 0.2 * bank[i][0], target=0).values for i in range(4)]
    stacked = torch.stack(samples)
    for aggregation, expected in (
        ("mean", stacked.mean(dim=0)),
        ("mean_square", (stacked.abs() ** 2).mean(dim=0)),
        ("variance", stacked.var(dim=0, unbiased=False)),
    ):
        result = attribution.noise_tunnel(
            x,
            method=attribution.saliency,
            model=model,
            target=0,
            n_samples=4,
            stdevs=0.2,
            aggregation=aggregation,
            noise_bank=bank,
        )
        torch.testing.assert_close(result.values, expected, rtol=1e-12, atol=1e-12)
        assert result.extra["noise_bank_used"] is True


def test_noise_tunnel_repeated_reference_and_integer_mask() -> None:
    """One tensor in two slots gets ONE noised object; the long mask is untouched."""

    model = _TwoLeafModel()
    a, _b, mask = _simple_inputs()

    seen: list[tuple[bool, Tensor]] = []

    def probe(inputs: tuple, input_kwargs: dict | None) -> attribution.AttributionResult:
        """Record identity/mask facts, then run input_x_grad."""

        noised_a, noised_b, noised_mask = inputs
        seen.append((noised_a is noised_b, noised_mask))
        return attribution.input_x_grad(model, inputs, input_kwargs, target=0)

    result = attribution.noise_tunnel(
        (a, a, mask), attribute=probe, n_samples=2, stdevs=0.1, seed=3
    )
    assert result.extra["child_method"] == "<closed callable>"
    for same_object, noised_mask in seen:
        assert same_object, "repeated references must share ONE noised object"
        assert noised_mask is mask or torch.equal(noised_mask, mask)
        assert noised_mask.dtype == torch.long


def test_noise_tunnel_composition_refusals() -> None:
    """D6: NT(smoothgrad) and NT(trace-bound occlusion) refuse typed."""

    model = _LinearOne()
    x = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(AttributionError) as sg:
        attribution.noise_tunnel(
            x, method=attribution.smoothgrad, model=model, target=0, stdevs=0.1
        )
    assert sg.value.fields["code"] == "noise_tunnel_composition_unsupported"
    assert "saliency" in str(sg.value)
    with pytest.raises(AttributionError) as tr:
        attribution.noise_tunnel(x, method=attribution.occlusion, model=model, target=0, stdevs=0.1)
    assert tr.value.fields["code"] == "noise_tunnel_composition_unsupported"


def test_noise_tunnel_child_failure_names_zero_based_sample() -> None:
    """D4: a child failure names its zero-based sample and returns nothing."""

    calls = {"count": 0}

    def flaky(inputs: Tensor, input_kwargs: dict | None) -> attribution.AttributionResult:
        """Fail on the second sample."""

        if calls["count"] == 1:
            raise RuntimeError("boom")
        calls["count"] += 1
        return attribution.saliency(_LinearOne(), inputs, input_kwargs, target=0)

    with pytest.raises(AttributionError) as excinfo:
        attribution.noise_tunnel(
            torch.zeros(1, 3, dtype=torch.float64),
            attribute=flaky,
            n_samples=3,
            stdevs=0.1,
            seed=0,
        )
    assert excinfo.value.fields["code"] == "sample_child_failed"
    assert excinfo.value.fields["failed_sample"] == 1
    assert "sample 1 (zero-based)" in str(excinfo.value)


def test_noise_tunnel_invariance_refusal() -> None:
    """D4: a shape-shifting child refuses with the invariance code."""

    calls = {"count": 0}

    def shifty(inputs: Tensor, input_kwargs: dict | None) -> attribution.AttributionResult:
        """Return a differently shaped values tree on the second sample."""

        del input_kwargs
        calls["count"] += 1
        shape = (1, 3) if calls["count"] == 1 else (1, 4)
        return attribution.AttributionResult(
            method="probe",
            values=torch.zeros(shape, dtype=torch.float64),
            target_repr="index=0",
            extra={},
        )

    with pytest.raises(AttributionError) as excinfo:
        attribution.noise_tunnel(
            torch.zeros(1, 3, dtype=torch.float64),
            attribute=shifty,
            n_samples=2,
            stdevs=0.1,
            seed=0,
        )
    assert excinfo.value.fields["code"] == "sample_invariance_violated"


def test_noise_tunnel_gradient_shap_substreams_disclosed() -> None:
    """D6: NT(gradient_shap) derives per-sample child seeds and discloses counts."""

    model = _LinearOne()
    x = torch.tensor([[0.5, -0.5, 0.25]], dtype=torch.float64)
    pool = torch.zeros(2, 3, dtype=torch.float64)
    result = attribution.noise_tunnel(
        x,
        method=attribution.gradient_shap,
        model=model,
        target=0,
        method_kwargs={"baselines": pool, "n_samples": 3},
        n_samples=2,
        stdevs=0.05,
        seed=11,
    )
    assert result.extra["child_method"] == "gradient_shap"
    assert result.extra["method_kwargs"]["n_samples"] == 3
    assert len(result.extra["child_seeds"]) == 2
    assert len(set(result.extra["child_seeds"])) == 2
    assert result.extra["child_n_samples"] == 3
    assert result.extra["multiplicative_cost_logical"] == 6
    # Reproducible under the same tunnel seed.
    again = attribution.noise_tunnel(
        x,
        method=attribution.gradient_shap,
        model=model,
        target=0,
        method_kwargs={"baselines": pool, "n_samples": 3},
        n_samples=2,
        stdevs=0.05,
        seed=11,
    )
    torch.testing.assert_close(result.values, again.values, rtol=0, atol=0)


def test_smoothgrad_is_the_thin_alias_with_absolute_before_mean() -> None:
    """D7: smoothgrad == seeded NT(saliency) mean, 25 samples, sigma 0.1 absolute.

    The per-sample ABSOLUTE VALUE precedes the mean because the child is
    saliency; the draw order is bit-identical, so the equality is exact.
    """

    model = _LinearOne()
    x = torch.tensor([[0.3, 0.9, -0.4]], dtype=torch.float64)
    legacy = attribution.smoothgrad(model, x, target=0, n_samples=25, noise_level=0.1, seed=123)
    tunneled = attribution.noise_tunnel(
        x,
        method=attribution.saliency,
        model=model,
        target=0,
        n_samples=25,
        stdevs=0.1,
        seed=123,
        aggregation="mean",
    )
    torch.testing.assert_close(legacy.values, tunneled.values, rtol=0, atol=0)
    assert legacy.method == "smoothgrad"
    assert legacy.extra == {"n_samples": 25, "noise_level": 0.1, "seed": 123}


def test_gradient_shap_stored_draw_oracle_pins_the_estimator() -> None:
    """D10: with stored draws the estimator equals the hand-rolled convention.

    grad at ``baseline + alpha * (x - baseline)`` times ``(x - baseline)``,
    zero noise, mean over samples -- computed independently here from the
    pinned captum ``InputBaselineXGradient`` convention.
    """

    model = _LinearOne()
    x = torch.tensor([[0.8, -0.2, 0.4]], dtype=torch.float64)
    pool = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.25, 0.1, 0.3]], dtype=torch.float64)
    alphas = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    indices = torch.tensor([[2], [0]], dtype=torch.long)
    result = attribution.gradient_shap(
        model,
        x,
        target=1,
        baselines=pool,
        n_samples=2,
        stdevs=0.0,
        draw_bank={"alphas": alphas, "pool_indices": indices},
        store_draws=True,
    )
    weight_row = model.head.weight[1]
    expected = torch.zeros_like(x)
    for sample in range(2):
        baseline = pool[indices[sample, 0]]
        # Linear model: gradient at any point is the weight row.
        expected = expected + weight_row * (x - baseline)
    expected = expected / 2
    torch.testing.assert_close(result.values, expected, rtol=1e-12, atol=1e-12)
    assert result.extra["estimator"].startswith("captum InputBaselineXGradient")
    diagnostics = result.extra["monte_carlo_diagnostics"]
    assert diagnostics["labeled"] == "diagnostics, never a completeness guarantee"
    # Linear model + exact gradients: residual of means is exactly zero.
    assert diagnostics["residual_of_means"] == pytest.approx(0.0, abs=1e-9)
    torch.testing.assert_close(result.extra["draws"]["alphas"], alphas)


def test_gradient_shap_converges_to_ig_single_baseline_zero_noise() -> None:
    """Large-sample GradientShap at ONE baseline, zero noise, approaches IG."""

    model = _SmoothTanh()
    x = torch.tensor([[0.6, -0.4, 0.2]], dtype=torch.float64)
    baseline = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    ig = attribution.integrated_gradients(model, x, target=0, n_steps=512, baseline=baseline)
    gs = attribution.gradient_shap(model, x, target=0, baselines=baseline, n_samples=512, seed=5)
    torch.testing.assert_close(gs.values, ig.values, rtol=0.05, atol=1e-4)


class _SmoothTanh(nn.Module):
    """Smooth nonlinearity for the IG-convergence row."""

    def __init__(self) -> None:
        """Build fixed weights."""

        super().__init__()
        self.hidden = nn.Linear(3, 4, dtype=torch.float64)
        self.head = nn.Linear(4, 2, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.arange(12, dtype=torch.float64).reshape(4, 3) / 9.0 - 0.5
            )
            self.hidden.bias.zero_()
            self.head.weight.copy_(torch.arange(8, dtype=torch.float64).reshape(2, 4) / 6.0)
            self.head.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Tanh MLP forward."""

        return self.head(torch.tanh(self.hidden(x)))


def test_gradient_shap_pool_refusals() -> None:
    """D10: missing pool, misaligned pool, inconsistent pool sizes refuse."""

    model = _LinearOne()
    x = torch.zeros(1, 3, dtype=torch.float64)
    with pytest.raises(AttributionError) as missing:
        attribution.gradient_shap(model, x, target=0, baselines=None)
    assert missing.value.fields["code"] == "gradient_shap_pool_invalid"
    with pytest.raises(AttributionError) as misaligned:
        attribution.gradient_shap(
            model, x, target=0, baselines=torch.zeros(2, 4, dtype=torch.float64)
        )
    assert misaligned.value.fields["code"] == "gradient_shap_pool_invalid"
    with pytest.raises(AttributionError) as bad_bank:
        attribution.gradient_shap(
            model,
            x,
            target=0,
            baselines=torch.zeros(2, 3, dtype=torch.float64),
            n_samples=2,
            draw_bank={"alphas": torch.zeros(2, 1), "pool_indices": torch.zeros(1, 1).long()},
        )
    assert bad_bank.value.fields["code"] == "gradient_shap_draw_bank_invalid"


def test_closed_callable_route_matches_sugar_route_disclosure() -> None:
    """D3: primitive-route users can achieve sugar-route provenance richness."""

    model = _LinearOne()
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    closed = functools.partial(attribution.saliency, model, target=0)

    def adapter(inputs: Tensor, input_kwargs: dict | None) -> attribution.AttributionResult:
        """Adapt partial(...) to the closed (inputs, input_kwargs) shape."""

        return closed(inputs, input_kwargs)

    via_attribute = attribution.noise_tunnel(x, attribute=adapter, n_samples=2, stdevs=0.1, seed=9)
    via_method = attribution.noise_tunnel(
        x, method=attribution.saliency, model=model, target=0, n_samples=2, stdevs=0.1, seed=9
    )
    torch.testing.assert_close(via_attribute.values, via_method.values, rtol=0, atol=0)
    # The sugar route adds model identity because it SAW the model; the
    # primitive route deliberately has none (a closed callable hides it).
    assert via_method.extra["model_identity"] == "_LinearOne"
    assert "model_identity" not in via_attribute.extra
