"""Deletion and insertion curves with AOPC (``tl.attribution.deletion_curve``).

A tiny linear model's curves equal the closed form exactly; on a small CNN
every recorded step replays through the intervention engine to its score; on
a model whose true ranking is known, the attribution's AOPC beats random
rankings; ``grad_cam`` results (a CNN and a tiny ViT) run through both curves.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution


class SmallCnn(nn.Module):
    """Two conv stages, global pooling, linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(6, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))


class SquareSum(nn.Module):
    """``f(x) = sum_i w_i x_i ** 2``: removing feature i lowers f by exactly w_i x_i ** 2."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        return (self.weight * x * x).sum(dim=1, keepdim=True)


class TwoInputs(nn.Module):
    """Adds two linear readouts: ``f(x, y) = a.x + b.y``."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(3, 1, bias=False, dtype=torch.float64)
        self.right = nn.Linear(2, 1, bias=False, dtype=torch.float64)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.left(x) + self.right(y)


def _linear_model() -> tuple[nn.Linear, Tensor]:
    """Integer-valued float64 weights and inputs: every partial sum is exact."""

    model = nn.Linear(6, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[3.0, -1.0, 4.0, 1.0, -5.0, 9.0], [2.0, 6.0, -5.0, 3.0, 5.0, -8.0]])
        )
        model.bias.copy_(torch.tensor([7.0, -2.0]))
    x = torch.tensor([[2.0, 3.0, -1.0, 4.0, 1.0, 2.0]], dtype=torch.float64)
    return model, x


@pytest.mark.parametrize("kind", ["deletion", "insertion"])
def test_curve_equals_closed_form_on_linear_model(kind: str) -> None:
    """One feature per step on ``w.x + b``: scores, area and AOPC equal the closed form."""

    model, x = _linear_model()
    result = attribution.input_x_grad(model, x, target=0)
    curve_fn = attribution.deletion_curve if kind == "deletion" else attribution.insertion_curve
    curve = curve_fn(model, x, target=0, attribution=result, n_steps=6)

    contributions = (model.weight[0] * x[0]).detach()
    order = torch.sort(contributions, descending=True, stable=True).indices
    bias = model.bias[0].detach()
    full = contributions.sum() + bias
    expected = [full] if kind == "deletion" else [bias]
    for index in order:
        step = contributions[index]
        expected.append(expected[-1] - step if kind == "deletion" else expected[-1] + step)
    expected_scores = torch.stack(expected).reshape(7, 1)

    assert torch.equal(curve.scores, expected_scores)
    assert torch.equal(curve.fractions, torch.arange(7, dtype=torch.float64) / 6)
    sign = 1.0 if kind == "deletion" else -1.0
    expected_aopc = float((sign * (expected_scores[0] - expected_scores)).mean())
    expected_area = float(
        sum((expected_scores[k, 0] + expected_scores[k + 1, 0]) / 2 for k in range(6)) / 6
    )
    assert curve.aopc == expected_aopc
    assert curve.area == pytest.approx(expected_area, abs=1e-12)
    assert curve.settings["features"] == "elements"
    assert curve.extra["n_features"] == 6
    assert [step.n_features for step in curve.steps] == list(range(7))


@pytest.mark.parametrize("kind", ["deletion", "insertion"])
def test_curve_steps_replay_to_their_scores_on_small_cnn(kind: str) -> None:
    """Every recorded step, replayed through ``tl.trace(..., intervene=)``, gives its score."""

    torch.manual_seed(0)
    model = SmallCnn().eval()
    x = torch.randn(1, 3, 8, 8)
    result = attribution.saliency(model, x, target=1)
    curve_fn = attribution.deletion_curve if kind == "deletion" else attribution.insertion_curve
    curve = curve_fn(model, x, target=1, attribution=result, n_steps=8, baseline=0.5)

    assert curve.settings["features"] == "pixels"
    assert len(curve.steps) == 9
    for step in curve.steps:
        replayed = tl.trace(model, x, intervene=step.intervention)
        output = replayed[replayed.output_layers[0]].out
        assert torch.allclose(output[:, 1].to(torch.float64), step.score, atol=1e-6)
        direct = model(step.inputs)[:, 1].detach().to(torch.float64)
        assert torch.allclose(direct, step.score, atol=1e-6)
        held = step.baseline_mask
        assert torch.all(step.inputs[held] == 0.5)
        assert torch.equal(step.inputs[~held], x[~held])


def test_attribution_aopc_beats_random_ranking_when_true_ranking_is_known() -> None:
    """On ``sum_i w_i x_i ** 2`` the input-times-gradient ranking is the true one."""

    torch.manual_seed(0)
    weight = torch.rand(16) + 0.1
    model = SquareSum(weight)
    x = torch.randn(1, 16)
    result = attribution.input_x_grad(model, x, target=0)
    curve = attribution.deletion_curve(model, x, target=0, attribution=result, n_steps=16)
    for seed in range(10):
        generator = torch.Generator().manual_seed(seed)
        random_map = torch.rand(1, 16, generator=generator)
        random_curve = attribution.deletion_curve(
            model, x, target=0, attribution=random_map, n_steps=16
        )
        assert curve.aopc > random_curve.aopc
    insertion = attribution.insertion_curve(model, x, target=0, attribution=result, n_steps=16)
    random_insertion = attribution.insertion_curve(
        model, x, target=0, attribution=torch.rand(1, 16), n_steps=16
    )
    assert insertion.aopc > random_insertion.aopc


def _tiny_vit() -> tuple[nn.Module, Any]:
    transformers = pytest.importorskip("transformers")
    config = transformers.ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=5,
    )
    model = transformers.ViTForImageClassification(config).eval()

    def target(output: Any) -> Tensor:
        return output.logits[:, 1].sum()

    return model, target


@pytest.mark.parametrize("family", ["cnn", "vit"])
def test_curves_accept_grad_cam_results(family: str) -> None:
    """A ``grad_cam`` map (CNN feature map, ViT token grid) ranks input pixels."""

    torch.manual_seed(0)
    if family == "cnn":
        model: nn.Module = SmallCnn().eval()
        x = torch.randn(1, 3, 8, 8)
        target: Any = 2
        cam = attribution.grad_cam(model, x, target=target, layer="features", overlay=False)
    else:
        model, target = _tiny_vit()
        x = torch.randn(1, 3, 32, 32)
        cam = attribution.grad_cam(model, x, target=target, layer="vit.layers.0", overlay=False)
        assert cam.extra["grid_route"] == "derived"

    for curve_fn in (attribution.deletion_curve, attribution.insertion_curve):
        curve = curve_fn(model, x, target=target, attribution=cam, n_steps=4, features="patches")
        assert curve.scores.shape == (5, 1)
        assert curve.extra["attribution_method"] == "grad_cam"
        start = curve.steps[0].inputs
        end = curve.steps[-1].inputs
        unperturbed, baseline = (start, end) if curve.metric == "deletion" else (end, start)
        assert torch.equal(unperturbed, x)
        assert torch.equal(baseline, torch.zeros_like(x))


def test_patch_and_token_grouping_counts_features() -> None:
    """Patches tile the grid (edge patches smaller); tokens group the last axis."""

    torch.manual_seed(0)
    model = SmallCnn().eval()
    x = torch.randn(2, 3, 10, 8)
    result = attribution.saliency(model, x, target=0)
    curve = attribution.deletion_curve(
        model, x, target=0, attribution=result, features="patches", patch_size=4, n_steps=6
    )
    assert curve.extra["n_features"] == 3 * 2
    assert curve.settings["patch_size"] == (4, 4)
    assert curve.scores.shape == (7, 2)
    assert len(curve.extra["aopc_per_example"]) == 2
    patch_mask = curve.steps[1].baseline_mask
    assert int(patch_mask[0, 0].sum()) in (16, 8)
    assert torch.equal(patch_mask[:, 0], patch_mask[:, 2])

    tokens = nn.Sequential(nn.Flatten(), nn.Linear(5 * 4, 2))
    sequence = torch.randn(1, 5, 4)
    token_result = attribution.saliency(tokens, sequence, target=1)
    token_curve = attribution.insertion_curve(
        tokens, sequence, target=1, attribution=token_result, n_steps=5
    )
    assert token_curve.settings["features"] == "tokens"
    assert token_curve.extra["n_features"] == 5
    revealed = ~token_curve.steps[2].baseline_mask
    assert torch.equal(revealed.all(dim=-1), revealed.any(dim=-1))
    assert int(revealed.all(dim=-1).sum()) == 2


def test_curve_over_two_inputs_ranks_features_jointly() -> None:
    """Two floating-point inputs: one ranking over both, one rule per input."""

    model = TwoInputs()
    with torch.no_grad():
        model.left.weight.copy_(torch.tensor([[1.0, 4.0, -2.0]], dtype=torch.float64))
        model.right.weight.copy_(torch.tensor([[3.0, 0.5]], dtype=torch.float64))
    x = torch.ones(1, 3, dtype=torch.float64)
    y = torch.ones(1, 2, dtype=torch.float64)
    result = attribution.input_x_grad(model, (x, y), target=0)
    curve = attribution.deletion_curve(model, (x, y), target=0, attribution=result, n_steps=5)
    # contributions 1, 4, -2 | 3, 0.5: removed in the order 4, 3, 1, 0.5, -2
    expected = torch.tensor([6.5, 2.5, -0.5, -1.5, -2.0, 0.0], dtype=torch.float64)
    assert torch.equal(curve.scores[:, 0], expected)
    assert len(curve.steps[2].intervention.rules) == 2
    step_x, step_y = curve.steps[2].inputs
    assert torch.equal(step_x, torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64))
    assert torch.equal(step_y, torch.tensor([[0.0, 1.0]], dtype=torch.float64))


def _remedy(message: str) -> str:
    found = re.search(r"Remedy: .*?(tl\.attribution\.\w+\(.*?\))\. See", message)
    assert found is not None, message
    return found.group(1)


def test_curve_refuses_misaligned_attribution_and_its_remedy_runs() -> None:
    """A layer-space map refuses with a code; the input-space spelling runs."""

    torch.manual_seed(0)
    model = SmallCnn().eval()
    x = torch.randn(1, 3, 8, 8)
    layer_result = attribution.layer_attribution(model, x, target=0, layer="features")
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.deletion_curve(model, x, target=0, attribution=layer_result)
    assert excinfo.value.fields.get("code") == "attribution_eval_map_misaligned"
    message = str(excinfo.value)
    assert "(1, 3, 8, 8)" in message
    fixed = eval(_remedy(message), {"tl": tl, "model": model, "x": x})  # noqa: S307
    assert isinstance(fixed, attribution.AttributionResult)
    assert attribution.deletion_curve(model, x, target=0, attribution=fixed).aopc == pytest.approx(
        attribution.deletion_curve(model, x, target=0, attribution=fixed).aopc
    )


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("n_steps", 0),
        ("features", "voxels"),
        ("baseline", "noise"),
        ("patch_size", (0, 2)),
    ],
)
def test_curve_refuses_bad_settings_and_remedies_run(setting: str, value: Any) -> None:
    """Each bad setting refuses ``attribution_eval_setting_invalid``; its remedy runs."""

    torch.manual_seed(0)
    model = SmallCnn().eval()
    x = torch.randn(1, 3, 8, 8)
    result = attribution.saliency(model, x, target=0)
    kwargs: dict[str, Any] = {setting: value}
    if setting == "patch_size":
        kwargs["features"] = "patches"
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.deletion_curve(model, x, target=0, attribution=result, **kwargs)
    assert excinfo.value.fields.get("code") == "attribution_eval_setting_invalid"
    fixed = eval(  # noqa: S307
        _remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": x, "result": result}
    )
    assert isinstance(fixed, attribution.CurveResult)


def test_curve_refuses_site_target() -> None:
    """Curves score the model output, so a ``SiteTarget`` refuses with a code."""

    torch.manual_seed(0)
    model = SmallCnn().eval()
    x = torch.randn(1, 3, 8, 8)
    result = attribution.saliency(model, x, target=0)
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.insertion_curve(
            model, x, target=attribution.SiteTarget("head", unit=0), attribution=result
        )
    assert excinfo.value.fields.get("code") == "attribution_site_target_unsupported"
