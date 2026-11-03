"""ROAD, remove and debias (``tl.attribution.road_curve``).

On a tiny linear model with noise off, every step's score equals the closed form computed
from an independent dense solve of the linear imputation; with noise on, the draws are seeded
and every step's recorded edit replays through ``trace.fork().do(...)`` to its score; bad
settings and non-image inputs refuse with a remedy that runs.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution_kit

# Stated before the first run: the noiseless curve matches the dense-solve closed form within
# 1e-10 (float64; the imputation is solved by conjugate gradients), and a replayed step matches
# its score within 1e-6 (float32 CNN).
CLOSED_FORM_ATOL = 1e-10
REPLAY_ATOL = 1e-6
_WEIGHTS = {(-1, 0): 1 / 6, (1, 0): 1 / 6, (0, -1): 1 / 6, (0, 1): 1 / 6}
_WEIGHTS.update({(-1, -1): 1 / 12, (-1, 1): 1 / 12, (1, -1): 1 / 12, (1, 1): 1 / 12})


def _dense_imputation(image: Tensor, removed: Tensor) -> Tensor:
    """Solve the linear imputation of one ``(H, W)`` plane with explicit loops."""

    height, width = image.shape
    unknown = [(row, col) for row in range(height) for col in range(width) if removed[row, col]]
    if len(unknown) == height * width:
        return torch.zeros_like(image)
    index = {position: number for number, position in enumerate(unknown)}
    system = torch.zeros(len(unknown), len(unknown), dtype=torch.float64)
    right = torch.zeros(len(unknown), dtype=torch.float64)
    for (row, col), number in index.items():
        for (d_row, d_col), weight in _WEIGHTS.items():
            other = (row + d_row, col + d_col)
            if not (0 <= other[0] < height and 0 <= other[1] < width):
                continue
            system[number, number] += weight
            if other in index:
                system[number, index[other]] -= weight
            else:
                right[number] += weight * image[other]
    solved = torch.linalg.solve(system, right) if unknown else right
    result = image.clone()
    for (row, col), number in index.items():
        result[row, col] = solved[number]
    return result


@pytest.mark.parametrize("ranking", ["most_relevant_first", "least_relevant_first"])
def test_noiseless_road_curve_equals_the_closed_form_on_a_linear_model(ranking: str) -> None:
    """With ``noise_std=0`` each step scores ``w . impute(x, mask_k) + b`` exactly."""

    generator = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 1)).double()
    with torch.no_grad():
        model[1].weight.copy_(torch.randint(-3, 4, (1, 16), generator=generator))
        model[1].bias.fill_(0.5)
    x = torch.randint(-4, 5, (1, 1, 4, 4), generator=generator).double()
    scores_map = torch.randperm(16, generator=generator).double().reshape(1, 1, 4, 4)
    curve = attribution_kit.road_curve(
        model, x, target=0, attribution=scores_map, ranking=ranking, num_steps=16, noise_std=0.0
    )
    order = torch.argsort(scores_map.reshape(-1), descending=ranking == "most_relevant_first")
    assert curve.kind == "road" and curve.settings["ranking"] == ranking
    for step in curve.steps:
        removed = torch.zeros(16, dtype=torch.bool)
        removed[order[: step.num_features]] = True
        imputed = _dense_imputation(x[0, 0], removed.reshape(4, 4))
        with torch.no_grad():
            expected = model(imputed.reshape(1, 1, 4, 4))[0, 0]
        assert float(step.score[0]) == pytest.approx(float(expected), abs=CLOSED_FORM_ATOL)
        assert torch.equal(step.build_baseline_mask, removed.reshape(1, 1, 4, 4))


class SmallCnn(nn.Module):
    """A tiny convolutional classifier on 6x6 RGB images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4 * 6 * 6, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Convolution, ReLU, flatten, linear."""

        return self.head(torch.relu(self.conv(x)).flatten(1))


def test_noisy_steps_are_seeded_and_replay_to_their_scores() -> None:
    """Default noise is seeded per step; each step's edit replays through ``fork().do``."""

    torch.manual_seed(0)
    model = SmallCnn().eval()
    x = torch.randn(1, 3, 6, 6)
    saliency = attribution_kit.input_x_grad(model, x, target=2)
    curve = attribution_kit.road_curve(model, x, target=2, attribution=saliency, num_steps=6)
    again = attribution_kit.road_curve(model, x, target=2, attribution=saliency, num_steps=6)
    other = attribution_kit.road_curve(
        model, x, target=2, attribution=saliency, num_steps=6, seed=1
    )
    assert torch.equal(curve.scores, again.scores)
    assert not torch.equal(curve.scores[1:], other.scores[1:])
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    for step in curve.steps:
        fork = trace.fork()
        try:
            for selection, edit in step.build_interventions:
                fork.do(selection, edit)
            replayed = fork[fork.output_layers[0]].out[0, 2].detach()
        finally:
            fork.cleanup()
        assert float(replayed) == pytest.approx(float(step.score[0]), abs=REPLAY_ATOL)
        kept = ~step.build_baseline_mask
        assert torch.equal(step.build_inputs[kept], x[kept])


def _remedy(message: str) -> str:
    found = re.search(r"(tl\.attribution\.(?:road|deletion)_curve\(.*\))\. See", message)
    assert found is not None, message
    return found.group(1)


@pytest.mark.parametrize(
    ("settings", "image"),
    [({"ranking": "random"}, True), ({"noise_std": -1.0}, True), ({}, False)],
    ids=["ranking-random", "noise-negative", "not-an-image"],
)
def test_road_refuses_bad_settings_and_the_remedy_runs(
    settings: dict[str, Any], image: bool
) -> None:
    """An unknown ranking, a negative noise or a non-image input refuse; the remedy runs."""

    torch.manual_seed(0)
    model: nn.Module = SmallCnn().eval() if image else nn.Linear(5, 3)
    x = torch.randn(1, 3, 6, 6) if image else torch.randn(2, 5)
    attribution = attribution_kit.input_x_grad(model, x, target=0)
    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution_kit.road_curve(model, x, target=0, attribution=attribution, **settings)
    assert excinfo.value.fields.get("code") == "attribution_eval_setting_invalid"
    scope = {"tl": tl, "model": model, "x": x, "attribution": attribution}
    fixed = eval(_remedy(str(excinfo.value)), scope)  # noqa: S307
    assert isinstance(fixed, attribution_kit.PerturbationCurve)
