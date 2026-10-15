"""The model-parameter randomization test (``tl.attribution.randomization_test``).

On a small CNN, saliency similarity falls as layers randomize; a method that
ignores the parameters shows no fall; the caller's model and random state are
untouched; ``grad_cam`` results (a CNN and a tiny ViT) run; the similarity
measures equal an independent computation.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution
from torchlens.attribution._result import AttributionWarning


class SmallCnn(nn.Module):
    """Two conv stages, global pooling, linear head, on 12x12 inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Linear(8 * 4, 3)

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        return self.head(self.pool(hidden).flatten(1))


def _input_itself(model: nn.Module, inputs: Tensor, input_kwargs: Any, *, target: Any) -> Tensor:
    """A stub explanation that ignores the parameters: the input is the map.

    It still runs the model (and discards the output) so the default layer
    discovery, which watches one method call, sees every parameterized layer.
    """

    del input_kwargs, target
    with torch.no_grad():
        model(inputs)
    return inputs.detach().clone()


@pytest.mark.parametrize("randomization", ["cascading", "independent"])
def test_saliency_similarity_falls_as_layers_randomize(randomization: str) -> None:
    """Top-down randomization: layers are found in order and similarity drops."""

    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(2, 3, 12, 12)
    result = attribution.randomization_test(
        model, x, target=1, method=attribution.saliency, randomization=randomization
    )

    assert result.layers == ("head", "conv2", "conv1")
    assert [step.layer for step in result.steps] == ["head", "conv2", "conv1"]
    spearman = result.similarity("spearman")
    assert all(value is not None and value < 0.9 for value in spearman)
    if randomization == "cascading":
        assert result.steps[-1].randomized == ("head", "conv2", "conv1")
        assert spearman[-1] < 0.5
        ssim = result.similarity("ssim")
        assert all(value is not None for value in ssim)
        assert ssim[-1] < ssim[0] or ssim[0] < 0.9
    else:
        assert all(step.randomized == (step.layer,) for step in result.steps)
    assert result.steps[0].values is not None
    assert result.steps[0].values.shape == x.shape


def test_parameter_ignoring_method_shows_no_fall() -> None:
    """A map that is the input itself is identical after every randomization."""

    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(1, 3, 12, 12)
    result = attribution.randomization_test(model, x, target=0, method=_input_itself)
    assert result.layers == ("head", "conv2", "conv1")
    assert len(result.steps) == 3
    for measure in ("spearman", "spearman_abs", "ssim"):
        values = result.similarity(measure)
        assert all(value == pytest.approx(1.0, abs=1e-12) for value in values), measure
    assert all(torch.equal(step.values, result.original_attribution) for step in result.steps)


def test_caller_model_and_random_state_untouched() -> None:
    """The test randomizes copies: caller parameters and the RNG are bit-identical."""

    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(1, 3, 12, 12)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state()
    result = attribution.randomization_test(
        model, x, target=0, method=attribution.saliency, seed=3, keep_maps=False
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(step.values is None for step in result.steps)
    again = attribution.randomization_test(
        model, x, target=0, method=attribution.saliency, seed=3, keep_maps=False
    )
    assert again.similarity("spearman") == result.similarity("spearman")


@pytest.mark.parametrize("family", ["cnn", "vit"])
def test_randomization_accepts_grad_cam_results(family: str) -> None:
    """``grad_cam`` maps (input resolution) are compared after each randomization."""

    torch.manual_seed(0)
    if family == "cnn":
        model: nn.Module = SmallCnn()
        x = torch.randn(1, 3, 12, 12)
        target: Any = 2
        kwargs: dict[str, Any] = {"layer": "conv2", "overlay": False}
        layers: Any = None
    else:
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
        model = transformers.ViTForImageClassification(config)
        x = torch.randn(1, 3, 32, 32)

        def target(output: Any) -> Tensor:
            return output.logits[:, 1].sum()

        kwargs = {"layer": "vit.layers.0", "overlay": False}
        linears = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        layers = list(reversed(linears))[:3]

    result = attribution.randomization_test(
        model,
        x,
        target=target,
        method=attribution.grad_cam,
        method_kwargs=kwargs,
        layers=layers,
    )
    assert result.original_attribution.shape == (1, 1, *x.shape[-2:])
    assert len(result.steps) == len(result.layers) >= 3
    assert all(value is not None for value in result.similarity("ssim"))
    assert all(value is not None for value in result.similarity("spearman_abs"))


def _spearman_by_hand(first: list[float], second: list[float]) -> float:
    """Average ranks and Pearson correlation in plain Python."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start
            while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
                end += 1
            for position in range(start, end + 1):
                out[order[position]] = (start + end) / 2 + 1
            start = end + 1
        return out

    left, right = ranks(first), ranks(second)
    mean_left, mean_right = sum(left) / len(left), sum(right) / len(right)
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    return float(cov / (var_left * var_right) ** 0.5)


def _ssim_by_hand(first: Tensor, second: Tensor) -> float:
    """SSIM with a 7x7 uniform window over every valid window, in plain loops."""

    values = []
    height, width = first.shape
    n = 49
    for row in range(height - 6):
        for col in range(width - 6):
            a = first[row : row + 7, col : col + 7].reshape(-1)
            b = second[row : row + 7, col : col + 7].reshape(-1)
            mean_a, mean_b = float(a.mean()), float(b.mean())
            var_a = float(((a - mean_a) ** 2).sum()) / (n - 1)
            var_b = float(((b - mean_b) ** 2).sum()) / (n - 1)
            cov = float(((a - mean_a) * (b - mean_b)).sum()) / (n - 1)
            c1, c2 = 0.01**2, 0.03**2
            values.append(
                (2 * mean_a * mean_b + c1)
                * (2 * cov + c2)
                / ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2))
            )
    return sum(values) / len(values)


def test_similarity_measures_match_independent_computation() -> None:
    """Spearman (with ties) and SSIM equal plain-Python computations."""

    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(1, 3, 12, 12)
    result = attribution.randomization_test(
        model, x, target=0, method=attribution.saliency, layers=["head"]
    )
    original = result.original_attribution[0]
    randomized = result.steps[0].values[0]
    tied_original = torch.round(original / original.abs().max() * 4) / 4
    tied_randomized = torch.round(randomized / randomized.abs().max() * 4) / 4

    from torchlens.attribution._eval_maps import _spearman, _ssim, _ssim_plane

    assert _spearman(tied_original, tied_randomized) == pytest.approx(
        _spearman_by_hand(tied_original.reshape(-1).tolist(), tied_randomized.reshape(-1).tolist()),
        abs=1e-12,
    )
    assert result.steps[0].similarity["spearman"] == pytest.approx(
        _spearman_by_hand(original.reshape(-1).tolist(), randomized.reshape(-1).tolist()),
        abs=1e-12,
    )
    plane_a, plane_b = _ssim_plane(original), _ssim_plane(randomized)
    assert plane_a is not None and plane_b is not None
    assert result.steps[0].similarity["ssim"] == pytest.approx(
        _ssim_by_hand(plane_a, plane_b), abs=1e-10
    )
    assert _ssim(plane_a, plane_b) == pytest.approx(_ssim_by_hand(plane_a, plane_b), abs=1e-10)


def _remedy(message: str) -> str:
    found = re.search(r"(tl\.attribution\.randomization_test\(.*?\))\. See", message)
    assert found is not None, message
    return found.group(1)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"layers": ["missing"]}, "attribution_randomization_layer_invalid"),
        ({"layers": []}, "attribution_randomization_layer_invalid"),
        ({"randomization": "sequential"}, "attribution_eval_setting_invalid"),
        ({"method": "saliency"}, "attribution_eval_setting_invalid"),
    ],
)
def test_randomization_refuses_bad_settings_and_remedies_run(
    kwargs: dict[str, Any], code: str
) -> None:
    """Unknown layers, an unknown order or a non-callable method refuse; remedies run."""

    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(1, 3, 12, 12)
    call: dict[str, Any] = {"target": 0, "method": attribution.saliency, **kwargs}
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.randomization_test(model, x, **call)
    assert excinfo.value.fields.get("code") == code
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": x})  # noqa: S307
    assert isinstance(fixed, attribution.RandomizationResult)


def test_randomization_refuses_a_map_that_changes_shape() -> None:
    """A method whose map layout changes after randomizing refuses with a code."""

    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(1, 3, 12, 12)
    head_weight = model.head.weight.detach().clone()

    def _shifting_layout(
        model: nn.Module, inputs: Tensor, input_kwargs: Any, *, target: Any
    ) -> Tensor:
        """Return the input while the head is untouched, a cropped map once it is randomized."""

        with torch.no_grad():
            model(inputs)
        untouched = torch.equal(model.head.weight, head_weight)
        return inputs if untouched else inputs[..., 1:]

    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.randomization_test(model, x, target=0, method=_shifting_layout)
    assert excinfo.value.fields.get("code") == "attribution_randomization_map_changed"
    message = str(excinfo.value)
    assert "(1, 3, 12, 12)" in message and "(1, 3, 12, 11)" in message
    assert "method_kwargs={'layer': ...}" in message


def test_kept_maps_past_the_size_guard_warn_once_and_name_keep_maps_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safe default keeps every map; past the guard it warns once with the cheaper spelling."""

    import torchlens.attribution._randomization as randomization_module

    monkeypatch.setattr(randomization_module, "_KEEP_MAPS_WARN_BYTES", 1)
    torch.manual_seed(0)
    model = SmallCnn()
    x = torch.randn(1, 3, 12, 12)
    with pytest.warns(AttributionWarning) as record:
        result = attribution.randomization_test(model, x, target=0, method=attribution.saliency)
    coded = [w for w in record if getattr(w.message, "fields", {}).get("code")]
    assert [w.message.fields["code"] for w in coded] == ["attribution_randomization_maps_large"]
    assert "keep_maps=False" in str(coded[0].message)
    assert all(step.values is not None for step in result.steps)
    lean = attribution.randomization_test(
        model, x, target=0, method=attribution.saliency, keep_maps=False
    )
    assert lean.similarity("spearman") == result.similarity("spearman")
