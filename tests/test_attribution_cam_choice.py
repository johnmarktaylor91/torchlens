"""Automatic Grad-CAM layer choice from the trace (``grad_cam`` without ``layer``).

The chosen layer equals a name stated here, and the automatic call's heatmap
equals the explicit-layer call bit for bit.
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


class TwoTower(nn.Module):
    """Two independent conv branches joined only after pooling: no map carries everything."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Conv2d(3, 2, 3, padding=1)
        self.right = nn.Conv2d(3, 2, 5, padding=2)
        self.head = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        left = self.left(x).mean(dim=(2, 3))
        right = self.right(x).mean(dim=(2, 3))
        return self.head(torch.cat([left, right], dim=1))


def _assert_auto_matches_explicit(model: nn.Module, inputs: Tensor, target: Any, name: str) -> None:
    auto = attribution.grad_cam(model, inputs, target=target, overlay=False)
    explicit = attribution.grad_cam(model, inputs, target=target, layer=name, overlay=False)

    assert auto.extra["layer"] == name
    assert auto.extra["layer_source"] == "trace"
    assert "every input-to-output path" in auto.extra["layer_reason"]
    assert explicit.extra["layer_source"] == "caller"
    assert explicit.extra["layer_reason"] is None
    assert torch.equal(auto.values, explicit.values)


def test_grad_cam_chooses_last_feature_stage_of_small_cnn() -> None:
    """The outermost module returning the last spatial map is chosen."""

    torch.manual_seed(0)
    _assert_auto_matches_explicit(SmallCnn(), torch.randn(1, 3, 8, 8), 1, "features")


def test_grad_cam_chooses_layer4_of_resnet18() -> None:
    """torchvision resnet18 (random init): ``layer4``, the last map before pooling."""

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None)
    _assert_auto_matches_explicit(model, torch.randn(1, 3, 64, 64), 3, "layer4")


def test_grad_cam_chooses_residual_stream_into_last_vit_block() -> None:
    """Tiny ViT: the last block's output feeds the head only through the class token.

    Its patch tokens carry a zero gradient, so the chooser steps back to the
    residual stream entering the last block.
    """

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
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

    def target(output: Any) -> Tensor:
        return output.logits[:, 1].sum()

    _assert_auto_matches_explicit(model, torch.randn(1, 3, 32, 32), target, "vit.encoder.layer.0")
    auto = attribution.grad_cam(model, torch.randn(1, 3, 32, 32), target=target, overlay=False)
    assert "'vit.layernorm'" in auto.extra["layer_reason"]
    assert auto.extra["prefix_tokens"] == 1


def test_grad_cam_refuses_ambiguous_choice_and_its_remedy_runs() -> None:
    """Two branches with no shared map: coded refusal, runnable remedy."""

    torch.manual_seed(0)
    model = TwoTower()
    inputs = torch.randn(1, 3, 8, 8)

    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.grad_cam(model, inputs, target=0)

    assert excinfo.value.fields.get("code") == "grad_cam_layer_choice_ambiguous"
    message = str(excinfo.value)
    remedy = re.search(r"e\.g\. (tl\.attribution\.grad_cam\(.*?\));", message)
    assert remedy is not None, message
    result = eval(remedy.group(1), {"tl": tl, "model": model, "inputs": inputs})  # noqa: S307
    assert result.extra["layer"] in {"left", "right"}


def test_grad_cam_refuses_when_no_layer_is_a_map() -> None:
    """A model with no spatial map and no token grid has nothing to choose."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.grad_cam(model, torch.randn(1, 4), target=0)
    assert excinfo.value.fields.get("code") == "grad_cam_layer_unchosen"
