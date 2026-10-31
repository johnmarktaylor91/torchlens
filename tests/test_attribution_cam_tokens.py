"""Native Grad-CAM on token-sequence layers: derived grids, refusals, caller reshape.

Each family is a tiny random-init model (no downloads). The derived grid and
prefix count are checked against values computed from the config, and the
heatmap against an independent computation: the test captures the same layer
activation and gradient with its own hook, lays the tokens out by hand, and
applies the Grad-CAM formula.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution

transformers = pytest.importorskip("transformers")

IMAGE = 32


def _named(model: nn.Module, suffix: str) -> str:
    """Return the one module name ending in ``suffix`` (layouts differ across releases)."""

    (name,) = [name for name, _ in model.named_modules() if name.endswith(suffix)]
    return name


def _vit() -> tuple[nn.Module, str, Callable[[Any], Tensor], int, int]:
    config = transformers.ViTConfig(
        image_size=IMAGE,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=5,
    )
    model = transformers.ViTForImageClassification(config)
    side = config.image_size // config.patch_size
    return model, _named(model, "1.layernorm_before"), _logit, side, 1


def _deit() -> tuple[nn.Module, str, Callable[[Any], Tensor], int, int]:
    config = transformers.DeiTConfig(
        image_size=IMAGE,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=5,
    )
    model = transformers.DeiTForImageClassificationWithTeacher(config)
    side = config.image_size // config.patch_size
    # Class token plus distillation token.
    return model, _named(model, "1.layernorm_before"), _logit, side, 2


def _dinov2_registers() -> tuple[nn.Module, str, Callable[[Any], Tensor], int, int]:
    config = transformers.Dinov2WithRegistersConfig(
        image_size=IMAGE,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_register_tokens=3,
        num_labels=5,
    )
    model = transformers.Dinov2WithRegistersForImageClassification(config)
    side = config.image_size // config.patch_size
    return (
        model,
        "dinov2_with_registers.encoder.layer.1.norm1",
        _logit,
        side,
        1 + config.num_register_tokens,
    )


def _clip_vision() -> tuple[nn.Module, str, Callable[[Any], Tensor], int, int]:
    config = transformers.CLIPVisionConfig(
        image_size=IMAGE,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        projection_dim=8,
    )
    model = transformers.CLIPVisionModelWithProjection(config)
    side = config.image_size // config.patch_size
    return model, "vision_model.encoder.layers.1.layer_norm1", _embed, side, 1


def _swin_stage(stage: int) -> tuple[nn.Module, str, Callable[[Any], Tensor], int, int]:
    config = transformers.SwinConfig(
        image_size=IMAGE,
        patch_size=4,
        embed_dim=16,
        depths=[1, 1],
        num_heads=[1, 2],
        window_size=4,
        num_labels=5,
    )
    model = transformers.SwinForImageClassification(config)
    # Patch merging halves the grid after every stage but the last.
    side = config.image_size // config.patch_size // 2**stage
    layer = "swin.encoder.layers.0.blocks.0.layernorm_after" if stage == 0 else "swin.layernorm"
    return model, layer, _logit, side, 0


def _logit(output: Any) -> Tensor:
    return output.logits[:, 1].sum()


def _embed(output: Any) -> Tensor:
    return output.image_embeds[:, 0].sum()


FAMILIES: dict[str, Callable[[], tuple[nn.Module, str, Callable[[Any], Tensor], int, int]]] = {
    "vit": _vit,
    "deit_distilled": _deit,
    "dinov2_registers": _dinov2_registers,
    "clip_vision": _clip_vision,
    "swin_stage0": lambda: _swin_stage(0),
    "swin_stage1": lambda: _swin_stage(1),
}


def _independent_cam(
    model: nn.Module,
    image: Tensor,
    layer: str,
    target: Callable[[Any], Tensor],
    side: int,
    prefix: int,
) -> Tensor:
    """Grad-CAM by hand: own hook, own token layout, the published formula."""

    captured: list[Tensor] = []
    module = dict(model.named_modules())[layer]
    handle = module.register_forward_hook(lambda _m, _a, out: captured.append(out))
    model.eval()
    try:
        leaf = image.clone().requires_grad_(True)
        scalar = target(model(leaf))
        (activation,) = captured
        (gradient,) = torch.autograd.grad(scalar, activation)
    finally:
        handle.remove()

    def layout(tokens: Tensor) -> Tensor:
        patches = tokens[:, prefix:, :]
        return patches.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[2], side, side)

    act_map, grad_map = layout(activation.detach()), layout(gradient)
    weights = grad_map.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act_map).sum(dim=1, keepdim=True))
    return F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_grad_cam_derives_token_grid_and_matches_independent_cam(family: str) -> None:
    """The trace proves the config's grid and prefix; the map equals a hand computation."""

    torch.manual_seed(0)
    model, layer, target, side, prefix = FAMILIES[family]()
    image = torch.randn(1, 3, IMAGE, IMAGE)

    result = attribution.grad_cam(model, image, target=target, layer=layer, overlay=False)
    expected = _independent_cam(model, image, layer, target, side, prefix)

    assert result.extra["grid_route"] == "derived"
    assert result.extra["native_map_resolution"] == (side, side)
    assert result.extra["prefix_tokens"] == prefix
    assert result.extra["grid_evidence"], "a derived grid names the ops that prove it"
    assert result.values.shape == (1, 1, IMAGE, IMAGE)
    assert bool((expected != 0).any()), "the comparison must not be between two zero maps"
    torch.testing.assert_close(result.values, expected, atol=1e-6, rtol=0)


def test_grad_cam_reshape_callable_reproduces_derived_vit_result() -> None:
    """The escape hatch gives the derived result when it states the same layout."""

    torch.manual_seed(0)
    model, layer, target, side, prefix = _vit()
    image = torch.randn(1, 3, IMAGE, IMAGE)

    derived = attribution.grad_cam(model, image, target=target, layer=layer, overlay=False)
    by_hand = attribution.grad_cam(
        model,
        image,
        target=target,
        layer=layer,
        reshape=lambda a: a[:, prefix:].transpose(1, 2).reshape(a.shape[0], a.shape[2], side, side),
        overlay=False,
    )

    assert by_hand.extra["grid_route"] == "reshape"
    assert by_hand.extra["prefix_tokens"] == prefix
    assert torch.equal(by_hand.values, derived.values)


class UnfoldTokens(nn.Module):
    """Tokens cut by ``F.unfold``: no rank-4 grid is ever flattened, so no grid is provable."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, 16)
        self.block = nn.Linear(16, 16)
        self.head = nn.Linear(16, 3)

    def forward(self, x: Tensor) -> Tensor:
        tokens = F.unfold(x, kernel_size=8, stride=8).transpose(1, 2)
        return self.head(torch.relu(self.block(self.proj(tokens))).mean(dim=1))


def test_grad_cam_refuses_unprovable_grid_and_its_remedy_runs() -> None:
    """The refusal is coded and its remedy spelling runs as written."""

    torch.manual_seed(0)
    model = UnfoldTokens()
    inputs = torch.randn(1, 3, IMAGE, IMAGE)

    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution.grad_cam(model, inputs, target=0, layer="block")

    error = excinfo.value
    assert error.fields.get("code") == "grad_cam_token_grid_unprovable"
    message = str(error)
    assert "'block'" in message and "(1, 16, 16)" in message
    remedy = re.search(r"e\.g\. (tl\.attribution\.grad_cam\(.*\)) \(drop", message)
    assert remedy is not None, message
    result = eval(remedy.group(1), {"tl": tl, "model": model, "inputs": inputs})  # noqa: S307
    assert result.extra["grid_route"] == "reshape"
    assert result.extra["native_map_resolution"] == (4, 4)


def test_grad_cam_refuses_reshape_that_returns_no_map() -> None:
    """A caller reshape must return an N,C,H,W tensor."""

    model = UnfoldTokens()
    with pytest.raises(tl.errors.AttributionError) as excinfo:
        attribution.grad_cam(
            model, torch.randn(1, 3, IMAGE, IMAGE), target=0, layer="block", reshape=lambda a: a
        )
    assert excinfo.value.fields.get("code") == "grad_cam_reshape_invalid"
