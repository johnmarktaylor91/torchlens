"""Tests for image auto-routing in ``tl.trace``."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge.hf as hf_bridge


class _ImageModel(nn.Module):
    """Tiny image model for preprocessed image tensors."""

    def __init__(self) -> None:
        """Initialize the test convolution."""

        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a minimal image forward pass."""

        return self.conv(x).mean(dim=(2, 3))


def _pil_image() -> Any:
    """Create a small RGB PIL image, skipping when PIL is unavailable.

    Returns
    -------
    Any
        PIL image instance.
    """

    image_module = pytest.importorskip("PIL.Image")
    return image_module.new("RGB", (32, 32), color=(120, 30, 200))


@pytest.mark.slow
def test_hf_vit_uses_auto_image_processor() -> None:
    """HF ViT image input should route through ``AutoImageProcessor``."""

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL.Image")
    model = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224")

    log = tl.trace(model, _pil_image(), layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "hf_auto_image_processor"
    assert log.input_preprocessor.verified is True


@pytest.mark.slow
def test_clip_image_uses_auto_processor_fallback() -> None:
    """CLIP image input should route through the HF image processor cascade."""

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL.Image")
    model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    log = tl.trace(model, _pil_image(), layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "hf_auto_image_processor"
    assert log.input_preprocessor.verified is True


def test_torchvision_resnet_weights_uses_tier_two_or_default() -> None:
    """Torchvision models use attached weights when available, otherwise default."""

    pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision.transforms")
    torchvision_models = pytest.importorskip("torchvision.models")
    weights = torchvision_models.ResNet50_Weights.DEFAULT
    model = torchvision_models.resnet50(weights=None)
    model._torchlens_weights = weights

    log = tl.trace(model, _pil_image(), layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source in {"torchvision_weights", "imagenet_default"}
    if log.input_preprocessor.source == "torchvision_weights":
        assert log.input_preprocessor.verified is True
    else:
        assert log.input_preprocessor.verified is False


@pytest.mark.slow
def test_timm_model_uses_default_cfg_transform() -> None:
    """timm models with ``default_cfg`` should use the timm resolver tier."""

    pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision.transforms")
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False)

    log = tl.trace(model, _pil_image(), layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "timm"
    assert log.input_preprocessor.verified is True


def test_unknown_cnn_uses_imagenet_default_with_warning() -> None:
    """Unknown PIL image models should fall back loudly to ImageNet defaults."""

    pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision.transforms")

    with pytest.warns(UserWarning, match="ImageNet default preprocessing"):
        log = tl.trace(_ImageModel(), _pil_image(), layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "imagenet_default"
    assert log.input_preprocessor.verified is False
    assert "UNVERIFIED" in log.input_preprocessor.description


def test_transform_override_skips_image_autoroute() -> None:
    """Explicit ``transform=`` should bypass image auto-routing."""

    pytest.importorskip("PIL.Image")

    log = tl.trace(
        _ImageModel(),
        _pil_image(),
        transform=lambda image: torch.ones(1, 3, 8, 8),
        layers_to_save="none",
    )

    assert log.input_preprocessor is None


def test_tensor_input_does_not_route_to_image_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tensor inputs should remain on the normal trace path."""

    def fail_trace_image(model: Any, image: Any, **kwargs: Any) -> object:
        """Fail if image bridge dispatch occurs."""

        raise AssertionError("image bridge should not fire for tensors")

    monkeypatch.setattr(hf_bridge, "trace_image", fail_trace_image)

    log = tl.trace(_ImageModel(), torch.ones(1, 3, 8, 8), layers_to_save="none")

    assert log.input_preprocessor is None


def test_list_of_pil_images_batches() -> None:
    """A list of PIL images should be transformed into a batch."""

    pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision.transforms")

    with pytest.warns(UserWarning, match="ImageNet default preprocessing"):
        log = tl.trace(_ImageModel(), [_pil_image(), _pil_image()], layers_to_save="none")

    assert log.input_preprocessor is not None
    assert log.num_ops > 0


def test_pil_import_failure_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image detection should fail closed when PIL cannot be imported."""

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise for PIL imports and delegate all others."""

        if name == "PIL.Image":
            raise ImportError("PIL unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert hf_bridge._is_hf_image_input(object()) is False
