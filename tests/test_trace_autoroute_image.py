"""Tests for image auto-routing in ``tl.trace``."""

from __future__ import annotations

import base64
import builtins
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.bridge.transformers as hf_bridge
import torchlens.preprocessing


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
    """Create a small RGB PIL image (Pillow is a required dependency).

    Returns
    -------
    Any
        PIL image instance.
    """

    from PIL import Image as image_module

    return image_module.new("RGB", (32, 32), color=(120, 30, 200))


@pytest.mark.slow
def test_hf_vit_uses_auto_image_processor() -> None:
    """HF ViT image input should route through ``AutoImageProcessor``."""

    transformers = pytest.importorskip("transformers")
    import PIL.Image  # noqa: F401

    model = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224")

    log = tl.trace(model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "hf_auto_image_processor"
    assert log.input_preprocessor.verified is True


@pytest.mark.slow
def test_clip_image_uses_auto_processor_fallback() -> None:
    """CLIP image input should route through the HF image processor cascade."""

    transformers = pytest.importorskip("transformers")
    import PIL.Image  # noqa: F401

    model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    log = tl.trace(model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "hf_auto_image_processor"
    assert log.input_preprocessor.verified is True


@pytest.mark.heavy
def test_torchvision_resnet_weights_uses_attached_weights() -> None:
    """Torchvision models resolve the attached weights preset's own transform."""

    import PIL.Image  # noqa: F401

    pytest.importorskip("torchvision.transforms")
    torchvision_models = pytest.importorskip("torchvision.models")
    weights = torchvision_models.ResNet50_Weights.DEFAULT
    model = torchvision_models.resnet50(weights=None)
    model._torchlens_weights = weights

    log = tl.trace(model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "torchvision_weights"
    assert log.input_preprocessor.verified is True


@pytest.mark.slow
def test_timm_model_uses_default_cfg_transform() -> None:
    """timm models with ``default_cfg`` should use the timm resolver tier."""

    import PIL.Image  # noqa: F401

    pytest.importorskip("torchvision.transforms")
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False)

    log = tl.trace(model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "timm"
    assert log.input_preprocessor.verified is True


def test_unknown_cnn_refuses_and_names_imagenet_opt_in() -> None:
    """A PIL image with no resolvable preprocessing refuses typed (walk U02.4).

    The ImageNet default is never applied silently; the refusal names the
    explicit opt-in.
    """

    import PIL.Image  # noqa: F401

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(
            _ImageModel(), _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none")
        )

    assert excinfo.value.fields["code"] == "preprocessing_unresolved"
    assert excinfo.value.fields["model_type"] == "_ImageModel"
    assert "torchlens.preprocessing.imagenet_default()" in str(excinfo.value)
    assert "torchlens.preprocessing.attach_authority(model, weights)" in str(excinfo.value)
    assert excinfo.value.fields["resolution_attempt"]["resolution_method"] == "none"


def test_refusal_remedy_attach_authority_resolves_the_pil_input() -> None:
    """Following the refusal's attach_authority remedy lets the PIL input through."""

    import PIL.Image  # noqa: F401

    torchvision = pytest.importorskip("torchvision")
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchlens.preprocessing.attach_authority(_ImageModel(), weights)

    trace = tl.trace(model, _pil_image(), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert trace.input_preprocessor.source == "torchvision_weights"


def test_imagenet_default_opt_in_applies_and_discloses() -> None:
    """The explicit opt-in applies the recipe and stamps an unverified record."""

    import PIL.Image  # noqa: F401

    pytest.importorskip("torchvision.transforms")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        log = tl.trace(
            _ImageModel(),
            _pil_image(),
            capture=tl.options.CaptureOptions(
                input_transform=torchlens.preprocessing.imagenet_default(), layers_to_save="all"
            ),
        )

    record = log.input_preprocessor
    assert record is not None
    assert record.source == "imagenet_default"
    assert record.verified is False
    assert record.status == "unverified_fallback"
    assert record.config["resolution_method"] == "explicit_opt_in"
    assert "UNVERIFIED" in record.description
    assert log.input is not None
    first_input = log[log.input_layers[0]].out
    assert tuple(first_input.shape) == (1, 3, 224, 224)


def test_transform_override_skips_image_autoroute() -> None:
    """Explicit ``transform=`` should bypass image auto-routing."""

    import PIL.Image  # noqa: F401

    log = tl.trace(
        _ImageModel(),
        _pil_image(),
        capture=tl.options.CaptureOptions(
            input_transform=lambda image: torch.ones(1, 3, 8, 8), layers_to_save="none"
        ),
    )

    # The explicit transform bypasses the image autoroute entirely: no
    # resolver tier ran, so the provenance is the honest user_transform
    # record (tvscope B1), NEVER the autoroute's imagenet_default.
    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "user_transform"
    assert log.input_preprocessor.verified is False


def test_pil_raw_input_svg_embeds_montage_data_uri(tmp_path: Path) -> None:
    """SVG raw-input image previews should embed local montages as data URIs."""

    from PIL import Image as image_module

    images = [
        image_module.new("RGB", (24, 24), color=(120, 30, 200)),
        image_module.new("RGB", (24, 24), color=(20, 180, 90)),
    ]
    log = tl.trace(
        _ImageModel(),
        images,
        capture=tl.options.CaptureOptions(
            input_transform=lambda _: torch.ones(2, 3, 8, 8), layers_to_save="none"
        ),
    )
    output_path = tmp_path / "input_display.svg"

    log.draw(
        path=str(output_path),
        vis_save_only=True,
        file_format="svg",
    )
    svg_text = output_path.read_text(encoding="utf-8")
    href_match = re.search(r'<image\b[^>]+(?:xlink:href|href)="([^"]+)"', svg_text)

    assert href_match is not None
    href = href_match.group(1)
    assert href.startswith("data:image/png;base64,")
    payload = href.removeprefix("data:image/png;base64,")
    assert len(base64.b64decode(payload)) > 100
    assert "/tmp/torchlens_visualizers_" not in svg_text


def test_tensor_input_does_not_route_to_image_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tensor inputs should remain on the normal trace path."""

    def fail_trace_image(model: Any, image: Any, **kwargs: Any) -> object:
        """Fail if image bridge dispatch occurs."""

        raise AssertionError("image bridge should not fire for tensors")

    monkeypatch.setattr(hf_bridge, "trace_image", fail_trace_image)

    log = tl.trace(
        _ImageModel(),
        torch.ones(1, 3, 8, 8),
        capture=tl.options.CaptureOptions(layers_to_save="none"),
    )

    assert log.input_preprocessor is None


def test_list_of_pil_images_batches() -> None:
    """A list of PIL images should be transformed into a batch."""

    import PIL.Image  # noqa: F401

    pytest.importorskip("torchvision.transforms")

    log = tl.trace(
        _ImageModel(),
        [_pil_image(), _pil_image()],
        capture=tl.options.CaptureOptions(
            input_transform=torchlens.preprocessing.imagenet_default(), layers_to_save="all"
        ),
    )

    assert log.input_preprocessor is not None
    assert log.input_preprocessor.source == "imagenet_default"
    assert tuple(log[log.input_layers[0]].out.shape) == (2, 3, 224, 224)
    assert log.num_ops > 0
    assert isinstance(log.input, list)


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
