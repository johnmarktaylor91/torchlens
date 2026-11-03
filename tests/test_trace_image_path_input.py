"""``tl.trace`` opens an image file path as an image before routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.preprocessing
from torchlens._errors import InvalidArgumentError
from torchlens._input_coerce import _materialize_image_paths


class _ImageModel(nn.Module):
    """Tiny image model for preprocessed image tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.conv(x).mean(dim=(2, 3))


def _write_image(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (32, 32), color=(120, 30, 200)).save(path)
    return path


def _attached_model() -> nn.Module:
    torchvision_models = pytest.importorskip("torchvision.models")
    weights = torchvision_models.ResNet18_Weights.IMAGENET1K_V1
    return torchlens.preprocessing.attach_authority(_ImageModel(), weights)


def test_image_path_string_takes_the_image_route(tmp_path: Path) -> None:
    """A path string to a real image file traces like the PIL image it names."""

    path = _write_image(tmp_path / "dog.jpg")
    model = _attached_model()

    trace = tl.trace(model, str(path), capture=tl.options.CaptureOptions(layers_to_save="none"))

    assert trace.provenance.input_preprocessor is not None
    assert trace.provenance.input_preprocessor.source == "torchvision_weights"


def test_pathlike_and_list_of_paths_materialize(tmp_path: Path) -> None:
    """``os.PathLike`` and a list of paths become one image or a list of images."""

    from PIL.Image import Image

    first = _write_image(tmp_path / "a.PNG")
    second = _write_image(tmp_path / "b.webp")

    single = _materialize_image_paths(_ImageModel(), first)
    assert isinstance(single, Image)
    assert single.mode == "RGB"
    batch = _materialize_image_paths(_ImageModel(), [str(first), second])
    assert isinstance(batch, list) and len(batch) == 2
    assert all(isinstance(item, Image) for item in batch)


def test_text_and_non_image_inputs_pass_through(tmp_path: Path) -> None:
    """Prose, non-image suffixes, tensors and mixed lists are left untouched."""

    model = _ImageModel()
    tensor = torch.ones(1, 3, 4, 4)
    image = _write_image(tmp_path / "c.jpg")
    for value in (
        "The capital of France is",
        "a dog.jpg story",
        "notes.txt",
        tensor,
        ["x", str(image)],
        [],
    ):
        assert _materialize_image_paths(model, value) is value


def test_missing_image_path_refuses_without_a_tokenizer() -> None:
    """A missing image path refuses typed instead of falling into the text path."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(_ImageModel(), "no_such_dog.jpg")
    assert excinfo.value.code == "image_path_missing"
    assert "no_such_dog.jpg" in str(excinfo.value)


def test_missing_image_path_stays_text_for_a_tokenizer_model() -> None:
    """A model with a tokenizer reads a missing image-suffixed string as text."""

    class _Tokenizing(nn.Module):
        tokenizer = staticmethod(
            lambda text, **_: {"input_ids": torch.ones(1, 2, dtype=torch.long)}
        )

    model = _Tokenizing()
    assert _materialize_image_paths(model, "no_such_dog.jpg") == "no_such_dog.jpg"


def test_undecodable_image_file_refuses_typed(tmp_path: Path) -> None:
    """An image-suffixed file that is not an image refuses ``image_path_unreadable``."""

    bogus = tmp_path / "broken.png"
    bogus.write_bytes(b"not an image")
    with pytest.raises(InvalidArgumentError) as excinfo:
        _materialize_image_paths(_ImageModel(), str(bogus))
    assert excinfo.value.code == "image_path_unreadable"


def test_explicit_input_transform_still_receives_the_opened_image(tmp_path: Path) -> None:
    """With ``input_transform`` set, the transform receives the opened image, not the path."""

    seen: list[Any] = []

    def transform(value: Any) -> torch.Tensor:
        seen.append(value)
        return torch.ones(1, 3, 4, 4)

    path = _write_image(tmp_path / "d.jpg")
    tl.trace(
        _ImageModel(),
        str(path),
        capture=tl.options.CaptureOptions(input_transform=transform, layers_to_save="none"),
    )
    assert len(seen) == 1
    assert not isinstance(seen[0], str)
