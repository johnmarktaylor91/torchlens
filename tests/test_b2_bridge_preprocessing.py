"""Preprocessing registry gaps and the on-request card check (batch-1 gaps, walk U24.4).

- torchvision weights: a model keeps no handle to its weights, so
  ``attach_authority`` records them and discovery resolves them;
- an explicit capture ``transform=`` that is a recognized authority stamps
  that authority's own record, while any other transform stays
  ``user_transform``;
- open_clip models resolve their own ``visual.preprocess_cfg``;
- ``check_card`` compares the recorded preprocessing with the model's card,
  warns on a mismatch, and runs only when called.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import preprocessing as tlp
from torchlens._errors import InvalidArgumentError

torchvision = pytest.importorskip("torchvision")

_V1 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
_V2 = torchvision.models.ResNet50_Weights.IMAGENET1K_V2


class _TinyImageNet(nn.Module):
    """A conv stem and a classifier over a 224 px image batch."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 7, stride=8)
        self.head = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool and classify."""

        return self.head(self.stem(x).mean(dim=(2, 3)))


def _trace_with(transform: Any, model: nn.Module | None = None) -> Any:
    model = model or _TinyImageNet().eval()
    image = torch.rand(1, 3, 256, 256)
    return tl.trace(model, image, capture=tl.options.CaptureOptions(transform=transform))


def _codes(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(getattr(item.message, "code", "")) for item in caught]


def test_attached_torchvision_weights_are_discovered() -> None:
    model = torchvision.models.resnet18(weights=None)
    modules_before = [name for name, _ in model.named_modules()]
    assert tlp.resolve(model=model).record.source == "unknown"

    assert tlp.attach_authority(model, _V1) is model

    resolution = tlp.resolve(model=model)
    assert resolution.record.source == "torchvision_weights"
    assert resolution.record.config["resolution_method"] == "model_metadata"
    assert resolution.status == tlp.STATUS_AUTHORITATIVE
    assert resolution.declared is not None and resolution.declared.resize_size == 256
    # Attaching records metadata only: the module tree is unchanged.
    assert [name for name, _ in model.named_modules()] == modules_before


def test_attach_refuses_an_opaque_callable() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        tlp.attach_authority(_TinyImageNet(), lambda image: image)
    assert info.value.fields["code"] == "preprocessing_attach_opaque"


def test_explicit_transform_that_is_an_authority_stamps_its_own_record() -> None:
    trace = _trace_with(_V1.transforms())
    record = trace.input_preprocessor
    expected = tlp.resolve(_V1.transforms()).record
    assert record.source == "torchvision_preset" == expected.source
    assert record.config == expected.config
    assert tlp.status_of(record) == tlp.STATUS_AUTHORITATIVE


def test_unrecognized_transform_still_stamps_user_transform() -> None:
    def halve(image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image, size=(224, 224)) / 2

    record = _trace_with(halve).input_preprocessor
    assert record.source == "user_transform"
    assert tlp.status_of(record) == tlp.STATUS_UNKNOWN


def test_open_clip_model_resolves_its_own_preprocess_cfg() -> None:
    open_clip = pytest.importorskip("open_clip")
    model = open_clip.create_model("ViT-B-32", pretrained=None)
    config = model.visual.preprocess_cfg

    resolution = tlp.resolve(model=model)

    assert resolution.record.source == "open_clip_preprocess_cfg"
    assert resolution.record.config["resolution_method"] == "model_metadata"
    assert resolution.status == tlp.STATUS_AUTHORITATIVE
    declared = resolution.declared
    assert declared is not None
    assert declared.mean == pytest.approx(tuple(config["mean"]))
    assert declared.std == pytest.approx(tuple(config["std"]))
    assert declared.crop_size == 224
    assert declared.interpolation == "bicubic"
    assert callable(resolution.transform)


def test_check_card_agrees_with_a_matching_card() -> None:
    model = tlp.attach_authority(_TinyImageNet().eval(), _V1)
    trace = _trace_with(_V1.transforms(), model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = tlp.check_card(trace)
    assert report.verdict == "verified"
    assert report.applied_source == "recorded"
    assert "preprocessing_card_mismatch" not in _codes(caught)


def test_check_card_warns_on_a_mismatched_card() -> None:
    trace = _trace_with(_V1.transforms())
    with pytest.warns(tl.errors.TorchLensWarning, match="does not match the model's card"):
        report = tlp.check_card(trace, authority=_V2)
    assert report.verdict == "mismatch"
    assert "resize_size" in report.mismatched_fields


def test_check_card_never_runs_on_its_own() -> None:
    """A capture whose transform contradicts the model's card stays silent."""

    model = tlp.attach_authority(_TinyImageNet().eval(), _V2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = _trace_with(_V1.transforms(), model)
    assert "preprocessing_card_mismatch" not in _codes(caught)
    # Asked explicitly, the same trace warns.
    with pytest.warns(tl.errors.TorchLensWarning):
        tlp.check_card(trace)


def test_check_card_on_an_opaque_record_is_unknown_not_a_match() -> None:
    trace = _trace_with(lambda image: torch.nn.functional.interpolate(image, size=(224, 224)))
    report = tlp.check_card(trace, authority=_V1)
    assert report.verdict == "unknown"


def test_check_card_without_a_card_refuses() -> None:
    record = tlp.resolve(_V1).record
    with pytest.raises(InvalidArgumentError) as info:
        tlp.check_card(record)
    assert info.value.fields["code"] == "preprocessing_card_unresolved"
