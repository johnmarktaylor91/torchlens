"""Sprint B3a model-profile computed-view regressions."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.constants import MODEL_LOG_FIELD_ORDER
from torchlens.data_classes.trace import ResolvedPostprocessing, ResolvedPreprocessing, Trace


class _ProfileTiny(nn.Module):
    """Small classifier for model-profile tests."""

    def __init__(self) -> None:
        """Initialize the classifier."""

        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a tiny batch."""

        return self.fc(x)


class _ImageLike:
    """Minimal image-like object for raw-image profile tests."""

    mode = "RGB"
    size = (8, 8)

    def copy(self) -> "_ImageLike":
        """Return a shallow image-like copy.

        Returns
        -------
        _ImageLike
            New image-like object.
        """

        return _ImageLike()


def _json_keys(value: object) -> set[str]:
    """Return all object keys from a nested JSON-like value.

    Parameters
    ----------
    value:
        JSON-like object.

    Returns
    -------
    set[str]
        All dictionary keys in the object tree.
    """

    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_json_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_json_keys(item))
        return keys
    return set()


def test_model_profile_recognizes_image_classifier_with_raw_images() -> None:
    """Profile should mark image classifiers with labels and raw images applicable."""

    trace = tl.trace(_ProfileTiny(), torch.ones(2, 4), layers_to_save="none")
    trace.raw_input = [_ImageLike(), _ImageLike()]
    trace.input_preprocessor = ResolvedPreprocessing(
        source="imagenet_default",
        identifier="ImageNet-default-resize256-crop224",
        verified=False,
        config={},
        description="ImageNet default",
    )
    trace.output_postprocessor = ResolvedPostprocessing(
        source="hf_config",
        identifier="tiny",
        verified=True,
        config={"id2label": {0: "zero", 1: "one"}},
        description="tiny labels",
        style="classification",
        label_source="config.id2label",
    )
    trace.output_id2label = {0: "zero", 1: "one"}
    trace.output_num_classes = 2

    profile = trace.model_profile

    assert profile.input_modality == "image"
    assert profile.input_preprocessing_source == "imagenet_default"
    assert profile.output_postprocessing_source == "hf_config"
    assert profile.output_label_count == 2
    assert profile.has_output_labels is True
    assert profile.num_stimuli == 2
    assert profile.has_raw_images is True
    assert profile.keystone_applicable is True


def test_model_profile_plain_tensor_trace_is_not_keystone_applicable() -> None:
    """Plain tensor traces should get a conservative non-applicable profile."""

    x = torch.ones(3, 4)
    trace = tl.trace(_ProfileTiny(), x, layers_to_save="none")
    trace.raw_input = x

    profile = trace.model_profile

    assert profile.input_modality == "tensor"
    assert profile.input_preprocessing_source is None
    assert profile.output_label_count is None
    assert profile.has_output_labels is False
    assert profile.num_stimuli == 3
    assert profile.has_raw_images is False
    assert profile.keystone_applicable is False


def test_model_profile_is_not_a_persisted_trace_field(tmp_path: Path) -> None:
    """The computed profile must stay out of field order and tlspec state."""

    trace = tl.trace(_ProfileTiny(), torch.ones(2, 4), layers_to_save="none")
    bundle_path = tmp_path / "profile.tlspec"

    trace.save(bundle_path)
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    loaded = tl.load(bundle_path)

    assert "model_profile" not in MODEL_LOG_FIELD_ORDER
    assert "model_profile" not in Trace.PORTABLE_STATE_SPEC
    assert "model_profile" not in _json_keys(manifest)
    assert "model_profile" not in loaded.__dict__
    assert loaded.model_profile.keystone_applicable is False
