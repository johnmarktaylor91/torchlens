"""Accept or refuse the preprocessing resolved for a PIL input on the image bridge."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torchlens._errors import InvalidArgumentError

__tl_layer__ = "L8"


def accept_image_resolution(model: Any, resolution: Any) -> tuple[Callable[[Any], Any], Any]:
    """Return the transform and record of a resolution, or raise the typed refusal.

    Whatever the registry resolves is applied. When NOTHING resolves the
    capture is refused ``preprocessing_unresolved``: TorchLens never applies a
    recipe the model was not shown to use.
    """

    record = resolution.preprocessing
    if resolution.input_transform is not None and record.source != "unknown":
        if record.source == "hf_image_processor":
            # Legacy source token for the model-discovered HF tier (pinned by
            # the autoroute contract tests since before the neutral resolver).
            record.source = "hf_auto_image_processor"
            record.description = f"AutoImageProcessor: {record.identifier}"
        return resolution.input_transform, record
    raise _preprocessing_unresolved(model, record)


def _preprocessing_unresolved(model: Any, record: Any) -> InvalidArgumentError:
    """Build the typed ``preprocessing_unresolved`` refusal for a PIL input."""

    reason = record.config.get("reason") if isinstance(record.config, dict) else None
    return InvalidArgumentError(
        f"torchlens.trace: no preprocessing authority resolved for the PIL image "
        f"input to {type(model).__name__} ({reason or 'nothing resolved'}). "
        "TorchLens refuses to guess a recipe the model may not have been "
        "trained with.",
        code="preprocessing_unresolved",
        remedy=(
            "record the weights the model was built with, e.g. "
            "torchlens.preprocessing.attach_authority(model, weights) (a torchvision "
            "model is recognised on its own only when its loaded state equals exactly "
            "one preset checkpoint already in the torch hub cache); or pass your "
            "loader's own transform, e.g. "
            "capture=tl.options.CaptureOptions(input_transform=weights.transforms()); "
            "or opt in to the ImageNet default (resize 256, center-crop 224, "
            "ImageNet mean/std) with "
            "capture=tl.options.CaptureOptions("
            "input_transform=torchlens.preprocessing.imagenet_default()); "
            "or pass an already-preprocessed tensor"
        ),
        model_type=type(model).__name__,
        resolution_attempt=dict(record.config) if isinstance(record.config, dict) else {},
    )
