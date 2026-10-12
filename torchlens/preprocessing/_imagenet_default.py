"""The explicit ImageNet-default opt-in (walk U02.4; spelling DOCUMENTED-UNSTABLE).

A PIL image whose model carries no resolvable preprocessing authority is
REFUSED at capture entry (``preprocessing_unresolved``): TorchLens never
silently feeds a model a recipe it was not trained with. A user who knows the
ImageNet recipe is right opts in by name::

    tl.trace(model, image, capture=tl.options.CaptureOptions(
        transform=tl.preprocessing.imagenet_default()))

The opt-in stays a TorchLens-authored recipe, never an authority: its record
keeps ``source="imagenet_default"`` and ``verified=False``, so its derived
status is ``unverified_fallback`` and it can never anchor a verified audit
(tvscope memo D9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchlens.data_classes.trace import ResolvedPreprocessing

__tl_layer__ = "L5"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESIZE = 256
_CROP = 224


def _imagenet_default_record() -> ResolvedPreprocessing:
    """Build the provenance record for the opted-in ImageNet default.

    Returns
    -------
    ResolvedPreprocessing
        ``source="imagenet_default"``, ``verified=False``; derived status
        ``unverified_fallback``.
    """

    from torchlens.data_classes.trace import ResolvedPreprocessing

    return ResolvedPreprocessing(
        source="imagenet_default",
        identifier="ImageNet-default-resize256-crop224",
        verified=False,
        config={
            "resolution_method": "explicit_opt_in",
            "do_resize": True,
            "size": _RESIZE,
            "do_center_crop": True,
            "center_crop_size": _CROP,
            "do_normalize": True,
            "image_mean": list(_IMAGENET_MEAN),
            "image_std": list(_IMAGENET_STD),
        },
        description=(
            "ImageNet default (UNVERIFIED, explicit opt-in): resize 256 -> "
            "center_crop 224 -> normalize mu=(0.485, 0.456, 0.406) "
            "sigma=(0.229, 0.224, 0.225). Not read from the model's own metadata."
        ),
    )


class _ImageNetDefaultTransform:
    """PIL image (or list of PIL images) to a batched ImageNet-normalized tensor."""

    def __init__(self) -> None:
        """Build the torchvision pipeline once."""

        from torchvision import transforms as T

        self._pipeline: Callable[[Any], Any] = T.Compose(
            [
                T.Resize(_RESIZE),
                T.CenterCrop(_CROP),
                T.ToTensor(),
                T.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
            ]
        )
        self._tl_preprocessing_record = _imagenet_default_record()

    def __call__(self, image: Any) -> torch.Tensor:
        """Apply the recipe and return an ``(N, 3, 224, 224)`` batch.

        Parameters
        ----------
        image:
            PIL image or non-empty list of PIL images.

        Returns
        -------
        torch.Tensor
            Batched, normalized image tensor.
        """

        if isinstance(image, list):
            items = [cast(torch.Tensor, self._pipeline(item)) for item in image]
            return torch.stack(items, dim=0)
        return cast(torch.Tensor, self._pipeline(image)).unsqueeze(0)

    def __repr__(self) -> str:
        """Return the stable spelling users typed."""

        return "torchlens.preprocessing.imagenet_default()"


def imagenet_default() -> Callable[[Any], torch.Tensor]:
    """Return the ImageNet-default image transform as an explicit opt-in.

    Resize 256, center-crop 224, ``ToTensor``, ImageNet mean/std
    normalization; a single PIL image becomes a batch of one and a list of
    PIL images becomes one batch. Pass it as
    ``CaptureOptions(transform=imagenet_default())``. The capture's
    ``input_preprocessor`` record says ``source="imagenet_default"`` with
    status ``unverified_fallback``: the recipe is TorchLens-authored, never
    read from the model.

    Returns
    -------
    Callable[[Any], torch.Tensor]
        The transform.
    """

    return _ImageNetDefaultTransform()
