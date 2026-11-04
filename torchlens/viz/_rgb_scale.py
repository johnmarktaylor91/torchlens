"""Scale a three-channel image into ``[0, 1]`` for display.

Shared by ``show(kind="rgb")`` and ``tl.viz.rgb`` so the two draw the same
picture. A leaf module: it imports nothing from TorchLens, so the tensor
display path stays free of the node-visualizer imports.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["RgbNormalize", "rgb_unit_hwc"]

RgbNormalize = Literal["minmax", "clip", "imagenet"]

#: ImageNet channel statistics, the usual torchvision ``Normalize`` constants.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_NORMALIZE_CHOICES = ("minmax", "clip", "imagenet")


def rgb_unit_hwc(image: torch.Tensor, *, normalize: RgbNormalize = "minmax") -> torch.Tensor:
    """Scale a ``(3, H, W)`` image into ``[0, 1]`` and return it as ``(H, W, 3)``.

    Parameters
    ----------
    image:
        Float ``(3, H, W)`` image.
    normalize:
        ``"minmax"`` stretches the whole image's range to ``[0, 1]``;
        ``"clip"`` clamps values already meant to lie in ``[0, 1]``;
        ``"imagenet"`` undoes ImageNet mean and standard deviation
        normalization, then clamps.

    Returns
    -------
    torch.Tensor
        ``(H, W, 3)`` float tensor in ``[0, 1]``.
    """

    if normalize not in _NORMALIZE_CHOICES:
        raise ValueError(f"normalize must be one of {_NORMALIZE_CHOICES}; got {normalize!r}.")
    if normalize == "imagenet":
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        image = (image * std + mean).clamp(0.0, 1.0)
    elif normalize == "clip":
        image = image.clamp(0.0, 1.0)
    else:
        image = image - image.min()
        top = image.max()
        if top > 0:
            image = image / top
    return image.permute(1, 2, 0)
