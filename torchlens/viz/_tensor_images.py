"""Image views of a bare tensor: ``rgb`` and ``chans`` (views.4).

TorchLens ships functions, never a mutation of ``torch.Tensor``: no import
adds attributes to tensors (the same stance as ``torchlens.stats.glance``).
``rgb(t)`` draws a three-channel tensor as one picture; ``chans(t)`` draws its
leading channels as a grid. Both return a PIL image, which a notebook shows
inline and ``.save(path)`` writes.

``rgb`` reuses the min-max scaling of ``show(kind="rgb")`` (the two share
:func:`rgb_unit_hwc`); ``chans`` is ``tl.viz.nodes.channel_grid`` applied to
the tensor. Spellings DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

from typing import Literal

import torch
from PIL import Image

from ._node_visualizers import channel_grid

__all__ = ["chans", "rgb", "rgb_unit_hwc"]

RgbNormalize = Literal["minmax", "clip", "imagenet"]

#: ImageNet channel statistics, the usual torchvision ``Normalize`` constants.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_NORMALIZE_CHOICES = ("minmax", "clip", "imagenet")


def _require_tensor(tensor: object, name: str) -> torch.Tensor:
    """Return ``tensor`` when it is a tensor; raise ``TypeError`` otherwise."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}() takes a torch.Tensor, got {type(tensor).__name__}.")
    return tensor


def _rgb_chw(tensor: torch.Tensor, *, channels_last: bool) -> torch.Tensor:
    """Return one ``(3, H, W)`` float32 image from a 3D or batched 4D tensor."""

    data = tensor.detach().to(device="cpu")
    if data.ndim == 4:
        data = data[0]  # first batch element only, as channel_grid does
    if data.ndim != 3:
        raise ValueError(
            f"rgb() needs a (3, H, W) or (B, 3, H, W) tensor, or (H, W, 3) with "
            f"channels_last=True; got shape {tuple(tensor.shape)}."
        )
    if channels_last:
        data = data.permute(2, 0, 1)
    if data.shape[0] != 3:
        axis = "last" if channels_last else "first"
        raise ValueError(
            f"rgb() needs 3 channels on the {axis} axis; got shape {tuple(tensor.shape)}. "
            "For another channel count use chans(); for channels on the other axis pass "
            f"channels_last={not channels_last}."
        )
    if data.is_complex():
        data = data.abs()
    return torch.nan_to_num(data.to(torch.float32))


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


def _fit(picture: Image.Image, max_size: int) -> Image.Image:
    """Shrink a large picture into ``max_size``; enlarge a small one by whole pixels."""

    side = max(picture.size)
    if side > max_size:
        fitted = picture.copy()
        fitted.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
        return fitted
    factor = max_size // side
    if factor < 2:
        return picture
    width, height = picture.size
    return picture.resize((width * factor, height * factor), Image.Resampling.NEAREST)


def rgb(
    tensor: torch.Tensor,
    *,
    channels_last: bool = False,
    normalize: RgbNormalize = "minmax",
    max_size: int = 300,
) -> Image.Image:
    """Draw a three-channel tensor as one RGB picture.

    Parameters
    ----------
    tensor:
        ``(3, H, W)`` or ``(B, 3, H, W)`` (the first batch element is drawn),
        or ``(H, W, 3)`` with ``channels_last=True``.
    channels_last:
        Whether the channel axis is last instead of first.
    normalize:
        ``"minmax"`` (the default, as ``show(kind="rgb")``), ``"clip"`` for
        values already in ``[0, 1]``, or ``"imagenet"`` to undo ImageNet
        normalization.
    max_size:
        Largest side of the picture. A larger image shrinks to fit; a smaller
        one grows by a whole-number factor, each pixel a square block.

    Returns
    -------
    Image.Image
        The RGB picture.

    Examples
    --------
    >>> picture = tl.viz.rgb(torch.rand(3, 8, 8))
    >>> picture.mode, picture.size
    ('RGB', (296, 296))
    """

    tensor = _require_tensor(tensor, "rgb")
    if max_size < 1:
        raise ValueError("rgb() max_size must be at least 1.")
    image = rgb_unit_hwc(_rgb_chw(tensor, channels_last=channels_last), normalize=normalize)
    pixels = (image * 255.0).round().to(torch.uint8).numpy()
    return _fit(Image.fromarray(pixels, mode="RGB"), max_size)


def chans(tensor: torch.Tensor, *, n: int = 16, max_size: int = 300) -> Image.Image:
    """Draw a tensor's leading channels as a grid of grayscale tiles.

    Parameters
    ----------
    tensor:
        ``(C, H, W)`` or ``(B, C, H, W)`` (the first batch element is drawn).
    n:
        Most channels to draw; a ``+K more`` marker counts the rest.
    max_size:
        Largest side of the grid.

    Returns
    -------
    Image.Image
        The channel grid (``tl.viz.nodes.channel_grid``); each tile is
        min-max scaled on its own, so tiles share no scale.

    Examples
    --------
    >>> grid = tl.viz.chans(torch.randn(1, 6, 8, 8), n=4)
    >>> grid.mode
    'RGB'
    """

    tensor = _require_tensor(tensor, "chans")
    picture = channel_grid(n=n, max_size=max_size)(tensor)
    if picture is None:
        raise ValueError(
            f"chans() needs a (C, H, W) or (B, C, H, W) tensor; got shape {tuple(tensor.shape)}."
        )
    return picture
