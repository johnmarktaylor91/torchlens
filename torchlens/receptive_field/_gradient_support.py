"""Support ranges and spatial support of a gradient receptive field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._utils._torch_compat import tensor_any_over_dims
from ._types import GridLayout

if TYPE_CHECKING:
    pass


def _support_ranges(mask: torch.Tensor) -> tuple[tuple[int, int], ...] | None:
    """Compute a tight half-open bounding box around a Boolean support mask.

    Parameters
    ----------
    mask:
        Boolean tensor of arbitrary rank.

    Returns
    -------
    tuple[tuple[int, int], ...] or None
        One half-open range per axis, or ``None`` when support is empty.
    """

    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        return None
    if mask.ndim == 0:
        return ()
    return tuple(
        (int(coordinates[:, axis].min().item()), int(coordinates[:, axis].max().item()) + 1)
        for axis in range(mask.ndim)
    )


def _spatial_support(mask: torch.Tensor, layout: GridLayout | None) -> torch.Tensor | None:
    """Reduce support over derived non-windowed axes when layout is known.

    Parameters
    ----------
    mask:
        Full input support mask.
    layout:
        Derived layout, or ``None`` for a disconnected requested input.

    Returns
    -------
    torch.Tensor or None
        Windowed-axis support mask, or ``None`` when layout is unknown.
    """

    if layout is None or "unknown" in layout.axis_kinds:
        return None
    reduce_axes = tuple(axis for axis in range(mask.ndim) if axis not in layout.windowed_axes)
    if not reduce_axes:
        return mask.clone()
    return tensor_any_over_dims(mask, reduce_axes)
