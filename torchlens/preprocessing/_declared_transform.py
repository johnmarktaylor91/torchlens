"""Executable pipeline for a field-level preprocessing declaration.

An explicit declaration mapping (``{"resize_size": 256, "crop_size": 224,
"mean": ..., "std": ...}``) and a timm data config without timm installed
carry the fields of a recipe but no callable. This module turns those fields
into a torchvision pipeline that executes exactly what was declared and
nothing more: an unexecutable declaration (an unknown interpolation name, a
non-RGB channel order, a value range other than ``[0, 1]``, a timm crop mode
other than center) yields ``None`` so the caller keeps the honest
declaration-only resolution. Spellings DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._records import DeclaredPreprocessing

__tl_layer__ = "L5"

_INTERPOLATION_MODES = {
    "nearest": "NEAREST",
    "nearest_exact": "NEAREST_EXACT",
    "bilinear": "BILINEAR",
    "bicubic": "BICUBIC",
    "box": "BOX",
    "hamming": "HAMMING",
    "lanczos": "LANCZOS",
}
_CENTER_CROP_MODES = {None, "center"}


def _size_arg(value: Any) -> int | list[int] | None:
    """Return a torchvision size argument for a declared size, or ``None``."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return int(value)


def _stats_pair(declared: DeclaredPreprocessing) -> tuple[list[float], list[float]] | None:
    """Return ``(mean, std)`` lists when both are declared, else ``None``."""

    mean, std = declared.mean, declared.std
    if mean is None and std is None:
        return None
    if mean is None or std is None:
        raise ValueError("mean and std must be declared together")
    mean_list = [float(v) for v in mean] if isinstance(mean, (list, tuple)) else [float(mean)]
    std_list = [float(v) for v in std] if isinstance(std, (list, tuple)) else [float(std)]
    return mean_list, std_list


def _declaration_is_executable(declared: DeclaredPreprocessing) -> bool:
    """Return whether the declared fields describe a plain RGB tensor recipe."""

    if declared.modality != "image":
        return False
    if declared.channel_order not in (None, "rgb"):
        return False
    value_range = declared.value_range
    if value_range is not None and tuple(float(v) for v in value_range) != (0.0, 1.0):
        return False
    if declared.interpolation is not None and declared.interpolation not in _INTERPOLATION_MODES:
        return False
    return declared.extras.get("crop_mode") in _CENTER_CROP_MODES


def transform_from_declared(declared: DeclaredPreprocessing) -> Callable[[Any], Any] | None:
    """Build a torchvision pipeline that executes the declared fields.

    Resize (declared interpolation, antialias unless declared off), center
    crop, ``ToTensor`` and ``Normalize`` run only when their field is
    declared; a single PIL image maps to one ``(C, H, W)`` tensor (the bridge
    batches). ``None`` when torchvision is unavailable, when no field asks
    for a step, or when the declaration cannot be executed faithfully.

    Parameters
    ----------
    declared:
        Normalized declaration fields.

    Returns
    -------
    Callable[[Any], Any] | None
        The pipeline, or ``None`` for an unexecutable declaration.
    """

    try:
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode
    except ImportError:
        return None
    if not _declaration_is_executable(declared):
        return None
    try:
        stats = _stats_pair(declared)
    except (TypeError, ValueError):
        return None
    steps: list[Any] = []
    resize = _size_arg(declared.resize_size)
    if resize is not None:
        mode_name = _INTERPOLATION_MODES[declared.interpolation or "bilinear"]
        antialias = declared.antialias if isinstance(declared.antialias, bool) else True
        steps.append(
            T.Resize(
                resize, interpolation=getattr(InterpolationMode, mode_name), antialias=antialias
            )
        )
    crop = _size_arg(declared.crop_size)
    if crop is not None:
        steps.append(T.CenterCrop(crop))
    if not steps and stats is None:
        return None
    steps.append(T.ToTensor())
    if stats is not None:
        steps.append(T.Normalize(mean=stats[0], std=stats[1]))
    return T.Compose(steps)


__all__ = ["transform_from_declared"]
