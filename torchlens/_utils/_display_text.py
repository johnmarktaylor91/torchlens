"""Tensor, dtype, shape and device spellings shared by every printout.

House style v1: ``float32 [1, 5, 3072] on cpu``. Only metadata attributes
(``dtype``, ``shape``, ``device``) are read; tensor data is never touched,
so a lazy, disk-backed or device tensor is safe to describe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def dtype_text(dtype: Any) -> str:
    """``torch.float32`` -> ``float32``; ``None`` -> ``dtype unknown``."""

    if dtype is None:
        return "dtype unknown"
    return str(dtype).removeprefix("torch.")


def shape_text(shape: Sequence[int] | None) -> str:
    """``(1, 5, 3072)`` -> ``[1, 5, 3072]``; scalar ``[]``; ``None`` -> ``shape unknown``."""

    if shape is None:
        return "shape unknown"
    return "[" + ", ".join(str(int(dim)) for dim in shape) + "]"


def device_text(device: Any) -> str:
    """``device(type='cpu')`` -> ``cpu``; ``cuda:0`` stays ``cuda:0``."""

    return str(device)


def tensor_text(dtype: Any, shape: Sequence[int] | None, device: Any = None) -> str:
    """``float32 [1, 5, 3072] on cpu``; the device clause is omitted when unknown."""

    text = f"{dtype_text(dtype)} {shape_text(shape)}"
    if device is not None:
        text += f" on {device_text(device)}"
    return text


def tensor_stand_in(tensor: Any) -> str:
    """Placeholder for a tensor inside another printout: ``<tensor float32 [4] on cpu>``.

    Reads metadata only; never prints elements.
    """

    shape = getattr(tensor, "shape", None)
    return f"<tensor {tensor_text(getattr(tensor, 'dtype', None), shape, getattr(tensor, 'device', None))}>"
