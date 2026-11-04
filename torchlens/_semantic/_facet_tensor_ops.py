"""Tensor-level helpers behind facet primitives (head geometry, scatter-back)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .facets import TransformPrimitive


def _heads_geometry(value: Any, n_heads: Any, d_head: Any) -> tuple[int, int]:
    """Resolve the ``(n_heads, d_head)`` pair of a heads primitive against ``value``.

    Parameters
    ----------
    value:
        Tensor-like value about to be reshaped.
    n_heads:
        Declared head count.
    d_head:
        Declared head dimension, or ``None`` to infer it from the last axis.

    Returns
    -------
    tuple[int, int]
        Head count and head dimension.

    Raises
    ------
    ValueError
        When the head count is missing or does not divide the last axis.
    """

    if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads <= 0:
        raise ValueError(f"heads() needs a positive integer head count, got {n_heads!r}.")
    last_dim = int(value.shape[-1])
    if d_head is None:
        if last_dim % n_heads:
            raise ValueError(
                f"Cannot split the last axis of shape {tuple(value.shape)} into "
                f"{n_heads} heads: {last_dim} is not divisible by {n_heads}."
            )
        return n_heads, last_dim // n_heads
    return n_heads, int(d_head)


def _ensure_scatter_capable(primitive: TransformPrimitive) -> None:
    """Raise if a primitive is not scatter-back capable.

    Parameters
    ----------
    primitive:
        Transform primitive to validate.

    Returns
    -------
    None
        Raises when the primitive is not write-safe.
    """

    if primitive.capability_class == "aliasing_selection":
        raise RuntimeError("aliasing_selection scatter_update requires an explicit alias policy.")
    if primitive.capability_class in {"computed", "codec"} or not primitive.flags.write:
        raise RuntimeError(f"Primitive {primitive.kind!r} is not scatter-back capable.")


def _copy_scatter_value(target: torch.Tensor, edited_slice: torch.Tensor, *, mode: str) -> None:
    """Copy an edited value into a scatter target after metadata checks.

    Parameters
    ----------
    target:
        View into the cloned full home tensor.
    edited_slice:
        Edited value returned by a facet hook.
    mode:
        Scatter mode, currently ``"replace"`` or ``"add"``.

    Returns
    -------
    None
        ``target`` is updated in place.
    """

    if target.dtype != edited_slice.dtype:
        raise TypeError(
            f"Facet scatter_update dtype mismatch: expected {target.dtype}, got {edited_slice.dtype}."
        )
    if target.device != edited_slice.device:
        raise TypeError(
            f"Facet scatter_update device mismatch: expected {target.device}, got "
            f"{edited_slice.device}."
        )
    if tuple(target.shape) != tuple(edited_slice.shape):
        raise ValueError(
            f"Facet scatter_update shape mismatch: expected {tuple(target.shape)}, "
            f"got {tuple(edited_slice.shape)}."
        )
    if mode == "replace":
        target.copy_(edited_slice)
        return
    if mode == "add":
        target.add_(edited_slice)
        return
    raise ValueError("Facet scatter_update mode must be 'replace' or 'add'.")
