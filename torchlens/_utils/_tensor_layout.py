"""Capture-time argument layouts for saved-DAG replay.

A saved op output is a compact copy: cloning a non-dense view (``x[:, :n]`` of a
wider tensor) yields a row-major tensor. Replay feeds those copies back into the
recorded functions, but a kernel may round differently on a strided operand than
on a contiguous one (a BLAS matmul over a sliced input, for example), so a replay
from the compact copy can differ in the last bits from what capture recorded.

Capture therefore records the layout an argument had when the function saw it
(:func:`layout_of`, kept on ``ParentRef._capture_layout``), and replay re-lays the
resolved value into that layout before the call (:func:`relay_to_layout`). The
re-lay is a bijective copy into a fresh buffer with the recorded stride and storage
offset, so the values are unchanged and only their placement in memory follows the
capture. Overlapping layouts (broadcast stride-0 dims, overlapping windows) are left
compact: a copy into them could not keep distinct values distinct.
"""

from __future__ import annotations

import torch

CaptureLayout = tuple[tuple[int, ...], tuple[int, ...], int]
"""``(shape, stride, storage_offset)`` of a tensor argument at capture."""

_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _plain_strided(value: object) -> bool:
    """Whether ``value`` is a plain (non-subclass) strided tensor."""

    return type(value) in _PLAIN_TENSOR_TYPES and value.layout == torch.strided  # type: ignore[attr-defined]


def _non_overlapping(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """Whether no two elements of ``shape``/``stride`` share a storage location.

    Sufficient test: ordered by stride, each dim of size above one must step past
    the whole extent the smaller-stride dims already cover.
    """

    extent = 0
    for step, size in sorted((st, sz) for sz, st in zip(shape, stride, strict=True) if sz > 1):
        if step <= extent:
            return False
        extent += (size - 1) * step
    return True


def layout_of(value: object) -> CaptureLayout | None:
    """Return the layout to record for a replayed argument, or ``None``.

    ``None`` when ``value`` is not a plain strided tensor, is row-major contiguous
    (the compact copy already has its layout), is empty, or has an overlapping
    layout replay could not reproduce. Callers inside a capture wrap this in the
    sanctioned internal-read context: ``stride()``/``storage_offset()`` are
    witnessed metadata surfaces.
    """

    if not _plain_strided(value):
        return None
    tensor: torch.Tensor = value  # type: ignore[assignment]
    if tensor.numel() == 0 or tensor.is_contiguous():
        return None
    shape = tuple(int(size) for size in tensor.shape)
    stride = tuple(int(step) for step in tensor.stride())
    if not _non_overlapping(shape, stride):
        return None
    return shape, stride, int(tensor.storage_offset())


def relay_to_layout(value: object, layout: CaptureLayout | None) -> object:
    """Return ``value`` placed in the recorded capture ``layout``.

    Returns ``value`` itself when there is no layout, when it is not a plain
    strided tensor of the recorded shape, or when it already has the recorded
    stride. Otherwise copies it into a fresh buffer viewed with the recorded
    stride and storage offset; the copy runs under the caller's grad mode, so a
    differentiable replay keeps its gradient path through it.
    """

    if layout is None or not _plain_strided(value):
        return value
    tensor: torch.Tensor = value  # type: ignore[assignment]
    shape, stride, offset = layout
    if tuple(tensor.shape) != shape or tuple(tensor.stride()) == stride:
        return value
    span = offset + sum((size - 1) * step for size, step in zip(shape, stride, strict=True)) + 1
    buffer = torch.empty(span, dtype=tensor.dtype, device=tensor.device)
    buffer.as_strided(shape, stride, offset).copy_(tensor)
    return buffer.as_strided(shape, stride, offset)


__all__ = ["CaptureLayout", "layout_of", "relay_to_layout"]
