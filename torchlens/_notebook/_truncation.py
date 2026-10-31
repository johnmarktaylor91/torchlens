"""Balanced truncation budgets, ported from treescope (B3, lane F16).

The two-stage truncation protocol is a NAMED CODE BORROW from treescope
(Daniel D. Johnson, Google DeepMind; Apache License 2.0), ported per the
treescope memo section 4 so the native grid, the bridge tensor gate, and
the offline report pick THE SAME visible cells whether or not treescope is
installed:

- :func:`infer_balanced_truncation` and :func:`compute_truncated_shape` are
  ports of the same-named functions in
  ``treescope/_internal/arrayviz_impl.py`` (treescope 0.1.10), INCLUDING
  the fifth ``doubling_bonus`` parameter -- dropping it silently changes
  which cells are visible (memo section 4, tier 1).
- :func:`truncate_tensor_with_mask` ports the slice-then-convert fetch
  recursion from ``treescope/external/torch_support.py``
  (``_truncate_and_copy`` + ``get_array_data_with_truncation``): the edges
  are sliced on the SOURCE tensor first and only the visible cells are
  converted/transferred, so picking treescope's cells never costs a
  whole-tensor copy.

The default budgets are treescope's ``ArrayAutovisualizer`` defaults
(4,000 cells / 128 per axis / 5 edge items) -- NOT ``render_array``'s
10,000/512/5, which an implementer calling upstream directly must override
explicitly (memo 3.5).

Truncation-mask law (memo 3.5): a truncation band travels as
``mask=False``, never as a data zero -- an unmasked zero band renders as
the NEUTRAL color on a diverging map, a fabricated "nothing here".

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "CELL_BUDGET_DEFAULT",
    "EDGE_ITEMS_DEFAULT",
    "PER_AXIS_BUDGET_DEFAULT",
    "compute_truncated_shape",
    "infer_balanced_truncation",
    "truncate_tensor_with_mask",
]

#: ArrayAutovisualizer default budgets (memo 3.5) -- the numbers every
#: TorchLens truncation call passes EXPLICITLY.
CELL_BUDGET_DEFAULT = 4_000
PER_AXIS_BUDGET_DEFAULT = 128
EDGE_ITEMS_DEFAULT = 5


def infer_balanced_truncation(
    shape: Sequence[int],
    maximum_size: int,
    cutoff_size_per_axis: int,
    minimum_edge_items: int,
    doubling_bonus: float = 10.0,
) -> tuple[int | None, ...]:
    """Infer a balanced truncation from a shape.

    Port of ``treescope._internal.arrayviz_impl.infer_balanced_truncation``
    (Apache-2.0, credited in the module docstring). Computes per-axis
    truncation sizes obeying the total and per-axis budgets while keeping
    the truncated array's aspect ratio resembling the original's
    (proportions follow the square root of each axis size), with a
    logarithmic ``doubling_bonus`` so longer axes stay visually longer.

    Parameters
    ----------
    shape:
        Shape of the array being truncated.
    maximum_size:
        Maximum number of elements to show; larger arrays truncate along
        one or more axes.
    cutoff_size_per_axis:
        Maximum elements of each individual axis to show without
        truncation; beyond it the visual size grows logarithmically.
    minimum_edge_items:
        Minimum values kept at each end of a truncated axis.
    doubling_bonus:
        Elements added to a truncated axis each time its true size doubles
        beyond ``cutoff_size_per_axis``.

    Returns
    -------
    tuple[int | None, ...]
        Per-axis edge sizes: ``None`` for no truncation, else the count of
        elements kept at the beginning AND at the end.
    """

    shape_arr = np.array(list(shape))
    remaining_elements_to_divide = float(maximum_size)
    edge_items_per_axis: dict[int, int | None] = {}
    # Order the shape from smallest to largest: the smallest axes need the
    # least truncation and carry the most stringent constraints.
    sorted_axes = np.argsort(shape_arr)
    sorted_shape = shape_arr[sorted_axes]

    # Per-axis maxima from the cutoff, with the doubling bonus past it.
    cutoff_adjusted_maximum_sizes = np.where(
        sorted_shape <= cutoff_size_per_axis,
        sorted_shape,
        cutoff_size_per_axis + doubling_bonus * np.log2(sorted_shape / cutoff_size_per_axis),
    )

    # Solve for the scale constant c of a proportionally-shrunk array:
    #   log(size) = ndim * log(c) + sum(log s_i)  =>
    #   c = exp((log size - sum(log s_i)) / ndim)
    axis_proportions = np.sqrt(sorted_shape)
    log_axis_proportions = np.log(axis_proportions)
    for i in range(len(sorted_axes)):
        original_axis = int(sorted_axes[i])
        size = int(shape_arr[original_axis])
        # If this axis and every later one truncated proportionally to
        # their weights, how small would this axis need to be?
        log_c = (np.log(remaining_elements_to_divide) - np.sum(log_axis_proportions[i:])) / (
            len(shape) - i
        )
        soft_limit_for_this_axis = np.exp(log_c + log_axis_proportions[i])
        cutoff_limit_for_this_axis = np.floor(
            np.minimum(soft_limit_for_this_axis, cutoff_adjusted_maximum_sizes[i])
        )
        if size <= 2 * minimum_edge_items + 1 or size <= cutoff_limit_for_this_axis:
            # Already smaller than its post-truncation size: keep it whole,
            # but consume its budget share so later axes stay monotone in
            # their true sizes.
            remaining_elements_to_divide = remaining_elements_to_divide / soft_limit_for_this_axis
            edge_items_per_axis[original_axis] = None
        elif cutoff_limit_for_this_axis < 2 * minimum_edge_items + 1:
            # Big enough to truncate but the naive target is below the
            # minimum allowed truncation: truncate to the minimum instead.
            edge_items_per_axis[original_axis] = minimum_edge_items
            remaining_elements_to_divide = remaining_elements_to_divide / (
                2 * minimum_edge_items + 1
            )
        else:
            # Truncate this axis and every remaining one at the target.
            for j in range(i, len(sorted_axes)):
                visual_size = np.floor(
                    np.minimum(
                        np.exp(log_c + log_axis_proportions[j]),
                        cutoff_adjusted_maximum_sizes[j],
                    )
                )
                edge_items_per_axis[int(sorted_axes[j])] = int(visual_size // 2)
            break

    return tuple(edge_items_per_axis[orig_axis] for orig_axis in range(len(shape)))


def compute_truncated_shape(
    shape: tuple[int, ...],
    edge_items: tuple[int | None, ...],
) -> tuple[int, ...]:
    """Compute the shape of a truncated array.

    Port of ``treescope._internal.arrayviz_impl.compute_truncated_shape``:
    a truncated axis renders ``2 * edge + 1`` cells (both edges plus the
    one-cell truncation band between them).

    Parameters
    ----------
    shape:
        Original array shape.
    edge_items:
        Per-axis edge sizes from :func:`infer_balanced_truncation`.

    Returns
    -------
    tuple[int, ...]
        Shape of the truncated (rendered) array.
    """

    return tuple(
        orig if edge is None else 2 * edge + 1 for orig, edge in zip(shape, edge_items, strict=True)
    )


def _truncate_and_copy(
    array_source: Any,
    array_dest: np.ndarray,
    prefix_slices: tuple[slice, ...],
    remaining_edge_items_per_axis: tuple[int | None, ...],
) -> None:
    """Recursively copy edge slices of a torch tensor into a numpy array.

    Port of ``treescope.external.torch_support._truncate_and_copy``: the
    source is sliced BEFORE conversion, so only visible cells transfer.
    The destination's middle band (between the two edges of a truncated
    axis) is never written -- it stays the fill value the caller chose,
    which for the mask is ``False`` (the truncation band).
    """

    if not remaining_edge_items_per_axis:
        array_dest[prefix_slices] = _to_numpy(array_source[prefix_slices])
        return
    axis = len(prefix_slices)
    edge_items = remaining_edge_items_per_axis[0]
    if edge_items is None:
        _truncate_and_copy(
            array_source,
            array_dest,
            prefix_slices + (slice(None),),
            remaining_edge_items_per_axis[1:],
        )
    else:
        if array_source.shape[axis] <= 2 * edge_items:
            raise ValueError(
                "internal truncation invariant: axis smaller than its edge budget "
                f"(axis {axis}, size {array_source.shape[axis]}, edge {edge_items})"
            )
        _truncate_and_copy(
            array_source,
            array_dest,
            prefix_slices + (slice(None, edge_items),),
            remaining_edge_items_per_axis[1:],
        )
        _truncate_and_copy(
            array_source,
            array_dest,
            prefix_slices + (slice(-edge_items, None),),
            remaining_edge_items_per_axis[1:],
        )


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert one (already sliced) tensor to numpy, widening 16-bit floats.

    Port of ``treescope.external.torch_support._tensor_to_numpy``; the
    ``force=True`` transfer only ever sees an edge slice, never the full
    payload.
    """

    import torch

    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().numpy(force=True)
    return tensor.numpy(force=True)


def truncate_tensor_with_mask(
    tensor: Any,
    edge_items_per_axis: tuple[int | None, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the truncated array and its validity mask for one tensor.

    Port of the slice-then-convert half of treescope's torch adapter
    (``TorchTensorAdapter.get_array_data_with_truncation``): edges are
    gathered on the source, the destination's truncation bands stay
    ``mask=False``, and the mask NEVER travels as a data zero.

    Parameters
    ----------
    tensor:
        Source ``torch.Tensor`` (any device; only edge slices transfer).
    edge_items_per_axis:
        Per-axis edge sizes from :func:`infer_balanced_truncation`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(values, mask)`` with identical shapes; ``mask`` is ``True``
        exactly on cells holding real source values.
    """

    import torch

    tensor = tensor.detach()
    if edge_items_per_axis == (None,) * tensor.ndim:
        values = _to_numpy(tensor)
        return values, np.ones(values.shape, dtype=bool)

    dest_shape = compute_truncated_shape(tuple(tensor.shape), edge_items_per_axis)
    # Dtype via a zero-size probe of the (possibly widened) conversion --
    # never a data-moving reshape of the payload itself.
    numpy_dtype = _to_numpy(torch.empty((0,), dtype=tensor.dtype)).dtype
    values = np.zeros(dest_shape, dtype=numpy_dtype)
    mask = np.zeros(dest_shape, dtype=np.bool_)
    _truncate_and_copy(tensor, values, (), edge_items_per_axis)
    # Broadcast view, exactly like upstream's adapter: zero-copy source of
    # ``True`` cells; only the visible edge slices materialize.
    ones_view = torch.broadcast_to(
        torch.ones((1,) * tensor.ndim, dtype=torch.bool), tuple(tensor.shape)
    )
    _truncate_and_copy(ones_view, mask, (), edge_items_per_axis)
    return values, mask
