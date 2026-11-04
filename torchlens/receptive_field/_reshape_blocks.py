"""Block decomposition of general reshapes for the receptive-field engines.

A reshape that mixes non-singleton axes (``(B, S, H*D) -> (B, S, H, D)``) destroys
no coordinate information: it regroups contiguous runs of axes whose extents
multiply to the same count. :func:`block_decomposition` finds those runs. A block
with at most one non-singleton axis per side is an exact axis map; a mixed block is
enumerated exactly by the per-unit index-set walks (:func:`block_value_image`) and
widened honestly by the axis-level descriptor engines, which have no per-axis slot
for a coordinate that depends on several others.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import prod
from typing import TYPE_CHECKING

from ._index_set import _AxisSets, _IndexSet

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from ._rules import _RuleResult

Block = tuple[tuple[int, ...], tuple[int, ...]]
"""One reshape block: ``(parent axes, output axes)``, each ascending and contiguous."""

_BLOCK_ENUMERATION_BUDGET = 1 << 16


def block_decomposition(
    parent_shape: Sequence[int], out_shape: Sequence[int]
) -> tuple[Block, ...] | None:
    """Split a reshape into contiguous axis runs of equal element count.

    Parameters
    ----------
    parent_shape:
        Shape before the reshape.
    out_shape:
        Shape after the reshape.

    Returns
    -------
    tuple[Block, ...] | None
        Blocks covering every axis of both shapes, or ``None`` when the shapes
        hold a zero extent or different element counts (the reshape is then
        not a regrouping the engines can reason about).
    """

    parent = [int(extent) for extent in parent_shape]
    out = [int(extent) for extent in out_shape]
    if any(extent <= 0 for extent in parent + out) or prod(parent) != prod(out):
        return None
    blocks: list[Block] = []
    parent_axis = 0
    out_axis = 0
    while parent_axis < len(parent) and out_axis < len(out):
        parent_axes = [parent_axis]
        out_axes = [out_axis]
        parent_count = parent[parent_axis]
        out_count = out[out_axis]
        parent_axis += 1
        out_axis += 1
        while parent_count != out_count:
            if parent_count < out_count:
                if parent_axis >= len(parent):
                    return None
                parent_count *= parent[parent_axis]
                parent_axes.append(parent_axis)
                parent_axis += 1
            else:
                if out_axis >= len(out):
                    return None
                out_count *= out[out_axis]
                out_axes.append(out_axis)
                out_axis += 1
        blocks.append((tuple(parent_axes), tuple(out_axes)))
    if any(extent != 1 for extent in parent[parent_axis:] + out[out_axis:]):
        return None
    blocks.extend(((axis,), ()) for axis in range(parent_axis, len(parent)))
    blocks.extend(((), (axis,)) for axis in range(out_axis, len(out)))
    return tuple(blocks)


def _ravel(coordinates: Sequence[int], extents: Sequence[int]) -> int:
    """Row-major flat index of one coordinate tuple."""

    flat = 0
    for coordinate, extent in zip(coordinates, extents, strict=True):
        flat = flat * extent + coordinate
    return flat


def _unravel(flat: int, extents: Sequence[int]) -> tuple[int, ...]:
    """Row-major coordinate tuple of one flat index."""

    coordinates: list[int] = []
    for extent in reversed(extents):
        coordinates.append(flat % extent)
        flat //= extent
    return tuple(reversed(coordinates))


def block_value_image(
    source_sets: Sequence[_IndexSet],
    source_extents: Sequence[int],
    dest_extents: Sequence[int],
) -> tuple[tuple[_IndexSet, ...], bool]:
    """Map per-axis index sets through one mixed reshape block exactly.

    Parameters
    ----------
    source_sets:
        One index set per source axis of the block (all constrained).
    source_extents:
        Extents of the source axes, in block order.
    dest_extents:
        Extents of the destination axes, in block order.

    Returns
    -------
    tuple[tuple[_IndexSet, ...], bool]
        One index set per destination axis and whether their Cartesian product
        is exactly the image. Over the enumeration budget the whole destination
        extents are returned as an inexact envelope.
    """

    sizes = [sum(item.count for item in index_set.progressions) for index_set in source_sets]
    if any(size == 0 for size in sizes):
        return tuple(_IndexSet.empty() for _ in dest_extents), True
    if prod(sizes) > _BLOCK_ENUMERATION_BUDGET:
        return (
            tuple(_IndexSet.interval(0, extent - 1, exact=False) for extent in dest_extents),
            False,
        )
    seen: set[tuple[int, ...]] = set()
    for coordinates in product(*(index_set.values() for index_set in source_sets)):
        if any(
            coordinate < 0 or coordinate >= extent
            for coordinate, extent in zip(coordinates, source_extents, strict=True)
        ):
            continue
        seen.add(_unravel(_ravel(coordinates, source_extents), dest_extents))
    per_axis = [sorted({point[axis] for point in seen}) for axis in range(len(dest_extents))]
    exact = all(index_set.exact for index_set in source_sets) and len(seen) == prod(
        len(values) for values in per_axis
    )
    return tuple(_IndexSet.from_values(values, exact=exact) for values in per_axis), exact


def apply_blocks(
    result: _RuleResult,
    source_sets: _AxisSets,
    source: Op,
    dest: Op,
    mapped: list[_IndexSet | None],
    exact: bool,
    *,
    forward: bool,
) -> tuple[_AxisSets, bool]:
    """Finish an axis-map hop: overlay the mixed-block images on ``mapped``.

    Axes outside every block keep the sets the caller already mapped; a block
    with an unconstrained source axis leaves its destination axes alone.
    """

    block_sets, blocks_exact = map_blocks(
        result.values.get("blocks"), source_sets, source.shape, dest.shape, forward=forward
    )
    for axis, block_set in block_sets.items():
        mapped[axis] = block_set
    return tuple(mapped), exact and blocks_exact


def map_blocks(
    blocks: object,
    source_sets: Sequence[_IndexSet | None],
    source_shape: Sequence[int],
    dest_shape: Sequence[int],
    *,
    forward: bool,
) -> tuple[dict[int, _IndexSet], bool]:
    """Map every mixed block of an axis-map rule result between two index-set frames.

    Parameters
    ----------
    blocks:
        The rule result's recorded ``blocks`` value: ``(parent axes, out axes)``
        pairs with more than one non-singleton axis on one side.
    source_sets:
        Index sets in the frame being left (output frame when ``forward`` is
        false, parent frame when true).
    source_shape:
        Shape of the frame being left.
    dest_shape:
        Shape of the frame being entered.
    forward:
        ``True`` when walking parent to output.

    Returns
    -------
    tuple[dict[int, _IndexSet], bool]
        Destination index sets keyed by destination axis, and whether every
        mapped block was exact. A block with an unconstrained source axis
        leaves its destination axes unconstrained.
    """

    mapped: dict[int, _IndexSet] = {}
    exact = True
    if not isinstance(blocks, Sequence):
        return mapped, exact
    for block in blocks:
        if not (isinstance(block, Sequence) and len(block) == 2):
            continue
        parent_axes, out_axes = (tuple(int(axis) for axis in side) for side in block)
        source_axes, dest_axes = (parent_axes, out_axes) if forward else (out_axes, parent_axes)
        if not source_axes or not dest_axes:
            continue
        sets = [source_sets[axis] for axis in source_axes]
        if any(item is None for item in sets):
            continue
        image, block_exact = block_value_image(
            [item for item in sets if item is not None],
            [int(source_shape[axis]) for axis in source_axes],
            [int(dest_shape[axis]) for axis in dest_axes],
        )
        mapped.update(zip(dest_axes, image, strict=True))
        exact = exact and block_exact
    return mapped, exact


__all__ = ["apply_blocks", "block_decomposition", "block_value_image", "map_blocks"]
