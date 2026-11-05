"""Storage-overlap geometry for mutation-label propagation (pure, no capture state).

Split out of ``wrappers.py``: decides whether two live views on one storage
can share element bytes, so a mutation label reaches only real aliases.
"""

from __future__ import annotations

import torch

from ... import _state


def _storage_overlap_byte_interval(t: torch.Tensor) -> tuple[int, int]:
    """Return the ``[start, end)`` byte interval ``t`` can address in its storage.

    Conservative interval form (stride gaps ignored): a strided view's
    addressable extent is treated as one contiguous byte range, so two views
    with interleaved-but-disjoint elements inside the same range are treated
    as overlapping. That errs toward LINKING a mutation to a possibly-affected
    alias, never toward missing one. Negative strides do not exist in torch;
    zero strides (``expand``) contribute nothing to the span.
    """

    element_size = t.element_size()
    start = int(t.storage_offset()) * element_size
    if t.numel() == 0:
        return (start, start)
    span = 1 + sum((int(size) - 1) * int(stride) for size, stride in zip(t.shape, t.stride()))
    return (start, start + span * element_size)


# Exact element-overlap scans are vectorized over the contiguous runs of one
# view; above this many runs fall back to the conservative byte-interval answer.
_EXACT_OVERLAP_SCAN_LIMIT = 65536


def _run_lattice(t: torch.Tensor) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Describe ``t``'s element offsets as contiguous runs on a mixed-radix lattice.

    Size-1 and stride-0 (``expand``) dims address nothing new and are dropped.
    The rest, ordered by descending stride, must be NESTED: every stride exceeds
    the span the smaller-stride dims cover, so each element offset has exactly
    one digit decomposition and the run starts enumerate in ascending order.
    Innermost dims that tile memory densely telescope into one contiguous run.
    A channel slice of a batched tensor (``x[:, 0:1]`` of ``(B, C, H, W)``) is
    ``B`` runs of ``H * W`` elements; a 1-D progression is runs of length 1 (or
    one run when dense). Self-overlapping layouts (``unfold``, equal strides)
    return ``None`` so the caller keeps the conservative interval answer.

    Parameters
    ----------
    t:
        Live view whose element offsets are being described.

    Returns
    -------
    tuple[int, int, list[tuple[int, int]]] | None
        ``(start_element, run_length, outer_dims)`` with ``outer_dims`` as
        ``(size, stride)`` pairs in descending stride order (strides in
        elements), or ``None`` when the layout is not a nested lattice.
    """

    start = int(t.storage_offset())
    dims = sorted(
        (
            (int(size), int(stride))
            for size, stride in zip(t.shape, t.stride())
            if size > 1 and stride != 0
        ),
        key=lambda dim: dim[1],
    )
    inner_span = 0
    for size, stride in dims:
        if stride <= inner_span:
            return None
        inner_span += (size - 1) * stride
    run_length = 1
    while dims and dims[0][1] == run_length:
        run_length *= dims.pop(0)[0]
    return (start, run_length, dims[::-1])


def _run_starts(lattice: tuple[int, int, list[tuple[int, int]]]) -> torch.Tensor:
    """Return a lattice's run start offsets (elements), ascending."""

    start, _run_length, outer_dims = lattice
    starts = torch.tensor([start], dtype=torch.long)
    for size, stride in outer_dims:
        starts = (starts[:, None] + torch.arange(size, dtype=torch.long) * stride).flatten()
    return starts


def _lattice_floor(
    lattice: tuple[int, int, list[tuple[int, int]]], offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the greatest run start ``<=`` each offset, and where one exists.

    Greedy digit extraction is exact on a nested lattice: the largest digit
    that fits the descending-stride dim leaves a remainder the smaller dims can
    floor, and digit zero always fits.
    """

    start, _run_length, outer_dims = lattice
    remainder = offsets - start
    exists = remainder >= 0
    remainder = remainder.clamp(min=0)
    floor = torch.full_like(offsets, start)
    for size, stride in outer_dims:
        digit = (remainder // stride).clamp(max=size - 1)
        floor = floor + digit * stride
        remainder = remainder - digit * stride
    return floor, exists


def _strided_views_share_storage_elements(mutated: torch.Tensor, alias: torch.Tensor) -> bool:
    """Return whether two same-storage views share at least one element's bytes.

    The byte-interval intersection is kept as the exact NEGATIVE test (disjoint
    intervals can never share bytes) and as the conservative fallback for
    layouts the exact test cannot express. When both views are nested run
    lattices (:func:`_run_lattice`: slices, permutes and channel slices of a
    batched tensor), the answer is exact: each run of the view with fewer runs
    is checked against the run of the other view that starts at or before its
    last element. Element-DISJOINT interleaved views (``base[::2]`` vs
    ``base[1::2]``, or sibling channel slices ``x[:, 0:1]`` vs ``x[:, 1:4]`` of
    a batch) receive no invented mutation edge, while genuinely overlapping
    views keep theirs.

    Parameters
    ----------
    mutated:
        The live tensor the op wrote through.
    alias:
        Another live labeled tensor on the same storage.

    Returns
    -------
    bool
        True when the views provably or possibly share storage bytes; False
        only on proof of disjointness.
    """

    mutated_lo, mutated_hi = _storage_overlap_byte_interval(mutated)
    alias_lo, alias_hi = _storage_overlap_byte_interval(alias)
    if not (alias_lo < mutated_hi and mutated_lo < alias_hi):
        return False
    if mutated.element_size() != alias.element_size():
        # Different element widths break the shared start-address grid; the
        # windows can partially overlap without equal starts. Stay conservative.
        return True
    if mutated.numel() == 0 or alias.numel() == 0:
        return False
    mutated_lattice = _run_lattice(mutated)
    alias_lattice = _run_lattice(alias)
    if mutated_lattice is None or alias_lattice is None:
        return True
    scanned, other = sorted(
        (mutated_lattice, alias_lattice),
        key=lambda lattice: _run_count(lattice),
    )
    if _run_count(scanned) > _EXACT_OVERLAP_SCAN_LIMIT:
        return True
    run_starts = _run_starts(scanned)
    floor, exists = _lattice_floor(other, run_starts + (scanned[1] - 1))
    return bool((exists & (floor + other[1] > run_starts)).any())


def _run_count(lattice: tuple[int, int, list[tuple[int, int]]]) -> int:
    """Return how many contiguous runs a lattice holds."""

    count = 1
    for size, _stride in lattice[2]:
        count *= size
    return count


def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return byte range occupied by a tensor inside its storage."""

    with _state.pause_logging():
        element_size = tensor.element_size()
        start = int(tensor.storage_offset()) * element_size
        end = start + int(tensor.numel()) * element_size
    return start, end


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open byte ranges overlap."""

    return left[0] < right[1] and right[0] < left[1]
