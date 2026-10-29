"""Storage-overlap geometry for mutation-label propagation (pure, no capture state).

Split out of ``wrappers.py``: decides whether two live views on one storage
can share element bytes, so a mutation label reaches only real aliases.
"""

from __future__ import annotations

import torch


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


# Exact element-overlap scans are vectorized O(min(n1, n2)); above this bound
# fall back to the conservative byte-interval answer instead of a large scan.
_EXACT_OVERLAP_SCAN_LIMIT = 65536


def _effective_1d_element_layout(t: torch.Tensor) -> tuple[int, int, int] | None:
    """Return ``(start_byte, stride_bytes, count)`` when element starts form one
    arithmetic progression.

    Size-1 and stride-0 (``expand``) dims contribute no distinct addresses and
    are dropped. A single remaining strided dim maps directly; a dense
    (memory-contiguous) multi-dim block telescopes to stride ``element_size``.
    Anything else (genuinely multi-strided ``as_strided`` lattices) returns
    ``None`` so the caller keeps the conservative interval answer.

    Parameters
    ----------
    t:
        Live view whose element addresses are being described.

    Returns
    -------
    tuple[int, int, int] | None
        Progression of element start addresses in storage bytes, or ``None``
        when the layout is not a single progression.
    """

    element_size = t.element_size()
    start_byte = int(t.storage_offset()) * element_size
    if t.numel() == 0:
        return (start_byte, element_size, 0)
    dims = [
        (int(size), int(stride))
        for size, stride in zip(t.shape, t.stride())
        if size > 1 and stride != 0
    ]
    if not dims:
        return (start_byte, element_size, 1)
    if len(dims) == 1:
        size, stride = dims[0]
        return (start_byte, stride * element_size, size)
    expected_stride = 1
    total = 1
    for size, stride in sorted(dims, key=lambda dim: dim[1]):
        if stride != expected_stride:
            return None
        expected_stride = stride * size
        total *= size
    return (start_byte, element_size, total)


def _strided_views_share_storage_elements(mutated: torch.Tensor, alias: torch.Tensor) -> bool:
    """Return whether two same-storage views share at least one element's bytes.

    The byte-interval intersection is kept as the exact NEGATIVE test (disjoint
    intervals can never share bytes) and as the conservative fallback for
    layouts the exact test cannot express. For the common case -- both views
    reducible to one arithmetic progression of equal-width element starts --
    the answer is computed exactly, so element-DISJOINT interleaved views
    (``base[::2]`` vs ``base[1::2]``) no longer receive an invented mutation
    edge (round-31 M4), while genuinely overlapping views keep theirs.

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
    mutated_layout = _effective_1d_element_layout(mutated)
    alias_layout = _effective_1d_element_layout(alias)
    if mutated_layout is None or alias_layout is None:
        return True
    start_a, stride_a, count_a = mutated_layout
    start_b, stride_b, count_b = alias_layout
    if count_a == 0 or count_b == 0:
        return False
    if count_a > count_b:
        start_a, stride_a, count_a, start_b, stride_b, count_b = (
            start_b,
            stride_b,
            count_b,
            start_a,
            stride_a,
            count_a,
        )
    if count_a > _EXACT_OVERLAP_SCAN_LIMIT:
        return True
    scan_starts = start_a + torch.arange(count_a, dtype=torch.long) * stride_a
    relative = scan_starts - start_b
    if stride_b <= 0:
        return True
    hits = (relative >= 0) & (relative < count_b * stride_b) & (relative % stride_b == 0)
    return bool(hits.any())
