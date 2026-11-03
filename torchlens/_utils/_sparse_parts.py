"""Sparse tensor components: the index and value tensors per layout, and rebuilding with new values.

A sparse tensor owns no storage of its own; its bytes live in the layout's index
and value component tensors. The capture-state snapshot, the state restore audit,
the save budget and validation's perturbation all read those components through
these helpers, so the per-layout accessor table exists once. Leaf module: it
imports torch only.
"""

from __future__ import annotations

from typing import Any

import torch

_SPARSE_COMPONENT_ACCESSORS: dict[Any, tuple[str, ...]] = {
    torch.sparse_coo: ("_indices", "_values"),
    torch.sparse_csr: ("crow_indices", "col_indices", "values"),
    torch.sparse_csc: ("ccol_indices", "row_indices", "values"),
    torch.sparse_bsr: ("crow_indices", "col_indices", "values"),
    torch.sparse_bsc: ("ccol_indices", "row_indices", "values"),
}
"""Physical component tensors per sparse layout: every index tensor AND values."""


def sparse_component_tensors(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the physical component tensors of a sparse tensor.

    Parameters
    ----------
    t:
        Sparse tensor (COO or any compressed layout).

    Returns
    -------
    tuple[torch.Tensor, ...]
        Index tensor(s) and values tensor backing ``t``.

    Raises
    ------
    ValueError
        For non-strided layouts without a known component decomposition
        (callers treat that as unmeasurable, never as zero-index-bytes).
    """

    accessors = _SPARSE_COMPONENT_ACCESSORS.get(t.layout)
    if accessors is None:
        raise ValueError(f"no known component decomposition for layout {t.layout}")
    return tuple(getattr(t, name)() for name in accessors)


def sparse_with_values(t: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Return a sparse tensor with ``t``'s layout, indices and shape but new ``values``.

    Parameters
    ----------
    t:
        Sparse tensor (COO or any compressed layout) whose structure is kept.
    values:
        Replacement values tensor, the same shape as ``t``'s own values component.

    Returns
    -------
    torch.Tensor
        A new sparse tensor; ``t`` is not modified.

    Raises
    ------
    ValueError
        For layouts without a known component decomposition.
    """

    *indices, _old_values = sparse_component_tensors(t)
    if t.layout is torch.sparse_coo:
        return torch.sparse_coo_tensor(indices[0], values, t.shape, device=t.device)
    compressed, plain = indices
    return torch.sparse_compressed_tensor(
        compressed, plain, values, t.shape, layout=t.layout, device=t.device
    )
