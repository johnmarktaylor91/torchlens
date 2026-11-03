"""Byte views of tensor state: the whole-storage snapshot and the byte-exact state compare.

The buffer-write tracker and the completeness witness snapshot a buffer's whole
storage as ``uint8`` and compare it at forward end; the capture-state restore
audit compares parameter and buffer values byte for byte. Both read a sparse
tensor through its index and value components. Leaf module: it imports torch
and the sparse component helpers only.
"""

from __future__ import annotations

import torch

from ..._utils._sparse_parts import sparse_component_tensors


def _whole_storage_uint8(source: torch.Tensor) -> torch.Tensor:
    """Return a ``uint8`` view over ``source``'s ENTIRE untyped storage (all bytes).

    A zero-copy host alias (``source.detach().numpy()`` / ``untyped_storage()`` / a raw
    ``data_ptr``) shares the WHOLE storage, so a host ``__setitem__`` / ``np.as_strided`` write can
    land on bytes OUTSIDE ``source``'s element extent. Comparing the whole storage (not
    ``source``'s own extent) makes ANY host write in the shared storage detectable at forward end.
    The completeness witness (``_completeness_dispatch``) compares against the same
    bytes, so it imports this function rather than keeping a copy. A sparse tensor has
    no storage of its own: its bytes are those of its index and value components
    (:func:`_sparse_components_uint8`).
    """

    if source.layout is not torch.strided:
        return _sparse_components_uint8(source)
    untyped = source.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=source.device)
    view.set_(untyped, 0, (untyped.nbytes(),), (1,))
    return view


def _sparse_components_uint8(source: torch.Tensor) -> torch.Tensor:
    """Return the concatenated whole-storage bytes of a sparse tensor's components.

    ``untyped_storage()`` raises ``NotImplementedError`` on a sparse tensor, which
    used to escape the buffer-write tracker's install (a sparse registered buffer
    made every capture of the model fail before its forward ran). The bytes a
    write can land on are those of the layout's index and value tensors, so their
    whole storages, concatenated in the layout's fixed order, stand in for the
    one storage a strided tensor has. The result is a copy (``torch.cat``), which
    the snapshot sites clone anyway; a write that changes ``nnz`` changes the byte
    count, which ``torch.equal`` reads as a change. An unknown sparse layout
    raises ``NotImplementedError`` like the strided read it replaces.
    """

    parts = [_whole_storage_uint8(part) for part in sparse_component_tensors(source)]
    if not parts:
        return torch.empty(0, dtype=torch.uint8, device=source.device)
    return torch.cat(parts)


def _state_bytes_equal(current: torch.Tensor, baseline: torch.Tensor) -> bool:
    """NaN-safe bitwise equality over two same-shape/dtype state tensors.

    This audit detects WRITES, so the compare must be byte-exact (a tolerant
    compare would hide a genuine small state write) and NaN-safe (IEEE
    ``torch.equal`` returns False for NaN==NaN, so a model legitimately
    holding a NaN parameter/buffer was falsely accused of a double-applied
    write and lost the rescue path). Element-extent uint8 reinterpret over
    resolved contiguous copies; exotic layouts fall back to the historical
    ``torch.equal`` verdict.
    """

    # detach-ok: read-only byte comparison of state snapshots; never a
    # training-path payload, nothing retains the detached views.
    if current.layout is not torch.strided or baseline.layout is not torch.strided:
        return _sparse_state_bytes_equal(current, baseline)
    try:
        current_bytes = (
            current.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
        ).view(torch.uint8)
        baseline_bytes = (
            baseline.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
        ).view(torch.uint8)
        return bool(torch.equal(current_bytes, baseline_bytes))
    except (RuntimeError, TypeError, NotImplementedError):
        return bool(torch.equal(current, baseline))


def _sparse_state_bytes_equal(current: torch.Tensor, baseline: torch.Tensor) -> bool:
    """Byte-exact equality for sparse state: every index and value component matches.

    A sparse tensor has no storage of its own, so the dense path's uint8
    reinterpret raises and ``torch.equal`` is not implemented for every sparse
    layout; comparing the components byte-for-byte keeps the audit exact (an
    nnz change shows as a component shape change).
    """

    if current.layout is not baseline.layout:
        return False
    try:
        current_parts = sparse_component_tensors(current)
        baseline_parts = sparse_component_tensors(baseline)
    except ValueError:
        return bool(torch.equal(current, baseline))
    return len(current_parts) == len(baseline_parts) and all(
        part.shape == other.shape and _state_bytes_equal(part, other)
        for part, other in zip(current_parts, baseline_parts, strict=True)
    )
