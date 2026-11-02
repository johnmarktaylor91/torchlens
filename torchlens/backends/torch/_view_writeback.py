"""Storage-alias helpers for view mutations.

``wrappers._propagate_mutation_label_to_storage_aliases`` advances a base
tensor's label to an in-place op on one of its views; this module finds the
live labeled aliases that mutation reaches and answers whether a ``.data``
alias write will reach a labeled base. The write-back of an augmented slice
assignment is bound to the base's pre-mutation producer by
``_ops_capture_records._bind_writeback_receiver_label``.
"""

from __future__ import annotations

from typing import Any

import torch

from ... import _state
from ._internal_read import internal_scalar_read
from ._storage_overlap import (
    _storage_overlap_byte_interval,
    _strided_views_share_storage_elements,
)
from ._tl import (
    get_tensor_label,
    session_storage_alias_candidates,
    storage_alias_index_key,
)


def view_mutation_propagation_enabled(trace: Any) -> bool:
    """Return whether this capture advances base aliases on a view mutation.

    Descriptor/grad-bound captures keep the historical topology (see the SCOPE
    note in ``wrappers._propagate_mutation_label_to_storage_aliases``).
    """

    return not (
        getattr(trace, "is_replayable", False)
        or trace._options_backward_ready
        or trace._options_save_grads not in (None, False)
    )


def _alias_shares_elements(mutated: torch.Tensor, alias: torch.Tensor, storage_ptr: Any) -> bool:
    """Return whether ``alias`` provably shares storage elements with ``mutated``.

    Element-exact where provable (round-31 M4): interleaved element-disjoint
    views must not inherit the mutation edge. A probe failure reads as no.
    """

    try:
        if (
            storage_alias_index_key(alias.untyped_storage()) != storage_ptr
            or alias.device != mutated.device
        ):
            return False
        return bool(_strided_views_share_storage_elements(mutated, alias))
    except Exception:
        return False


def overlapping_labeled_aliases(
    mutated: torch.Tensor, exclude_label: str | None
) -> list[tuple[torch.Tensor, str]]:
    """Return every OTHER live labeled tensor sharing storage elements with ``mutated``.

    Callers run this under ``pause_logging`` and ``internal_scalar_read``: the
    storage and stride reads are TorchLens's own bookkeeping.

    Parameters
    ----------
    mutated
        The live tensor an in-place op writes through.
    exclude_label
        A label to skip (the mutating op's own), or ``None``.

    Returns
    -------
    list[tuple[torch.Tensor, str]]
        ``(alias, alias_label)`` pairs, in session index order.
    """

    try:
        # Meta-safe key (D20): data_ptr() reads 0 for EVERY meta storage.
        storage_ptr = storage_alias_index_key(mutated.untyped_storage())
    except Exception:
        return []
    if storage_ptr is None:
        return []
    candidates = session_storage_alias_candidates(storage_ptr)
    if not candidates or (len(candidates) == 1 and candidates[0] is mutated):
        return []
    mutated_lo, mutated_hi = _storage_overlap_byte_interval(mutated)
    if mutated_hi <= mutated_lo:
        return []
    found: list[tuple[torch.Tensor, str]] = []
    seen: set[int] = set()
    for alias in candidates:
        if (
            id(alias) in seen
            or alias is mutated
            or not isinstance(alias, torch.Tensor)
            or isinstance(alias, torch.nn.Parameter)
        ):
            continue
        # Gated read: rejects foreign-session stamps and storage-rebound
        # objects, so a stale index entry can never act as a live alias.
        alias_label = get_tensor_label(alias)
        if alias_label is None or alias_label == exclude_label:
            continue
        if _alias_shares_elements(mutated, alias, storage_ptr):
            seen.add(id(alias))
            found.append((alias, alias_label))
    return found


def data_alias_write_reaches_base(trace: Any, receiver: torch.Tensor) -> bool:
    """Return whether an in-place write through a ``.data`` alias will advance a labeled base.

    The ``.data`` getter is captured as ``detach``; a receiver mutation through
    it reaches the base tensor in the graph only through storage-alias label
    propagation, which runs in default captures and only when a labeled alias
    shares the receiver's elements (the base itself).

    Parameters
    ----------
    trace
        Active capture trace.
    receiver
        The ``.data`` alias about to be written in place.

    Returns
    -------
    bool
        True when the propagation will connect the write to a labeled base.
    """

    if not view_mutation_propagation_enabled(trace):
        return False
    with _state.pause_logging(), internal_scalar_read():
        return bool(overlapping_labeled_aliases(receiver, get_tensor_label(receiver)))
