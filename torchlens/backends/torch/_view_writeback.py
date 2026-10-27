"""Bind ``x[idx] += y``'s write-back ``__setitem__`` to the base's own version.

``wrappers._propagate_mutation_label_to_storage_aliases`` advances a base
tensor's label to an in-place op on one of its views. For the write-back of an
augmented slice assignment that made one label stand for two values; this
module remembers the base's label from before the view mutation and rebinds
the write-back's base slot to it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.weak import WeakIdKeyDictionary

from ... import _state
from ._tl import (
    active_label_session_token,
    get_tensor_label,
    session_storage_alias_candidates,
    set_tensor_label,
    storage_alias_index_key,
)
from .completeness_witness import internal_scalar_read

_VIEW_MUTATION_PRIOR_LABELS: WeakIdKeyDictionary = WeakIdKeyDictionary()
"""Base alias -> (session token, view-mutation label, the base's label before it).

Written by ``_propagate_mutation_label_to_storage_aliases`` when a view
mutation advances an overlapping base. Read only by
``bind_setitem_writeback_base_to_prior_label``.
"""


def view_mutation_propagation_enabled(trace: Any) -> bool:
    """Return whether this capture advances base aliases on a view mutation.

    Descriptor/grad-bound captures keep the historical topology (see the SCOPE
    note in ``wrappers._propagate_mutation_label_to_storage_aliases``).
    """

    return not (
        getattr(trace, "intervention_ready", False)
        or getattr(trace, "backward_ready", False)
        or getattr(trace, "save_grads", None) not in (None, False)
    )


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

    from .wrappers import _storage_overlap_byte_interval, _strided_views_share_storage_elements

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
        try:
            if (
                storage_alias_index_key(alias.untyped_storage()) != storage_ptr
                or alias.device != mutated.device
            ):
                continue
            # Element-exact where provable (round-31 M4): interleaved
            # element-disjoint views must not inherit the mutation edge.
            shares_elements = _strided_views_share_storage_elements(mutated, alias)
        except Exception:
            continue
        if shares_elements:
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


def record_view_mutation_prior_label(
    alias: torch.Tensor, mutation_label: str, prior_label: str
) -> None:
    """Remember ``alias``'s label from before a view mutation advanced it to ``mutation_label``."""

    _VIEW_MUTATION_PRIOR_LABELS[alias] = (active_label_session_token(), mutation_label, prior_label)


def bind_setitem_writeback_base_to_prior_label(args: tuple[Any, ...]) -> None:
    """Give ``x[idx] += y``'s write-back ``__setitem__`` its base's own parent label.

    Python runs ``x[idx] += y`` as ``v = x[idx]; v += y; x[idx] = v``. The
    in-place ``v += y`` advances the base ``x`` to the view's label, so the
    write-back ``__setitem__(x, idx, v)`` would bind ONE label to two slots
    holding two different values (the full base and the view). When the value
    is that same view of ``idx``'s destination region and the base's label
    came from that view mutation, the base slot is bound to the base's label
    from before the mutation instead. This is exact: the write-back overwrites
    precisely the region the mutation wrote, so
    ``setitem(base_before, idx, v)`` equals the result, and the consumed
    base's actual content still reaches replay through the per-child version
    snapshot.

    Parameters
    ----------
    args
        Positional arguments of the ``__setitem__`` call, before it runs.
    """

    if len(args) < 3:
        return
    base, index, value = args[0], args[1], args[2]
    if not isinstance(base, torch.Tensor) or not isinstance(value, torch.Tensor):
        return
    if base is value:
        return
    shared_label = get_tensor_label(value)
    if shared_label is None or get_tensor_label(base) != shared_label:
        return
    entry = _VIEW_MUTATION_PRIOR_LABELS.get(base)
    if entry is None:
        return
    session_token, mutation_label, prior_label = entry
    if session_token != active_label_session_token() or mutation_label != shared_label:
        return
    if _value_is_the_destination_view(base, index, value):
        set_tensor_label(base, prior_label)


def _value_is_the_destination_view(base: torch.Tensor, index: Any, value: torch.Tensor) -> bool:
    """Return whether ``value`` is exactly the view ``base[index]`` (same storage region).

    The storage, offset, shape and stride reads are TorchLens's own bookkeeping,
    so they run under ``pause_logging`` and ``internal_scalar_read``; an index
    that cannot be re-applied reads as not the same region.
    """

    with _state.pause_logging(), internal_scalar_read():
        try:
            destination = base[index]
            return bool(
                storage_alias_index_key(destination.untyped_storage())
                == storage_alias_index_key(value.untyped_storage())
                and destination.storage_offset() == value.storage_offset()
                and tuple(destination.shape) == tuple(value.shape)
                and tuple(destination.stride()) == tuple(value.stride())
            )
        except (IndexError, RuntimeError, TypeError, ValueError):
            return False
