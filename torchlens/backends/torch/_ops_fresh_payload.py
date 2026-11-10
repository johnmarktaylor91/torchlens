"""Direct payload path for a fresh plain output under the default save options.

:func:`save_fresh_plain_activation` saves the activation of one function output
when every check the general path (``_ops_activations._save_activation_fields``)
runs is known to be a no-op or to have a known result for this output, and
records exactly what the general path records: the same payload (an eager clone
or the deferred clone-on-write alias, chosen by the same gate), the same
save-budget ledger state and refusals, the same identity-dedup entry and
ordinal, and the same field values in the same order. Anything else returns
``False`` and the caller runs the general path.

Eligibility, with the option or fact that makes each skipped step a no-op:

* ``save_mode="copy"``, the default: the effective mode is ``"copy"`` for every
  op (only ``"reference"`` depends on the op), so no reference stamp is written.
* no activation transform: the raw payload is stored and no transform runs.
* ``output_device`` is ``"same"``: the copy stays on the source device, so the
  retention device is the source's and no move follows.
* ``save_args`` off: no argument snapshot, and identity dedup is enabled.
* no streaming writer and no payload sink on the trace: nothing is streamed.
* identity dedup mode (the default): the pre-copy probe and the post-copy insert
  both key ``id(t)``; with no live entry for this exact tensor the probe misses
  and the insert takes the next ordinal, which is done here directly.
* the output is a plain ``torch.Tensor`` (not a parameter, subclass, lazy or
  meta tensor) with no TorchLens metadata: it carries no capture label to copy
  onto the payload and no same-object live value to clear, i.e. it is a fresh
  output of the call.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import torch

from ..._data_classes._op_dedup import _IdentityDedupCache
from ..._save_budget import _RetainedStorageEntry
from ..._state import pause_logging
from ..._utils import tensor_utils as _tensor_utils
from ..._utils._tensor_label import get_tensor_meta, mark_detached_saved_activation
from ..._utils._torch_compat import tensor_version_or_none
from ..._utils._torch_symbols import tensor_read
from ..._utils.tensor_utils import (
    _copy_tensor_payload,
    _try_defer_payload_alias,
    get_memory_amount_from_metadata,
)
from ._internal_read import internal_scalar_read

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = ("save_fresh_plain_activation", "fresh_plain_payload_eligible")

_TENSOR = torch.Tensor


def fresh_plain_payload_eligible(
    trace: Trace, fields_dict: dict[str, Any], t: Any, activation_transform: Any
) -> _IdentityDedupCache | None:
    """Return the trace's identity cache when ``t`` may take the direct path, else ``None``."""

    if (
        activation_transform is not None
        or type(t) is not _TENSOR
        or trace._options_save_mode != "copy"
        or trace._options_save_args
        or fields_dict["output_device"] != "same"
        or getattr(trace, "_out_writer", None) is not None
        or getattr(trace, "_out_sink", None) is not None
        or getattr(trace, "_out_dedup_mode", "identity") != "identity"
        or t.is_meta
        or get_tensor_meta(t) is not None
    ):
        return None
    cache = getattr(trace, "_out_identity_cache", None)
    if type(cache) is not _IdentityDedupCache:
        return None
    cached = cache.get(id(t))
    if cached is not None and cached[0]() is t:
        return None
    return cache


def _clone_gate(t: torch.Tensor, detach: bool) -> torch.Tensor:
    """Return the payload ``_clone_tensor_payload`` returns for a plain unlabelled tensor.

    The same decision as its ``"copy"`` mode with no move: the deferred alias
    while the payload window is armed and the tensor is eligible, otherwise the
    eager copy. Callers hold ``pause_logging``.
    """

    payload = None
    if _tensor_utils._DEFER_WINDOW_DEPTH:
        if detach or not t.requires_grad or not torch.is_grad_enabled():
            payload = _try_defer_payload_alias(t)
        elif _tensor_utils._DEFER_GRAD_ENABLED:
            payload = _try_defer_payload_alias(t, graph_connected=True)
    if payload is None:
        payload = _copy_tensor_payload(t, detach_tensor=detach, save_mode="copy")
    return payload


def _storage_identity(raw_out: torch.Tensor) -> tuple[Any, ...] | None:
    """Return the strided storage identity ``SaveBudget`` charges, or ``None`` if unreadable."""

    try:
        storage = tensor_read(raw_out, "untyped_storage")(raw_out)
        return (str(raw_out.device), int(storage.data_ptr()), int(storage.nbytes()))
    except Exception:
        return None


def _commit_fresh(budget: Any, reservation: Any, raw_out: torch.Tensor, identity: Any) -> None:
    """Reconcile one admission whose payload is a new storage of exactly the admitted size.

    ``SaveBudget.commit`` would subtract the estimate and charge the same bytes
    back under a new storage entry, so the totals and the refusal outcome are
    unchanged and only the entry and its release watcher are added. Any other
    payload (a storage already charged, or a size that differs) goes through
    ``commit``.
    """

    ledger = budget._ledger_for(raw_out.device)
    if (
        identity is None
        or identity[2] != reservation.num_bytes
        or identity in ledger.retained_storage
        or raw_out.device != reservation.device
    ):
        budget.commit(reservation, (raw_out, None))
        return
    ledger.retained_storage[identity] = _RetainedStorageEntry(
        physical_bytes=reservation.num_bytes, live_refs=1
    )
    budget._watch_payload(raw_out, identity[0], identity)


def save_fresh_plain_activation(
    trace: Trace, fields_dict: dict[str, Any], t: torch.Tensor, cache: _IdentityDedupCache
) -> None:
    """Save ``t``'s activation into ``fields_dict`` on the direct path.

    ``cache`` is the identity cache :func:`fresh_plain_payload_eligible` returned.
    """

    label = fields_dict.get("_layer_label_raw")
    shape = tuple(t.shape)
    dtype = t.dtype
    num_bytes = get_memory_amount_from_metadata(t, shape, dtype)
    budget = getattr(trace, "_save_budget_accountant", None)
    reservation = None
    if budget is not None:
        reservation = budget.admit(
            str(label or fields_dict.get("_label_raw") or "<unlabeled>"), t.device, num_bytes
        )
    identity = None
    with pause_logging():
        raw_out = _clone_gate(t, fields_dict["detach_saved_activations"])
        if reservation is not None and raw_out.layout is torch.strided:
            identity = _storage_identity(raw_out)
    with internal_scalar_read():
        version = tensor_version_or_none(t)
        source_connected = bool(t.requires_grad and t.grad_fn is not None)
    fields_dict["shape"] = shape
    fields_dict["dtype"] = dtype
    fields_dict["activation_memory"] = num_bytes
    cache[id(t)] = (weakref.ref(t), label, raw_out, version, cache.next_ordinal())
    if source_connected:
        mark_detached_saved_activation(t, raw_out, label)
    fields_dict["out"] = raw_out
    fields_dict["transformed_out"] = None
    fields_dict["transformed_out_shape"] = None
    fields_dict["transformed_out_dtype"] = None
    fields_dict["transformed_activation_memory"] = None
    fields_dict["has_saved_activation"] = True
    if reservation is not None:
        _commit_fresh(budget, reservation, raw_out, identity)
    fields_dict["saved_args"] = None
    fields_dict["saved_kwargs"] = None
