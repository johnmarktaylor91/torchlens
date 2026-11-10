"""Direct payload path for a fresh plain output under the default save options.

:func:`save_fresh_plain_activation` saves the activation of one function output
when every check the general path (``_ops_activations._save_activation_fields``)
runs is known to be a no-op or to have a known result for this output, and
records exactly what the general path records: the same payload (an eager clone
or the deferred clone-on-write alias, chosen by the same gate), the same
save-budget ledger state and refusals, the same identity-dedup entry and
ordinal, and the same field values in the same order. Anything else takes the
general path.

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
* the output is a plain strided ``torch.Tensor`` (not a parameter, subclass,
  lazy or meta tensor) with no TorchLens metadata (no ``_tl`` namespace): it
  carries no capture label to copy onto the payload and no same-object live
  value to clear, i.e. it is a fresh output of the call.

The source's version is read under ``pause_logging`` with the copy, as the
deferred alias's own version read already is: it is an attribute read that runs
no operation, so the internal-read marker the general path wraps it in has
nothing to mark.

The payload (the deferred alias or the eager copy) comes from
``_ops_fresh_alias.fresh_payload`` and the save-budget admission and
reconciliation from ``_ops_fresh_budget``; each module names the general-path
checks it skips and why they are settled for this output.
"""

from __future__ import annotations

import weakref
from math import prod
from typing import TYPE_CHECKING, Any

import torch

from ..._data_classes._op_dedup import _IdentityDedupCache
from ..._state import pause_logging
from ..._utils._tensor_label import mark_detached_saved_activation
from ._ops_fresh_alias import fresh_payload
from ._ops_fresh_budget import admit_fresh, commit_fresh

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = ("save_fresh_plain_activation", "fresh_plain_payload_eligible")

_TENSOR = torch.Tensor
_STRIDED = torch.strided


def fresh_plain_payload_eligible(
    trace: Trace, fields_dict: dict[str, Any], t: Any, activation_transform: Any
) -> _IdentityDedupCache | None:
    """Return the trace's identity cache when ``t`` may take the direct path, else ``None``."""

    try:
        if (
            activation_transform is not None
            or type(t) is not _TENSOR
            or trace._options_save_mode != "copy"
            or trace._options_save_args
            or fields_dict["output_device"] != "same"
            or trace._out_writer is not None
            or trace._out_sink is not None
            or trace._out_dedup_mode != "identity"
            or t.is_meta
            or t.layout is not _STRIDED
            or getattr(t, "_tl", None) is not None
        ):
            return None
        cache = trace._out_identity_cache
    except AttributeError:
        return None
    if type(cache) is not _IdentityDedupCache:
        return None
    key = id(t)
    if key in cache and cache[key][0]() is t:
        return None
    return cache


def save_fresh_plain_activation(
    trace: Trace, fields_dict: dict[str, Any], t: torch.Tensor, cache: _IdentityDedupCache
) -> None:
    """Save ``t``'s activation into ``fields_dict`` on the direct path.

    ``cache`` is the identity cache :func:`fresh_plain_payload_eligible` returned.
    """

    label: str = fields_dict["_layer_label_raw"]
    shape = tuple(t.shape)
    dtype = t.dtype
    num_bytes = int(prod(shape) * dtype.itemsize)
    device = t.device
    budget = trace._save_budget_accountant
    admitted = None if budget is None else admit_fresh(budget, device, num_bytes, fields_dict)
    with pause_logging():
        detach = fields_dict["detach_saved_activations"]
        raw_out, identity, requires_grad = fresh_payload(t, detach, num_bytes, admitted is not None)
        try:
            # ``tensor_version_or_none``: an exact ``torch.Tensor``'s version is an int,
            # and an inference tensor's read raises.
            version: int | None = t._version
        except (RuntimeError, AttributeError, NotImplementedError, TypeError):
            version = None
    fields_dict["shape"] = shape
    fields_dict["dtype"] = dtype
    fields_dict["activation_memory"] = num_bytes
    cache[id(t)] = (weakref.ref(t), label, raw_out, version, cache.next_ordinal())
    if requires_grad:
        # Marks only a payload TorchLens severed from a connected source.
        mark_detached_saved_activation(t, raw_out, label)
    fields_dict["out"] = raw_out
    fields_dict["transformed_out"] = None
    fields_dict["transformed_out_shape"] = None
    fields_dict["transformed_out_dtype"] = None
    fields_dict["transformed_activation_memory"] = None
    fields_dict["has_saved_activation"] = True
    if budget is not None and admitted is not None:
        commit_fresh(budget, admitted, identity, device, fields_dict)
    fields_dict["saved_args"] = None
    fields_dict["saved_kwargs"] = None
