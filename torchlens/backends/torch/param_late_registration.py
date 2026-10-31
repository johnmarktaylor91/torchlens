"""Late registration of parameters the captured forward itself created (lazy modules).

Model preparation stamps every parameter of the module tree before the forward runs
(``model_prep._create_session_param_logs``). A lazy module built INSIDE forward
(``self.lazy = nn.LazyLinear(4)`` followed by ``self.lazy(x)``) is not in that tree yet:
torch materializes its pending ``UninitializedParameter`` slots in place from the lazy
pre-hook, fills them in ``reset_parameters``, and the module body then consumes them
through ``F.linear`` with no session provenance. This is the parameter twin of
``buffer_writes.resolve_or_late_index_buffer_address``: reached only when a wrapped call
consumes a Parameter the wrapper watched torch materialize from a pending slot during this
forward, it registers the parameter through the same session registry preparation uses
(``set_param_meta``, ``trace.param_logs``, the session inventory) and discloses the creation
as a ``parameter_created_in_forward`` capture advisory.

The carve-out is narrow on purpose (r77 F1: a fresh in-forward ``nn.Parameter`` or a foreign
model's parameter must never launder provenance). A parameter registers here only when ALL
of these hold at consumption time:

* the wrapper saw torch's materialization plumbing run on the object while it was still a
  pending lazy slot (the ``.data`` reads and setter of ``UninitializedTensorMixin.materialize``,
  recorded by object identity in ``WrapperRuntimeWorkspace.materialized_lazy_slots``);
* the object is now a materialized ``nn.Parameter`` outside the session registry;
* it is a live slot of the captured model's own module tree (``model.named_parameters``,
  walked only on this rare path).

A Parameter attached to the tree inside forward from a captured value
(``self.p = nn.Parameter(y.detach())``) never was a pending slot, so it keeps the break
marker and the honest ``source_provenance`` failure, exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

from ..._capture_honesty import (
    ADVISORIES_ANNOTATIONS_KEY,
    ADVISORY_PARAMETER_CREATED_IN_FORWARD,
    append_capture_advisory,
)
from ..._capture_state_helpers import _is_uninitialized_param
from ..._data_classes.param import Param, ParamAccessor
from ..._quantities import Bytes
from ..._utils.hashing import make_random_barcode
from ..._utils.tensor_utils import get_memory_amount
from ._tl import set_param_meta
from .buffer_writes import peek_buffer_write_tracker
from .param_mutation import is_unregistered_parameter

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = (
    "late_register_forward_created_parameter",
    "materialized_lazy_slots",
    "note_lazy_slot_materializing",
)


def materialized_lazy_slots(trace: Trace) -> Any:
    """Return the forward's watched pending-slot set, or ``None`` when none was seen.

    Parameters
    ----------
    trace:
        Active capture trace.

    Returns
    -------
    Any
        The ``WeakIdKeyDictionary`` of pending slots torch materialized through the
        wrapper during this forward, or ``None`` (the hot-path fast exit).
    """

    return trace._wrapper_runtime_ws.materialized_lazy_slots


def note_lazy_slot_materializing(trace: Trace, slot: torch.Tensor) -> None:
    """Record that torch's materialization plumbing ran on a pending lazy slot.

    Called by the wrapper for every passed-through call whose receiver is an
    ``UninitializedParameter`` or ``UninitializedBuffer``. Identity-keyed: torch's
    ``materialize`` swaps the object's ``__class__`` in place, so the same key later
    names the materialized Parameter.

    Parameters
    ----------
    trace:
        Active capture trace.
    slot:
        The pending receiver of the passed-through call.
    """

    workspace = trace._wrapper_runtime_ws
    slots = workspace.materialized_lazy_slots
    if slots is None:
        slots = WeakIdKeyDictionary()
        workspace.materialized_lazy_slots = slots
    slots[slot] = True


def _live_tree_addresses(model: nn.Module, param: torch.Tensor) -> list[str]:
    """Return the dotted addresses at which ``param`` is a live slot of ``model``."""

    return [
        name
        for name, candidate in model.named_parameters(remove_duplicate=False)
        if candidate is param
    ]


def _force_requires_grad_like_prep(trace: Trace, param: nn.Parameter) -> None:
    """Force ``requires_grad`` on exactly the terms preparation uses for a prepped Parameter."""

    if (
        param.is_leaf
        and not getattr(trace, "backward_ready", False)
        and (torch.is_floating_point(param) or torch.is_complex(param))
    ):
        param.requires_grad = True


def _has_optimizer_like_prep(trace: Trace) -> bool | None:
    """Mirror the prep inventory's optimizer column: ``None`` without an optimizer, else False.

    A parameter created inside forward cannot belong to an optimizer built before it.
    """

    for existing in trace.param_logs:
        return None if existing.has_optimizer is None else False
    return None


def _record_creation_advisory(trace: Trace, entry: str) -> None:
    """Add ``entry`` to the forward's ``parameter_created_in_forward`` advisory row.

    The lazy module's parameters register at different wrapped calls (the weight at its
    ``kaiming_uniform_``, the bias at its ``uniform_``), so the first registration writes
    the row and later ones extend it, keeping one row per kind and forward.
    """

    annotations = getattr(trace, "annotations", None)
    rows = annotations.get(ADVISORIES_ANNOTATIONS_KEY) if isinstance(annotations, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("kind") == ADVISORY_PARAMETER_CREATED_IN_FORWARD:
                row["count"] = int(row.get("count", 0)) + 1
                row["message"] = f"{row.get('message', '')}; {entry}"
                return
    append_capture_advisory(trace, ADVISORY_PARAMETER_CREATED_IN_FORWARD, [entry])


def late_register_forward_created_parameter(trace: Trace, param: torch.Tensor) -> str | None:
    """Register a Parameter torch materialized from a pending slot during this forward.

    Parameters
    ----------
    trace:
        Active capture trace.
    param:
        A ``nn.Parameter`` argument of a wrapped call that the wrapper watched torch
        materialize from a pending lazy slot earlier in this forward.

    Returns
    -------
    str | None
        The registered parameter address, or ``None`` when the object is not a live slot
        of the captured model's own tree (or is already registered); nothing is stamped
        in that case and the consuming op keeps its unattributed-argument marker.
    """

    slots = materialized_lazy_slots(trace)
    if slots is None or param not in slots:
        return None
    if _is_uninitialized_param(param):
        return None
    # The slot is consumed once: a later look-up is the registered parameter itself.
    slots.pop(param, None)
    if not isinstance(param, nn.Parameter) or not is_unregistered_parameter(trace, param):
        return None
    tracker = peek_buffer_write_tracker(trace)
    model = tracker.model_ref() if tracker is not None else None
    if model is None:
        return None
    addresses = _live_tree_addresses(model, param)
    if not addresses:
        return None

    address = addresses[0]
    module_address, _, name = address.rpartition(".")
    requires_grad_before = param.requires_grad
    _force_requires_grad_like_prep(trace, param)
    barcode = make_random_barcode()
    set_param_meta(
        param, barcode=barcode, address=address, requires_grad_before=requires_grad_before
    )
    param_log = Param(
        module_address=module_address or "self",
        name=name,
        shape=tuple(param.shape),
        dtype=param.dtype,
        num_params=param.numel(),
        param_memory=Bytes(get_memory_amount(param)),
        trainable=requires_grad_before,
        address=address,
        barcode=barcode,
        has_optimizer=_has_optimizer_like_prep(trace),
    )
    param_log._param_ref = param
    # Pending when the slot entered the session (creation time here, prep time for a
    # prep-registered lazy slot): the materialization writes that follow create the
    # parameter's first value instead of mutating a prepared one, so they keep the
    # plain-capture treatment, and step 15 reads the live geometry.
    param_log._lazy_at_prep = True  # type: ignore[attr-defined]
    for alias in addresses[1:]:
        param_log.all_addresses.append(alias)
        alias_module = alias.rpartition(".")[0] or "self"
        if alias_module not in param_log.all_module_addresses:
            param_log.all_module_addresses.append(alias_module)

    existing: dict[str, Param] = dict(trace.param_logs._dict)
    displaced = existing.get(address)
    existing[address] = param_log
    trace.param_logs = ParamAccessor(existing)
    trace._param_log_by_pid[id(param)] = address
    trace._session_param_inventory.append(param)

    detail = "lazy slot created and materialized inside forward"
    if displaced is not None:
        detail += "; it replaces the pending slot registered at preparation"
    _record_creation_advisory(trace, f"{address} ({detail})")
    return address
