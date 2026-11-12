"""The thin record: one plain admitted call's commit as a deferred journal row.

Stage 1 of the thin record path. Today's commit for an exhaustive output builds an
``ExhaustiveOpDraft``, freezes it through ``commit_op`` into a copy of the fields
dictionary held by a ``_PendingRow``, and wraps the handle in a ``LiveOpView``. For
a call the thin emit serves (one bare tensor output in a plain session) the fields
dictionary is the call's own (``_build_shared_fields_dict`` returns a fresh one and
nothing writes to it after the commit), so :func:`commit_thin` binds it to the
pending row by reference and appends the handle, with no draft, no copy and no view.

The pending row is the protocol every handle reader already uses (``name in row``,
``row[name]``, ``row.read(handle, name)``, ``row.settle(handle)``, ``row.labels``),
so a trace may hold thin rows and today's rows side by side in op order. Its
settle writes the row and residual with the same builders as today's
(``_PendingRow._build`` and ``write_draft_row``).
"""

from __future__ import annotations

import time
import weakref
from typing import TYPE_CHECKING, Any

from ..._capture.plan import RetentionKind
from ..._capture.session import CaptureSession
from ..._data_classes.op import _OP_STORE_LAYOUT
from ..._ir._row_journal import _new_object, _RowOpRecord
from ..._utils.display import _record_phase_timing
from ._ops_row_journal import _fill_items, _PendingRow, _row_journal_for

if TYPE_CHECKING:
    import torch

    from ..._data_classes.trace import Trace

__all__ = ("commit_thin",)


def commit_thin(
    trace: Trace,
    tensor: torch.Tensor,
    fields: dict[str, Any],
    copies: tuple[tuple[Any, ...], dict[str, Any]],
    module_stack: Any,
    requires_grad: bool,
) -> _RowOpRecord:
    """Save the output, seal its edge truth and append its deferred row; return the handle.

    The steps ``_make_layer_log_entry`` takes for a call in a plain session, in its
    order: the save decision, the retention escrow, the saved value copy, the
    parent edge truth, then the commit. ``copies`` is ``(arg_copies,
    kwarg_copies)``; ``requires_grad`` is the output's, read by the caller inside
    its internal-read block.
    """

    trace_dict = trace.__dict__
    fire_results = tuple(fields.pop("fire_results")) if "fire_results" in fields else ()
    raw_index: int = fields["raw_index"]
    layer_nums_to_save: Any = trace._layer_nums_to_save
    save_this_activation = (layer_nums_to_save == "all") or (raw_index in layer_nums_to_save)
    if "_capture_session" in trace_dict:
        _escrow_candidate(trace_dict["_capture_session"], fields, tensor, save_this_activation)
    if save_this_activation:
        save_started = time.perf_counter()
        try:
            _save_activation_fields(
                trace, fields, tensor, copies[0], copies[1], trace._options_activation_transform
            )
        finally:
            _record_phase_timing(
                trace, "clone_save_activation_fields", time.perf_counter() - save_started
            )
    _seal_parent_edge_truth(trace_dict, fields)
    handle = _append_row(trace, fields, tensor, (fire_results, module_stack), requires_grad)
    grad_fn_handle = fields["grad_fn_handle"]
    if grad_fn_handle is not None:
        trace.capture_events.grad_fn_handles_by_label_raw[fields["_label_raw"]] = grad_fn_handle
    return handle


def _escrow_candidate(
    session: Any, fields: dict[str, Any], tensor: Any, save_this_activation: bool
) -> None:
    """Offer the output to the session's retention escrow, as ``_make_layer_log_entry`` does."""

    if not (type(session) is CaptureSession or isinstance(session, CaptureSession)):
        return
    profile = session.plan.retention_profile
    if (
        profile.activation_kind is RetentionKind.ACTIVATION and not save_this_activation
    ) or profile.gradient_kind is RetentionKind.GRADIENT_REFERENCE:
        session.escrow_candidate(
            fields["raw_index"],
            tensor,
            retain_activation=not save_this_activation,
            raw_label=fields.get("_label_raw"),
        )


def _append_row(
    trace: Trace,
    fields: dict[str, Any],
    tensor: Any,
    side: tuple[tuple[Any, ...], Any],
    requires_grad: bool,
) -> _RowOpRecord:
    """Bind ``fields`` to a pending row by reference and append its handle to the journal.

    The values ``commit_row_record`` snapshots for a first output with an empty call
    box, without the dictionary copy: the parents list and the output versions are
    copied (the commit's own copies), the parent events and the journal are bound now.
    """

    pending = _PendingRow()
    parents = fields["parents"] = list(fields["parents"])
    events = trace.capture_events
    found = tuple(map(events.live_index.by_raw_label.get, parents))
    if None in found:
        found = tuple(map(events.live_index.require_event, parents))
    journal = events.row_journal
    if journal is None:
        journal = _row_journal_for(trace, _OP_STORE_LAYOUT)
    versions = fields["out_versions_by_child"]
    fields["out_versions_by_child"] = dict(versions) if versions else {}
    pending.trace_ref = weakref.ref(trace)
    pending.fields = fields
    pending.fire_results, pending.module_stack = side
    pending.params = _param_refs_from_fields(fields) if fields["parent_params"] else ()
    pending.requires_grad = bool(requires_grad)
    pending.device = str(tensor.device)
    pending.handle_id = str(id(tensor))
    pending._options_save_mode = trace._options_save_mode
    pending.events = dict(zip(parents, found, strict=True))
    pending.journal = journal
    edge_uses = fields["_edge_uses"]
    pending.edge_uses = tuple(edge_uses) if edge_uses else _build_edge_use_records
    pending.side = pending.output = pending.built = pending.function = None
    pending.labels = (fields["_label_raw"], parents)
    _fill_items(pending, fields)
    handle = _new_object(_RowOpRecord)
    handle._row = handle._rest = handle._record = handle._parents = None
    handle._journal = None  # type: ignore[assignment]  # bound when the pending row settles
    handle._patch = {}
    handle._pending = pending
    handle.seq = 0
    events.append_thin(handle, *pending.labels)
    return handle


# Late imports: the split modules of this package import one another (cycles), so this
# module imports them once every definition above exists.
from ._ops_activations import _save_activation_fields  # noqa: E402
from ._ops_arguments import _build_edge_use_records  # noqa: E402
from ._ops_finalize import _seal_parent_edge_truth  # noqa: E402
from ._ops_freeze_refs import _param_refs_from_fields  # noqa: E402
