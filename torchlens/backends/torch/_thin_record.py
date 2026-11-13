"""The thin record: one plain admitted call's commit as a deferred journal row.

Today's commit for an exhaustive output builds an ``ExhaustiveOpDraft``, freezes
it through ``commit_op`` into a copy of the fields dictionary held by a
``_PendingRow``, and wraps the handle in a ``LiveOpView``. For a call the thin emit
serves (one bare tensor output in a plain session) the fields dictionary is the
call's own (``_build_shared_fields_dict`` returns a fresh one and nothing writes to
it after the commit), so the commit :func:`make_commit` binds holds it in the
pending row by reference and appends the handle, with no draft, no copy and no view.
The settle families the thin builders leave out are filled by the row on first need
(:class:`_ThinRow`), which then becomes a plain pending row.

The pending row is the protocol every handle reader already uses (``name in row``,
``row[name]``, ``row.read(handle, name)``, ``row.settle(handle)``, ``row.labels``),
so a trace may hold thin rows and today's rows side by side in op order. Its
settle writes the row and residual with the same builders as today's
(``_PendingRow._build`` and ``write_draft_row``).

This module imports nothing from the capture import cycle: the helpers it calls
come in through :class:`._thin_emit.ThinHelpers`.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from ..._capture.plan import RetentionKind
from ..._ir._row_journal import _new_object, _RowOpRecord
from ..._utils.display import _record_phase_timing
from ._ops_freeze_refs import _param_refs_from_fields
from ._ops_row_journal import _fill_items, _PendingRow, _row_journal_for
from ._thin_settle import fill_settle_fields

if TYPE_CHECKING:
    import torch

    from ..._data_classes.trace import Trace
    from ._thin_emit import ThinHelpers

__all__ = ("make_commit",)

_Commit = Callable[..., _RowOpRecord]


def make_commit(helpers: ThinHelpers) -> _Commit:
    """Return the thin commit bound to the cycle helpers in ``helpers``."""

    save_fields = helpers.save_activation_fields
    seal_edges = helpers.seal_parent_edge_truth
    session_type = helpers.capture_session_type
    append = _appender(helpers.build_edge_use_records, helpers.op_store_layout, helpers.describe)

    def commit_thin(
        trace: Trace,
        tensor: torch.Tensor,
        fields: dict[str, Any],
        copies: tuple[tuple[Any, ...], dict[str, Any]],
        facts: tuple[Any, bool, tuple[Any, ...]],
    ) -> _RowOpRecord:
        """Save the output, seal its edge truth and append its deferred row; return the handle.

        The steps ``_make_layer_log_entry`` takes for a call in a plain session, in its
        order: the save decision, the retention escrow, the saved value copy, the
        parent edge truth, then the commit. ``copies`` is ``(arg_copies,
        kwarg_copies)``; ``facts`` is ``(module_stack, requires_grad, fire_results)``, the
        output's ``requires_grad`` read by the caller inside its internal-read block and the
        call's fire results (the thin emit keeps them out of ``fields``).
        """

        trace_dict = trace.__dict__
        raw_index: int = fields["raw_index"]
        layer_nums_to_save: Any = trace._layer_nums_to_save
        save_this = (layer_nums_to_save == "all") or (raw_index in layer_nums_to_save)
        if "_capture_session" in trace_dict:
            session = trace_dict["_capture_session"]
            if type(session) is session_type or isinstance(session, session_type):
                _escrow_candidate(session, fields, tensor, save_this)
        if save_this:
            save_started = time.perf_counter()
            try:
                save_fields(
                    trace, fields, tensor, copies[0], copies[1], trace._options_activation_transform
                )
            finally:
                _record_phase_timing(
                    trace, "clone_save_activation_fields", time.perf_counter() - save_started
                )
        seal_edges(trace_dict, fields)
        module_stack, requires_grad, fire_results = facts
        handle = append(trace, fields, tensor, (fire_results, module_stack), requires_grad)
        grad_fn_handle = fields["grad_fn_handle"]
        if grad_fn_handle is not None:
            trace.capture_events.grad_fn_handles_by_label_raw[fields["_label_raw"]] = grad_fn_handle
        return handle

    return commit_thin


def _escrow_candidate(
    session: Any, fields: dict[str, Any], tensor: Any, save_this_activation: bool
) -> None:
    """Offer the output to the session's retention escrow, as ``_make_layer_log_entry`` does."""

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


class _ThinRow(_PendingRow):
    """A thin call's pending row: today's pending row, with the settle families filled late.

    The thin emit leaves out of ``fields`` what no call-time step reads: the defaults
    tables, the callable's descriptors and ``func_non_tensor_args``
    (:func:`._thin_settle.fill_settle_fields`). The first lazy ref or settle fills them,
    with the same builders and tables today's emit uses at the call, then the row
    becomes a plain ``_PendingRow`` (the same slots), so later reads pay nothing extra.
    The reads that need no settle family never fill: ``parents`` and
    ``parent_arg_positions`` (``_PendingRow.read``) and the side refs a module exit reads
    (the thin emit writes ``transformed_out``, the one default they read). ``device`` and ``handle_id`` are kept as the
    device object and the integer id until the fill. ``describe`` is a class attribute
    of the per-commit subclass :func:`_appender` makes.
    """

    __slots__ = ()
    describe: ClassVar[Callable[..., Any]]

    def _fill(self) -> None:
        """Fill the settle families and the commit's strings, then become a plain pending row."""

        fill_settle_fields(self.fields, self.describe)
        self.device = str(self.device)
        self.handle_id = str(self.handle_id)
        self.__class__ = _PendingRow

    def function_ref(self) -> Any:
        """Return the op's function-call ref (``_PendingRow.function_ref``), filled first."""

        self._fill()
        return _PendingRow.function_ref(self)

    def output_ref(self) -> Any:
        """Return the op's output ref (``_PendingRow.output_ref``), filled first."""

        self._fill()
        return _PendingRow.output_ref(self)

    def settle(self, handle: _RowOpRecord) -> None:
        """Write the row and residual once (``_PendingRow.settle``), filled first."""

        self._fill()
        _PendingRow.settle(self, handle)


def _appender(edge_use_builder: Any, layout: Any, describe: Any) -> _Commit:
    """Return the row append bound to the edge-use builder, the op store layout and ``describe``."""

    row_type = type(
        "_BoundThinRow", (_ThinRow,), {"__slots__": (), "describe": staticmethod(describe)}
    )

    def append_row(
        trace: Trace,
        fields: dict[str, Any],
        tensor: Any,
        side: tuple[tuple[Any, ...], Any],
        requires_grad: bool,
    ) -> _RowOpRecord:
        """Hold ``fields`` in a pending row by reference and append its handle to the journal.

        The values ``commit_row_record`` snapshots for a first output with an empty
        call box, without the dictionary copy: the parents list and the output
        versions are copied (the commit's own copies), the parent events and the
        journal are bound now.
        """

        pending = row_type()
        parents = fields["parents"] = list(fields["parents"])
        events = trace.capture_events
        found = tuple(map(events.live_index.by_raw_label.get, parents))
        if None in found:
            found = tuple(map(events.live_index.require_event, parents))
        journal = events.row_journal
        if journal is None:
            journal = _row_journal_for(trace, layout)
        versions = fields["out_versions_by_child"]
        fields["out_versions_by_child"] = dict(versions) if versions else {}
        pending.trace_ref = weakref.ref(trace)
        pending.fields = fields
        pending.fire_results, pending.module_stack = side
        pending.params = _param_refs_from_fields(fields) if fields["parent_params"] else ()
        pending.requires_grad = bool(requires_grad)
        pending.device = tensor.device
        pending.handle_id = id(tensor)  # type: ignore[assignment]  # a str from the fill on
        pending._options_save_mode = trace._options_save_mode
        pending.events = dict(zip(parents, found, strict=True))
        pending.journal = journal
        edge_uses = fields["_edge_uses"]
        pending.edge_uses = tuple(edge_uses) if edge_uses else edge_use_builder
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

    return append_row
