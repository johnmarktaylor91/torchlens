"""Exhaustive commit into the row-backed capture journal (capture C2b).

The exhaustive commit writes the op's store cells into the journal's
capture-time ``OpRowStore`` (C2a's cells, written by :func:`write_draft_row`)
and returns a private row handle instead of freezing an ``OpRecord``. The values
the cells cannot carry exactly are snapshotted into the handle's residual here,
at commit, from the same builders the eager freeze uses
(``_op_record_from_log``), so the record built later equals the eager one.

The commit is deferred: :func:`commit_row_record` keeps a shallow snapshot of the
fields, the parent events the live index holds and the three live-tensor facts
(``requires_grad``, device, ``id``) in a :class:`_PendingRow`, answers the hot
reads from it, and writes the row and residual from it on the first other read.

This module imports no module of the package's largest import cycle (``ops``,
inside it, calls :func:`commit_row_record`). The two inputs that come from cycle
modules, the op store layout and the edge-use records, are read by the caller
(``ExhaustiveOpDraft.freeze`` in ``ops.py``) and passed in.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, cast

from ..._ir._row_journal import (
    _new_object,
    _parent_edges,
    _RowJournal,
    _RowOpRecord,
    _RowResidual,
)
from ..._trace_core.op_store import OpRowStore
from ..._utils._internal_read import internal_scalar_read
from . import _ops_row_cells
from ._ops_freeze_refs import (
    _copy_parent_arg_positions,
    _exhaustive_capture_policy,
    _exhaustive_output_ref,
    _exhaustive_side_refs,
    _function_call_ref_from_fields,
    _param_refs_from_fields,
    _resolve_call_function_ref,
    _tensor_ref_from_facts,
)
from ._ops_row_cells import _ANCESTRY_DEFAULT

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._trace_core.op_store import OpStoreLayout
    from .ops import ExhaustiveOpDraft

__all__: tuple[str, ...] = ()


def _row_journal_for(trace: Trace, layout: OpStoreLayout) -> _RowJournal:
    """Return the capture journal's row store, creating it with ``layout`` on the first commit."""

    events = trace.capture_events
    journal = events.row_journal
    if journal is None:
        journal = _RowJournal(OpRowStore(layout))
        events.row_journal = journal
    return journal


class _SnapshotIndex:
    """The parent events the commit saw, in place of ``trace.capture_events.live_index``.

    The deferred row build resolves parent labels here, so it reads the events
    the eager commit read even after an amendment replaced one in the live
    index. ``capture_events`` and ``live_index`` answer the index itself, so it
    also stands in for the trace argument of the edge-use builder.
    """

    __slots__ = ("events", "trace_ref")

    def __init__(self, events: dict[str, Any], trace_ref: Any) -> None:
        """Bind the parent label -> event snapshot and the trace's weak reference."""

        self.events = events
        self.trace_ref = trace_ref

    @property
    def capture_events(self) -> _SnapshotIndex:
        """Answer the index itself (the edge-use builder reads ``trace.capture_events``)."""

        return self

    @property
    def live_index(self) -> _SnapshotIndex:
        """Answer the index itself (the builders read ``capture_events.live_index``)."""

        return self

    def require_event(self, label_raw: str) -> Any:
        """Return the snapshot event for ``label_raw``, else the live index's answer."""

        event = self.events.get(label_raw)
        if event is None:
            return self.trace_ref().capture_events.live_index.require_event(label_raw)
        return event


class _PendingRow(dict):  # type: ignore[type-arg]
    """A deferred exhaustive commit: the hot reads as dict items, the rest on settle.

    The items are the values the row and residual would answer for the names
    read on every op; :meth:`read` adds the lazily built ones; :meth:`settle`
    writes the row and residual from the snapshot once and binds them to a handle.
    """

    __slots__ = (
        "trace_ref",
        "fields",
        "fire_results",
        "module_stack",
        "box",
        "function",
        "params",
        "requires_grad",
        "device",
        "handle_id",
        "_options_save_mode",
        "index",
        "layout",
        "edge_uses",
        "labels",
        "side",
        "output",
        "built",
    )

    def function_ref(self) -> Any:
        """Return the op's function-call ref, resolving a first output's ref once."""

        function = self.function
        if function is None:
            function = self.function = _function_call_ref_from_fields(self.fields)
            box = self.box
            if box and box[0] is self:
                box[0] = function
        return function

    def side_refs(self) -> Any:
        """Return (module frames, transformed-output ref, backend semantics), built once."""

        side = self.side
        if side is None:
            side = self.side = _exhaustive_side_refs(self.fields, self.module_stack)
        return side

    def output_ref(self) -> Any:
        """Return the op's output ref, built once from the commit-time tensor facts."""

        output = self.output
        if output is None:
            fields = self.fields
            tensor_ref = _tensor_ref_from_facts(
                fields, self.requires_grad, self.device, self.handle_id
            )
            output = self.output = _exhaustive_output_ref(fields, tensor_ref, self.side_refs()[1])
        return output

    def read(self, handle: _RowOpRecord, name: str) -> Any:
        """Answer a fast read the items lack: build a lazy value, else settle and re-read."""

        fields = self.fields
        if name == "parents":
            edges = _parent_edges(fields["parents"], fields["parent_arg_positions"])
            self[name] = handle._parents = edges
            return edges
        if name == "parent_arg_positions":
            value = self[name] = _copy_parent_arg_positions(fields["parent_arg_positions"])
            return value
        if name == "output":
            value = self[name] = self.output_ref()
            return value
        if name == "backend_semantics":
            value = self[name] = self.side_refs()[2]
            return value
        if name == "function":
            value = self[name] = self.function_ref()
            return value
        self.settle(handle)
        return getattr(handle, name)

    def settle(self, handle: _RowOpRecord) -> None:
        """Write the row and residual once, then bind them to ``handle``."""

        built = self.built
        if built is None:
            built = self.built = self._build()
        handle._journal, handle._row, handle._rest = built
        handle._pending = None

    def _build(self) -> tuple[_RowJournal, list[Any], _RowResidual]:
        """Write the row and residual the eager commit wrote, from the snapshot."""

        trace = self.trace_ref()
        fields = self.fields
        journal = _row_journal_for(trace, self.layout)
        function = self.function_ref()
        edge_uses = self.edge_uses
        if type(edge_uses) is not tuple:
            edge_uses = tuple(
                edge_uses(
                    self.index,
                    fields["parent_arg_positions"],
                    fields["_label_raw"],
                    fields["func_call_id"],
                    func_name=fields.get("func_name"),
                )
            )
        frames, _, semantics = self.side_refs()
        rest = _new_object(_RowResidual)
        rest.kind = self["kind"]
        rest.output = self.output_ref()
        rest.function = function
        rest.module_stack = frames
        rest.edge_uses = edge_uses
        rest.params = self.params if self.params is not None else _param_refs_from_fields(fields)
        rest.parent_params = tuple(fields["parent_params"])
        rest.backend_semantics = semantics
        rest.policy = _exhaustive_capture_policy(cast(Any, self), fields)
        rest.is_scalar_bool = fields["is_scalar_bool"]
        rest.fire_results = self.fire_results
        store = journal.store
        row = store._rows[
            _ops_row_cells.write_draft_row(
                store, trace, fields, self.fire_results, function, self.index.require_event
            )
        ]
        if "parent_arg_positions" in self:
            # The copy the handle already answered is the row's cell, as at an eager commit.
            row[journal.fids["parent_arg_positions"]] = self["parent_arg_positions"]
        return journal, row, rest


def commit_row_record(
    draft: ExhaustiveOpDraft,
    layout: OpStoreLayout,
    edge_uses: Any,
) -> _RowOpRecord:
    """Commit one exhaustive draft as a deferred row handle (row written on first need).

    Reads now what ``_op_record_from_log`` reads off live state: a shallow copy of
    the fields, the output's ``requires_grad`` (under the internal read marker),
    device and ``id``, the parameter refs, the save mode and the parent events.
    A sibling output resolves its function-call ref now through the shared call
    box; a first output parks its pending row there (a sibling commit forces it).
    ``layout`` is the op store layout; ``edge_uses`` is the draft's edge-use
    tuple, or ``_ops_arguments._build_edge_use_records`` to build it on settle.
    """

    trace, fields = draft.trace, dict(draft.fields_dict)
    tensor = draft.tensor
    with internal_scalar_read():
        requires_grad = bool(tensor.requires_grad)
    trace_ref = weakref.ref(trace)
    parents = fields["parents"] = list(fields["parents"])
    by_label = trace.capture_events.live_index.by_raw_label
    found = tuple(map(by_label.get, parents))
    if None in found:
        for label in parents:
            trace.capture_events.live_index.require_event(label)
    versions = fields["out_versions_by_child"]
    fields["out_versions_by_child"] = dict(versions) if versions else {}
    pending = _PendingRow()
    pending.trace_ref = trace_ref
    pending.fields = fields
    pending.fire_results = draft.fire_results
    pending.module_stack = draft.module_stack
    pending.params = _param_refs_from_fields(fields) if fields["parent_params"] else ()
    pending.requires_grad = requires_grad
    pending.device = str(tensor.device)
    pending.handle_id = str(id(tensor))
    pending._options_save_mode = trace._options_save_mode
    pending.index = _SnapshotIndex(dict(zip(parents, found)), trace_ref)
    pending.layout = layout
    pending.edge_uses = edge_uses
    pending.side = pending.output = pending.built = pending.function = None
    box = draft.call_ref_box
    pending.box = box
    if box:
        if box[0].__class__ is _PendingRow:
            box[0].function_ref()
        pending.function = _resolve_call_function_ref(fields, box)
    elif box is not None:
        box += (pending,)
    label = fields["_label_raw"]
    pending.labels = (label, parents)
    _fill_thin_items(pending, fields, label)
    handle = _new_object(_RowOpRecord)
    handle._journal = handle._row = handle._rest = handle._record = handle._parents = None
    handle._patch = {}
    handle._pending = pending
    handle.seq = 0
    return handle


def _fill_thin_items(pending: _PendingRow, fields: dict[str, Any], label: str) -> None:
    """Store the hot reads' values as ``pending``'s items, as the row would answer them.

    The ancestry sets become frozensets here, so a later in-place change to a
    field set reaches neither the answers nor the row written from the snapshot.
    """

    pending["label_raw"] = label
    pending["func_call_id"] = fields["func_call_id"]
    pending["kind"] = "source" if fields["is_input"] or fields["is_buffer"] else "op"
    inputs = fields["input_ancestors"] = frozenset(fields["input_ancestors"])
    internal = fields["internal_source_ancestors"] = frozenset(fields["internal_source_ancestors"])
    roots = fields["root_ancestors"] = frozenset(fields["root_ancestors"])
    has_internal = fields["has_internal_source_ancestor"]
    if not (inputs or internal or roots or has_internal):
        inputs = frozenset(_ANCESTRY_DEFAULT.input_ancestors)
        internal = frozenset(_ANCESTRY_DEFAULT.internal_source_ancestors)
        roots = frozenset(_ANCESTRY_DEFAULT.root_ancestors)
        has_internal = _ANCESTRY_DEFAULT.has_internal_source_ancestor
    pending["input_ancestors"] = inputs
    pending["internal_source_ancestors"] = internal
    pending["root_ancestors"] = roots
    pending["has_internal_source_ancestor"] = has_internal
    semantics = fields["backend_semantics"] if "backend_semantics" in fields else None
    if semantics is not None:
        pending["backend_semantics"] = semantics
