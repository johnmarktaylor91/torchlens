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
from ._relations_resolve import ANCESTRY_NAMES, ANCESTRY_PARENTS, is_deferred, resolve_fields

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


_SIDE_REF_SLOTS = {"module_stack": 0, "backend_semantics": 2}
"""Lazy names answered from ``side_refs()`` (frames, transformed-output ref, semantics) by slot."""

_OUTPUT_REQUIRES_GRAD: list[Any] = [None, False]
"""(weak ref to an op output, its ``requires_grad``), read in the exhaustive emit.

``_emit_exhaustive_operation_events`` reads the output's ``requires_grad`` inside
its internal-read marker block and parks it here; the commit uses it when the
draft's tensor is that output, else reads it under its own marker.
"""


class _PendingRow(dict[str, Any]):
    """A deferred exhaustive commit: the hot reads as dict items, the rest on settle.

    The items are the values the row and residual would answer for the names
    read on every op; :meth:`read` adds the lazily built ones; :meth:`settle`
    writes the row and residual from the snapshot once and binds them to a handle.
    It also stands in for ``trace.capture_events.live_index`` (and the trace
    argument of the edge-use builder): parent labels resolve to the events the
    commit saw, even after an amendment replaced one in the live index.
    """

    __slots__ = (
        "trace_ref",
        "fields",
        "fire_results",
        "module_stack",
        "function",
        "params",
        "requires_grad",
        "device",
        "handle_id",
        "_options_save_mode",
        "events",
        "journal",
        "edge_uses",
        "labels",
        "side",
        "output",
        "built",
    )
    trace_ref: Any
    fields: dict[str, Any]
    fire_results: Any
    module_stack: Any
    function: Any
    params: Any
    requires_grad: bool
    device: str
    handle_id: str
    _options_save_mode: Any
    events: dict[str, Any]
    journal: _RowJournal
    edge_uses: Any
    labels: tuple[str, list[str]]
    side: Any
    output: Any
    built: Any

    @property
    def capture_events(self) -> _PendingRow:
        """Answer the row itself (the edge-use builder reads ``trace.capture_events``)."""

        return self

    @property
    def live_index(self) -> _PendingRow:
        """Answer the row itself (the builders read ``capture_events.live_index``)."""

        return self

    def require_event(self, label_raw: str) -> Any:
        """Return the commit's event for parent ``label_raw``, else the live index's answer."""

        event = self.events.get(label_raw)
        if event is None:
            return self.trace_ref().capture_events.live_index.require_event(label_raw)
        return event

    def function_ref(self) -> Any:
        """Return the op's function-call ref, built once (a first output's, deferred)."""

        function = self.function
        if function is None:
            function = self.function = _function_call_ref_from_fields(self.fields)
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
        if name in _SIDE_REF_SLOTS:
            value = self[name] = self.side_refs()[_SIDE_REF_SLOTS[name]]
            return value
        if name == "function":
            value = self[name] = self.function_ref()
            return value
        if name in ANCESTRY_NAMES:
            self.resolve_ancestry()  # the answer is an item now, so the read stays deferred
        else:
            self.settle(handle)
        return getattr(handle, name)

    def resolve_ancestry(self) -> None:
        """Build deferred ancestry from the recorded parent records and answer it as items."""

        fields = self.fields
        if is_deferred(fields):
            resolve_fields(fields)
        if "input_ancestors" not in self:
            _fill_ancestry(self, fields)

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
        self.resolve_ancestry()
        journal = self.journal
        function = self.function_ref()
        edge_uses = self.edge_uses
        if type(edge_uses) is not tuple:
            edge_uses = tuple(
                edge_uses(
                    self,
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
        row = store._rows[_ops_row_cells.write_draft_row(store, trace, self, function)]
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
    the fields, the output's ``requires_grad``, device and ``id``, the parameter
    refs, the save mode and the parent events. A sibling output resolves its
    function-call ref through the shared call box; a first output parks its
    pending row there. ``edge_uses`` is the draft's edge-use tuple, or
    ``_ops_arguments._build_edge_use_records`` to build it on settle.
    """

    trace, fields, tensor = draft.trace, dict(draft.fields_dict), draft.tensor
    pending = _PendingRow()
    parked = _OUTPUT_REQUIRES_GRAD
    if parked[0] is not None and parked[0]() is tensor:
        pending.requires_grad = bool(parked[1])
    else:
        with internal_scalar_read():
            pending.requires_grad = bool(tensor.requires_grad)
    parents = fields["parents"] = list(fields["parents"])
    events = trace.capture_events
    found = tuple(map(events.live_index.by_raw_label.get, parents))
    if None in found:
        found = tuple(map(events.live_index.require_event, parents))
    # Bound now: a failed capture's scrub settles rows after dropping capture_events.
    journal = events.row_journal
    if journal is None:
        journal = _row_journal_for(trace, layout)
    versions = fields["out_versions_by_child"]
    fields["out_versions_by_child"] = dict(versions) if versions else {}
    pending.trace_ref = weakref.ref(trace)
    pending.fields = fields
    pending.fire_results = draft.fire_results
    pending.module_stack = draft.module_stack
    pending.params = _param_refs_from_fields(fields) if fields["parent_params"] else ()
    pending.device = str(tensor.device)
    pending.handle_id = str(id(tensor))
    pending._options_save_mode = trace._options_save_mode
    pending.events = dict(zip(parents, found, strict=True))
    pending.journal = journal
    pending.edge_uses = edge_uses
    pending.side = pending.output = pending.built = pending.function = None
    box: Any = draft.call_ref_box
    if box:
        pending.function = _resolve_call_function_ref(fields, box)
    elif box is not None:
        box += (pending,)
    pending.labels = (fields["_label_raw"], parents)
    _fill_items(pending, fields)
    handle = _new_object(_RowOpRecord)
    handle._row = handle._rest = handle._record = handle._parents = None
    handle._journal = None  # type: ignore[assignment]  # bound when the pending row settles
    handle._patch = {}
    handle._pending = pending
    handle.seq = 0
    return handle


def _fill_items(pending: _PendingRow, fields: dict[str, Any]) -> None:
    """Store the hot reads' values as ``pending``'s items, as the row would answer them.

    Deferred ancestry (an op with parents) is not an item until its first read
    (:meth:`_PendingRow.read`) or the settle resolves it.
    """

    pending["label_raw"] = fields["_label_raw"]
    pending["raw_index"] = fields["raw_index"]
    pending["func_call_id"] = fields["func_call_id"]
    pending["kind"] = "source" if fields["is_input"] or fields["is_buffer"] else "op"
    if ANCESTRY_PARENTS not in fields:  # ``is_deferred`` inlined: no call in the commit
        _fill_ancestry(pending, fields)
    if "backend_semantics" in fields:
        semantics = fields["backend_semantics"]
        if semantics is not None:
            pending["backend_semantics"] = semantics


def _fill_ancestry(pending: _PendingRow, fields: dict[str, Any]) -> None:
    """Store the resolved ancestry as ``pending``'s items, as the row would answer them.

    The sets become frozensets here, so a later in-place change to a field set
    reaches neither the answers nor the row written from the snapshot.
    """

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
