"""Exhaustive commit into the row-backed capture journal (capture C2b).

The exhaustive commit writes the op's store cells into the journal's
capture-time ``OpRowStore`` (C2a's cells, written by :func:`write_draft_row`)
and returns a private row handle instead of freezing an ``OpRecord``. The values
the cells cannot carry exactly are snapshotted into the handle's residual here,
at commit, from the same builders the eager freeze uses
(``_op_record_from_log``), so the record built later equals the eager one.

This module imports no module of the package's largest import cycle (``ops``,
inside it, calls :func:`commit_row_record`). The two inputs that come from cycle
modules, the op store layout and the edge-use records, are read by the caller
(``ExhaustiveOpDraft.freeze`` in ``ops.py``) and passed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._ir._row_journal import _new_object, _RowJournal, _RowOpRecord, _RowResidual
from ..._trace_core.op_store import OpRowStore
from . import _ops_row_cells
from ._ops_freeze_refs import (
    _exhaustive_capture_policy,
    _exhaustive_freeze_refs,
    _exhaustive_output_ref,
    _param_refs_from_fields,
    _resolve_call_function_ref,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.events import FunctionCallRef
    from ..._trace_core.op_store import OpStoreLayout
    from ...intervention.types import EdgeUseRecord
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


def _residual_from_draft(
    draft: ExhaustiveOpDraft,
    function: FunctionCallRef,
    edge_uses: tuple[EdgeUseRecord, ...],
) -> _RowResidual:
    """Snapshot the record values the cells cannot carry, read now as the freeze reads them.

    Sets every slot ``_RowResidual.__init__`` sets, without its keyword dict and call.
    """

    trace, fields_dict = draft.trace, draft.fields_dict
    tensor_ref, frames, transformed_ref, backend_semantics = _exhaustive_freeze_refs(
        fields_dict, draft.tensor, draft.module_stack
    )
    rest = _new_object(_RowResidual)
    rest.kind = "source" if fields_dict["is_input"] or fields_dict["is_buffer"] else "op"
    rest.output = _exhaustive_output_ref(fields_dict, tensor_ref, transformed_ref)
    rest.function = function
    rest.module_stack = frames
    rest.edge_uses = edge_uses
    rest.params = _param_refs_from_fields(fields_dict)
    rest.parent_params = tuple(fields_dict["parent_params"])
    rest.backend_semantics = backend_semantics
    rest.policy = _exhaustive_capture_policy(trace, fields_dict)
    rest.is_scalar_bool = fields_dict["is_scalar_bool"]
    rest.fire_results = draft.fire_results
    return rest


def commit_row_record(
    draft: ExhaustiveOpDraft,
    layout: OpStoreLayout,
    edge_uses: tuple[EdgeUseRecord, ...],
) -> _RowOpRecord:
    """Write one exhaustive draft's row and return its journal handle.

    Reads what ``_op_record_from_log`` reads off the draft. The function-call
    ref is resolved once through the call box (sibling outputs share it as the
    freeze shares it), then the row is written from the same draft and that ref.

    Parameters
    ----------
    draft
        The final exhaustive draft (``ExhaustiveOpDraft``).
    layout
        The op store layout (``_data_classes.op._OP_STORE_LAYOUT``); the
        capture journal's row store is created with it on the first commit.
    edge_uses
        The op's edge-use records: the draft's ``_edge_uses`` when set, else
        the records ``_ops_arguments._build_edge_use_records`` builds from its
        parent argument positions. Snapshotted into the residual.

    Returns
    -------
    _RowOpRecord
        The journal handle for the new row.
    """

    trace, fields_dict = draft.trace, draft.fields_dict
    journal = _row_journal_for(trace, layout)
    box = draft.call_ref_box if draft.call_ref_box is not None else []
    function = _resolve_call_function_ref(fields_dict, box)
    rest = _residual_from_draft(draft, function, edge_uses)
    row = _ops_row_cells.write_draft_row(
        journal.store, trace, fields_dict, draft.fire_results, function
    )
    return journal.handle(row, rest)
