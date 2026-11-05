"""Exhaustive commit into the row-backed capture journal (capture C2b).

The exhaustive commit writes the op's store cells into the journal's
capture-time ``OpRowStore`` (C2a's :func:`op_row_cells` / :func:`write_op_row`)
and returns a private row handle instead of freezing an ``OpRecord``. The values
the cells cannot carry exactly are snapshotted into the handle's residual here,
at commit, from the same builders the eager freeze uses
(``_op_record_from_log``), so the record built later equals the eager one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._data_classes.op import _OP_STORE_LAYOUT
from ..._ir._row_journal import _RowJournal, _RowOpRecord, _RowResidual
from ..._trace_core.op_store import OpRowStore

# Module import, not ``from ._ops_row_cells import ...``: a cold import of
# ``_ops_row_cells`` reaches this module (through ``ops``) while it is still loading.
from . import _ops_row_cells
from ._ops_arguments import _build_edge_use_records
from ._ops_capture_records import (
    _exhaustive_capture_policy,
    _exhaustive_freeze_refs,
    _exhaustive_output_ref,
    _param_refs_from_fields,
    _resolve_call_function_ref,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.refs import FunctionCallRef
    from .ops import ExhaustiveOpDraft

__all__: tuple[str, ...] = ()


def _row_journal_for(trace: Trace) -> _RowJournal:
    """Return the capture journal's row store, creating it on the first commit."""

    events = trace.capture_events
    journal = events.row_journal
    if journal is None:
        journal = _RowJournal(OpRowStore(_OP_STORE_LAYOUT))
        events.row_journal = journal
    return journal


def _residual_from_draft(draft: ExhaustiveOpDraft, function: FunctionCallRef) -> _RowResidual:
    """Snapshot the record values the cells cannot carry, read now as the freeze reads them."""

    trace, fields_dict = draft.trace, draft.fields_dict
    tensor_ref, frames, transformed_ref, backend_semantics = _exhaustive_freeze_refs(
        fields_dict, draft.tensor, draft.module_stack
    )
    edge_uses = tuple(
        fields_dict["_edge_uses"]
        or _build_edge_use_records(
            trace,
            fields_dict["parent_arg_positions"],
            fields_dict["_label_raw"],
            fields_dict["func_call_id"],
            func_name=fields_dict.get("func_name"),
        )
    )
    return _RowResidual(
        kind="source" if fields_dict["is_input"] or fields_dict["is_buffer"] else "op",
        output=_exhaustive_output_ref(fields_dict, tensor_ref, transformed_ref),
        function=function,
        module_stack=frames,
        edge_uses=edge_uses,
        params=_param_refs_from_fields(fields_dict),
        parent_params=tuple(fields_dict["parent_params"]),
        backend_semantics=backend_semantics,
        policy=_exhaustive_capture_policy(trace, fields_dict),
        is_scalar_bool=fields_dict["is_scalar_bool"],
        fire_results=draft.fire_results,
    )


def commit_row_record(draft: ExhaustiveOpDraft) -> _RowOpRecord:
    """Write one exhaustive draft's row and return its journal handle.

    Reads what ``_op_record_from_log`` reads off the draft. The function-call
    ref is resolved once through the call box (sibling outputs share it as the
    freeze shares it), then the cells are written from the same draft.
    """

    trace, fields_dict = draft.trace, draft.fields_dict
    journal = _row_journal_for(trace)
    box = draft.call_ref_box if draft.call_ref_box is not None else []
    function = _resolve_call_function_ref(fields_dict, box)
    rest = _residual_from_draft(draft, function)
    cells = _ops_row_cells.op_row_cells(trace, fields_dict, draft.fire_results, box)
    return journal.handle(_ops_row_cells.write_op_row(journal.store, cells), rest)
