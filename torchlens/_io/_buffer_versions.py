"""Each buffer's version op labels, read once the op rows are written.

Step 16 runs this scan when it seals the module facts, so the deferred
module build reads plain labels; ``rebuild_buffer_accessor`` runs it itself
when it is given none.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from .._data_classes._op_column_scan import ops_with
from .._trace_core.record_refs import name_owner_by_row
from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def seal_buffer_versions(trace: Trace) -> dict[str, list[str]]:
    """Name gradient-record owners and return each buffer's version op labels.

    Parameters
    ----------
    trace:
        Trace whose params and gradient-record or buffer ops are scanned.

    Returns
    -------
    dict[str, list[str]]
        Buffer address to the labels of its version ops, in op order.
    """

    trace_ref = weakref.ref(trace)
    for param in trace.__dict__.get("_param_logs") or ():
        name_owner_by_row(trace_ref, param)
    versions: dict[str, list[str]] = {}
    # Only an op with gradient records or a buffer flag has work here.
    for entry in ops_with(trace, "_grad_records", "is_buffer"):
        name_owner_by_row(trace_ref, entry)
        if record_member(entry, "is_buffer", False):
            address = entry.buffer_address
            if address is not None:
                versions.setdefault(address, []).append(entry.label)
    return versions
