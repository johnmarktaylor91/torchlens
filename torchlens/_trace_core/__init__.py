"""Private per-trace semantic store substrate (trace_core_design.md).

One ``TraceCore`` per trace: numpy-backed typed columns, per-trace intern
pools, one canonical edge-occurrence table with CSR indexes, group tables,
an identity-preserving payload arena, sparse versioned mutation overlays,
and the facade cache. Nothing in this package is public API; public classes
(`Trace`, `Op`, ...) present rows from here as facades.

Zero production consumers until the Op seam; the substrate ships with
its own unit suite (tests/test_trace_core_substrate.py) and executable
prototypes of every hard seam (copy, fork, payload identity, hydration).
"""

from .columns import ColumnBuilder, FrozenColumn
from .core import TraceCore
from .op_store import DetachedOpStore, OpRowStore, OpStoreLayout
from .overlays import RowOverlay, Transaction
from .payloads import PayloadArena
from .pools import InternPool
from .relations import EdgeTable

__all__ = [
    "ColumnBuilder",
    "DetachedOpStore",
    "EdgeTable",
    "FrozenColumn",
    "InternPool",
    "OpRowStore",
    "OpStoreLayout",
    "PayloadArena",
    "RowOverlay",
    "TraceCore",
    "Transaction",
]
