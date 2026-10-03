"""One-backward attribution reads (workstream F04; M(reads) memo).

One backward pass, a table of "how much does each site matter": the read
seeds ``autograd.grad`` at GradientEdges resolved from EXISTING op fields,
runs inside a mandatory suppression context, and returns a :class:`ReadTable`
with honesty as columns. Every spelling here is DOCUMENTED-UNSTABLE pending
the naming sprint; the semantics and the stable error codes are the contract.

Doc of record: ``docs/reference/onebackward_reads.md``.
"""

from ._accessor import ReadEdgeIndex, SiteEdge, read_edge_index
from ._edge_plumbing import (
    GradInputUse,
    GradInputUseMap,
    PassOriginStamp,
    backward_pass_origins,
    mint_grad_input_use_map,
    stamp_backward_pass_origin,
)
from ._errors import ReadError, ReadInternalError
from ._frozen import DEFAULT, FrozenPlan, resolve_frozen
from ._read import METHODS, REDUCTIONS, read
from ._suppress import read_suppressed
from ._table import ReadRow, ReadTable, TableProvenance, load_read_table
from ._targets import SeedTarget, seed

__all__ = [
    "DEFAULT",
    "FrozenPlan",
    "GradInputUse",
    "GradInputUseMap",
    "METHODS",
    "PassOriginStamp",
    "REDUCTIONS",
    "ReadEdgeIndex",
    "ReadError",
    "ReadInternalError",
    "ReadRow",
    "ReadTable",
    "SeedTarget",
    "SiteEdge",
    "TableProvenance",
    "backward_pass_origins",
    "load_read_table",
    "mint_grad_input_use_map",
    "read",
    "read_edge_index",
    "read_suppressed",
    "resolve_frozen",
    "seed",
    "stamp_backward_pass_origin",
]
