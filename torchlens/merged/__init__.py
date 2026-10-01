"""Cross-rank trace merging (merge-ranks tier (c), rung C1).

``torchlens.merged`` stitches N rank-local captures -- live traces or saved
rank cores -- into one :class:`MergedTrace` at their explicit collective
boundaries. The rank cores remain the single truth (P1: the merge never
rewrites a per-rank claim); correlation is by counting, verified by value
(P3: witness digests only ever DEMOTE); every ambiguity is a typed finding or
refusal (P4).

Entry points::

    merged = tl.merge_ranks([trace_or_path, ...])   # raises on conflict
    report = tl.merge_report([trace_or_path, ...])  # graph-free diagnostic
    merged.save("run.merged.tlspec")
    merged = torchlens.merged.load("run.merged.tlspec")  # full rederivation

C1 scope: explicit in-forward c10d python collectives, SPMD, symmetric-issue
channels. Point-to-point/pipeline pairing is rung C3 and DTensor topologies
are rung C2; both refuse typed here. Merged replay does not exist.

The frozen enums, error codes, and finding kinds are documented exhaustively
in ``docs/reference/merged_trace_contract.md`` under the ordered-list-equality
doc-vs-enum gate.
"""

from ._artifact import load_merged as load, save_merged, tree_hash
from ._engine import (
    JoinRecord as JoinRecord,
    MergeDerivation as MergeDerivation,
    PerRankRef as PerRankRef,
    derive_merge,
)
from ._enums import (
    MERGE_FINDING_KINDS,
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
)
from ._errors import (
    MergeConflictError,
    MergedArtifactError,
    MergedFinding,
    MergedSurfaceUnsupportedError,
    MergedTraceError,
    MergeInputError,
)
from ._presenter import (
    CollectiveJoin as CollectiveJoin,
    MergedTrace,
    MergeReport,
    merge_ranks,
    merge_report,
)

__all__ = [
    "MERGE_FINDING_KINDS",
    "BoundaryConsistency",
    "MergeAlignment",
    "MergeConflictError",
    "MergeInputError",
    "MergeReport",
    "MergeValueStatus",
    "MergedArtifactError",
    "MergedErrorCode",
    "MergedFinding",
    "MergedSurfaceUnsupportedError",
    "MergedTrace",
    "MergedTraceError",
    "derive_merge",
    "load",
    "merge_ranks",
    "merge_report",
    "save_merged",
    "tree_hash",
]
