"""Frozen vocabulary for cross-rank merged traces (merge-ranks rung C1).

Every enum here is FROZEN under the ordered-list-equality doc-vs-enum gate in
``docs/reference/merged_trace_contract.md`` (the runnable-contract precedent):
adding, removing, or reordering a member is a reviewed contract change that
must land with the matching doc edit or the gate goes red. No existing
TorchLens vocabulary (``ReadinessStatus``, ``PathFaithfulness``, capture
verdicts, ...) gains a new value or a weaker meaning here; the merge layer
speaks only these enums (design-merge-ranks-c v5, section 3.1).
"""

from __future__ import annotations

from enum import Enum
from typing import Final

__all__ = [
    "BoundaryConsistency",
    "MERGE_FINDING_KINDS",
    "MERGED_DESCRIPTOR_KIND",
    "MERGED_DESCRIPTOR_SCHEMA_VERSION",
    "MERGED_BUNDLE_FORMAT",
    "MERGED_TLSPEC_VERSION",
    "MergeAlignment",
    "MergeValueStatus",
    "MergedErrorCode",
    "REDUCE_OP_KINDS",
    "TENSORLESS_KINDS",
    "WITNESS_IDENTITY_KINDS",
    "WITNESS_NOT_APPLICABLE_KINDS",
    "WITNESS_SLICE_KINDS",
    "WITNESS_VERDICT_BACKENDS",
]

MERGED_DESCRIPTOR_KIND: Final = "cross_rank_merge"
"""Closed descriptor discriminator for ``merge/descriptor.json``."""

MERGED_DESCRIPTOR_SCHEMA_VERSION: Final = 1
"""Frozen merged-descriptor schema version; parse refuses any other value."""

MERGED_BUNDLE_FORMAT: Final = "merged-directory"
"""Root ``bundle_format`` value for merged artifacts.

Deliberate addition to the closed bundle-format vocabulary: runtimes without
the merged loader refuse it typed at the existing closed-vocabulary check in
``torchlens._io.manifest`` -- old readers can never half-load a merged root.
"""

MERGED_TLSPEC_VERSION: Final = 7
"""Portable schema version stamped in merged ROOT manifests only.

The merged root is a different discriminated object from a rank core's
``manifest.json`` (it has no tensor table), so it versions independently of
the trace-bundle ``TLSPEC_VERSION``; member rank cores inside ``members/``
remain ordinary bundles gated by the ordinary version rules.
"""


class MergeAlignment(str, Enum):
    """STRUCTURAL merge verdict; value divergence never changes it (contract 3.3)."""

    ALIGNED = "aligned"
    PARTIAL = "partial"
    CONFLICTED = "conflicted"


class BoundaryConsistency(str, Enum):
    """Per-join witness verdict from the totalized derivation (contract 3.4).

    Kind capability evaluates BEFORE the capture setting: an intrinsically
    unwitnessable kind is ``not_applicable`` at every witness level including
    ``"none"``; an applicable kind with absent or incomplete witnesses is
    ``not_present``; any byte-exact digest mismatch is ``mismatched``.
    """

    ATTESTED = "attested"
    MISMATCHED = "mismatched"
    NOT_APPLICABLE = "not_applicable"
    NOT_PRESENT = "not_present"


class MergeValueStatus(str, Enum):
    """Merge-level value verdict derived from the join ledger (contract 3.5)."""

    DIVERGENT = "divergent"
    ATTESTED_COMPLETE = "attested_complete"
    ATTESTED_PARTIAL = "attested_partial"
    UNWITNESSED = "unwitnessed"


class MergedErrorCode(str, Enum):
    """Frozen machine-readable merge error taxonomy.

    Public code branches on these values (or the structured findings), never
    on exception message text (P4).
    """

    MERGE_INPUT_INVALID = "merge_input_invalid"
    MERGE_SCOPE_UNSUPPORTED = "merge_scope_unsupported"
    GROUP_LIFETIME_EVIDENCE_CONFLICT = "group_lifetime_evidence_conflict"
    MERGE_CONFLICT = "merge_conflict"
    MERGED_SCHEMA_INVALID = "merged_schema_invalid"
    MERGED_DESCRIPTOR_TAMPER = "merged_descriptor_tamper"
    MERGE_RUN_UNSUPPORTED = "merge_run_unsupported"
    MERGED_SELECTOR_UNSUPPORTED = "merged_selector_unsupported"
    MERGED_SURFACE_UNSUPPORTED = "merged_surface_unsupported"
    MERGED_MEMBER_RELEASED = "merged_member_released"


MERGE_FINDING_KINDS: Final[tuple[str, ...]] = (
    "group_lifetime_evidence_conflict",
    "presence_gap",
    "relation_violation",
    "order_contradiction",
    "correlation_delta_mismatch",
    "value_divergence",
    "load_degradation",
)
"""Frozen, ordered merge finding-kind vocabulary (contract-doc gated).

``group_lifetime_evidence_conflict`` / ``relation_violation`` /
``order_contradiction`` / ``correlation_delta_mismatch`` are STRUCTURAL:
any of them makes the merge ``conflicted`` and construction refuses.
``presence_gap`` demotes structure to ``partial``. ``value_divergence``
demotes only the value status (never structural, contract 3.4/3.5). ``load_degradation``
caps the EFFECTIVE alignment at load time (contract 3.3).
"""

# --- Witness capability tables (contract 3.4/3.5) -----------------------------------
#
# Semantic authority is the VERSIONED public-contract capability allowlist,
# not CI probes: torch documents all_reduce results bitwise identical across
# processes, and the copy-semantics kinds are verdict-grade by definitional
# data movement. Unknown kinds/backends are never guessed.

TENSORLESS_KINDS: Final[frozenset[str]] = frozenset(
    {
        "barrier",
        "all_gather_object",
        "broadcast_object_list",
        "gather_object",
        "scatter_object_list",
    }
)
"""Journal-only kinds that legitimately carry ZERO tensor roles.

Mirrors the capture-side ``CollectiveSite.tensorless`` flags exactly: every
other kind executes on at least one tensor argument (the real c10d call would
have raised before the boundary was journaled otherwise), so a tensor-carrying
boundary presenting zero roles is tamper or corruption, never honest evidence.
"""

REDUCE_OP_KINDS: Final[frozenset[str]] = frozenset(
    {"all_reduce", "reduce", "reduce_scatter", "reduce_scatter_tensor"}
)
"""Kinds that always carry a ``ReduceOp`` (capture-side ``has_reduce_op``).

A successful call of these kinds always records a non-null ``reduce_op``
string (``apply_defaults`` materializes ``ReduceOp.SUM`` even when the caller
omitted it), so a null or deleted record is tamper, not honest evidence --
uniform deletion across every rank core used to vacuously satisfy the
reduce-op agreement check exactly like the roles-deletion escape.
"""

WITNESS_IDENTITY_KINDS: Final[frozenset[str]] = frozenset(
    {"all_reduce", "all_gather", "all_gather_into_tensor", "broadcast"}
)
"""Kinds whose destination bytes are identical on every member rank."""

WITNESS_SLICE_KINDS: Final[frozenset[str]] = frozenset({"gather", "scatter", "all_to_all"})
"""Kinds witnessed per slice via group-rank-indexed digest lists."""

WITNESS_NOT_APPLICABLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "barrier",
        "all_gather_object",
        "broadcast_object_list",
        "gather_object",
        "scatter_object_list",
    }
)
"""Kinds with no byte-identity witness.

``reduce``/``reduce_scatter*`` have no cross-rank identity property to
witness; ``all_to_all_single`` exchanges intra-tensor slices that whole-tensor
digests cannot witness; ``barrier`` and the object collectives carry no
digestable tensor payload. Diagnostic recompute stays possible offline; the
verdict is honestly ``not_applicable``.
"""

WITNESS_VERDICT_BACKENDS: Final[frozenset[str]] = frozenset({"gloo", "nccl", "mpi"})
"""Backends whose collective semantics are documented torch API contract.

A group on any other (third-party) backend renders every witness
``not_applicable`` with a disclosure -- unknown semantics are never guessed.
"""
