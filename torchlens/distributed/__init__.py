"""Distributed capture opt-in: arming, group lifetime identity, and evidence.

``torchlens.distributed`` is a power-user submodule (imported as
``tl.distributed``; deliberately not in the top-level ``__all__``, matching
``tl.debug``). It owns the C0 correlation/evidence layer of the merge-ranks
tier:

* :func:`arm` -- the explicit process-start opt-in that installs the
  group-lifecycle wraps, verifies the five-namespace collective recognizer,
  and stamps this rank's install-epoch record. Required spelling for MPMD
  programs and any program whose ranks enter capture at different times.
* :class:`GroupLifecycleLedger` / :func:`audit_membership_lineages` -- the
  per-rank lifecycle evidence and the pure PRE-JOIN membership-lineage audit
  the merge engine runs over it.
* Typed refusal vocabulary: ``ambiguous_group_lifetime``,
  ``group_lifetime_evidence_conflict``, ``uncaptured_collective_op``.
* :func:`has_vetted_snapshot` -- the read-only capability probe for whether
  this torch build's collective dispatcher matches a censused row in
  :data:`torchlens.distributed._recognizer.VETTED_NAMESPACE_SNAPSHOTS`. Never
  raises; use it to know in advance whether :func:`arm` can succeed, without
  changing arm()'s own fail-closed behavior on an unvetted build.

Capture of explicit ``torch.distributed`` collectives as boundary nodes is
armed automatically at capture entry for already-initialized SPMD processes;
see :func:`torchlens.distributed.arm` for when the explicit spelling is
required.
"""

from ._audit import (
    GROUP_LIFETIME_EVIDENCE_CONFLICT,
    MembershipLineageVerdict as MembershipLineageVerdict,
    audit_membership_lineages,
)
from ._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    LineageEntry as LineageEntry,
    LineageVector as LineageVector,
    membership_digest_for_ranks,
)
from ._lifecycle import (
    AMBIGUOUS_GROUP_LIFETIME,
    AmbiguousGroupLifetimeError,
    ArmingRecord,
    GroupIdentity,
    arm,
    auto_arm_degradation,
    disarm,
    is_armed,
)
from ._recognizer import (
    COLLECTIVE_NAMESPACES,
    UNCAPTURED_COLLECTIVE_OP,
    CollectiveRecognizer as CollectiveRecognizer,
    UncapturedCollectiveOpError,
    derive_collective_recognizer,
    has_vetted_snapshot,
)

__all__ = [
    "AMBIGUOUS_GROUP_LIFETIME",
    "COLLECTIVE_NAMESPACES",
    "GROUP_LIFETIME_EVIDENCE_CONFLICT",
    "UNCAPTURED_COLLECTIVE_OP",
    "AmbiguousGroupLifetimeError",
    "ArmingRecord",
    "GroupIdentity",
    "GroupLifecycleEvent",
    "GroupLifecycleLedger",
    "UncapturedCollectiveOpError",
    "arm",
    "audit_membership_lineages",
    "auto_arm_degradation",
    "derive_collective_recognizer",
    "disarm",
    "has_vetted_snapshot",
    "is_armed",
    "membership_digest_for_ranks",
]
