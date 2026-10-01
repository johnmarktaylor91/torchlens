# distributed/ - Implementation Guide

C0 correlation/evidence layer for merge-ranks: arming, group-lifetime identity, and
the membership-lineage audit. Imported lazily as `tl.distributed` and deliberately
not in the top-level `__all__` (matching `tl.debug`). Explicit `torch.distributed`
collectives inside a traced forward become first-class boundary nodes; the boundary
op capture itself lives in `backends/torch/collectives.py`.

## _lifecycle.py
- `arm()` is the explicit process-start opt-in: installs group-lifecycle wraps
  (`_install_lifecycle_wraps`), verifies the recognizer fail-closed, and stamps this
  rank's `ArmingRecord` (install epoch). REQUIRED for MPMD / spawn-rank programs.
- `maybe_auto_arm()` is the lazy capture-entry path for already-initialized SPMD
  processes; `disarm()` / `is_armed()` / `armed_state()` manage the singleton.
- `resolve_group_identity(group)` mints `GroupIdentity` (membership digest +
  lifetime ordinal); `next_seq(identity, channel)` issues the per-(uid, channel)
  correlation counters.
- Unprovable pre-arming group lifetime raises `AmbiguousGroupLifetimeError`
  (code constant `AMBIGUOUS_GROUP_LIFETIME`).

## _ledger.py
- `GroupLifecycleLedger` is the per-rank lifecycle evidence, serialized under
  `trace.annotations["distributed"]`; rows are `GroupLifecycleEvent`.
- `LineageEntry` / `LineageVector` model per-membership create/destroy history;
  `membership_digest_for_ranks()` is the canonical digest.
- Parsers validate against closed vocabularies (`_require_vocabulary`); never
  accept free-form strings.

## _recognizer.py
- `derive_collective_recognizer()` builds the five-namespace `CollectiveRecognizer`
  (`COLLECTIVE_NAMESPACES`) at arm time and checks set-equality against the runtime
  plus a dispatcher schema scan; any unrecognized collective raises
  `UncapturedCollectiveOpError` (`UNCAPTURED_COLLECTIVE_OP`). Fail-closed: an
  unknown collective refuses arming rather than passing uncaptured.
- `VETTED_NAMESPACE_SNAPSHOTS` is a reviewed census, one row per torch build that
  has had a capture-fidelity review (currently only `"torch-2.13"`); extending it
  is a reviewed change, never a routine compat patch. `has_vetted_snapshot()` is
  the companion read-only capability probe (layer 1 only, never raises): it
  answers whether THIS runtime matches a row, so callers (including tests) can
  know in advance whether `arm()` can succeed instead of branching on the raised
  exception. It is exported as `tl.distributed.has_vetted_snapshot` and documented
  there; it lives here (not `torchlens/utils/_torch_compat.py`) because it checks
  against our own census table, not a generic torch API capability. F1 ruling
  2026-10-01: on an unvetted torch, the gloo/distributed/merge-ranks test suites
  assert the typed `UncapturedCollectiveOpError` fail-closed refusal (gated
  `skipif(has_vetted_snapshot())`) instead of running the full arming tests
  (gated `skipif(not has_vetted_snapshot())`); adding a census row for a new torch
  minor is a separate, reviewed capture-fidelity sprint, not a side effect of a CI
  fix.

## _audit.py
- `audit_membership_lineages()` is the pure PRE-JOIN audit the C1 merge engine runs
  over per-rank ledgers; returns `MembershipLineageVerdict`s. Conflicting evidence
  is `GROUP_LIFETIME_EVIDENCE_CONFLICT`, a typed finding, never a silent join.

## _dtensor.py
- `dtensor_dual_geometry(value)` extracts per-site logical/local dual geometry for
  DTensor findings (used by the tier-(a) refusal report); returns `None` for
  non-DTensor values.

## Gotchas
- Arming relaxes NO tier-(a) refusal: DTensor/TP/FSDP2/PP capture stays refused.
- `wildcard_recv_unsupported` (`WILDCARD_RECV_UNSUPPORTED`) is raised from
  `backends/torch/collectives.py`, not from this package.
- Decisions here are evidence-recording only; nothing in this package may guess an
  unobserved completion or membership -- unknowns become typed refusals/findings.
- Distributed rank processes are the one sanctioned exception to the
  no-child-process capture guard; do not widen that carve-out here.
