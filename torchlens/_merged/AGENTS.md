# merged/ - Implementation Guide

Cross-rank merging (merge-ranks rung C1). `tl.merge_ranks([trace_or_path, ...])`
stitches N rank cores into a `MergedTrace` at explicit collective boundaries;
`tl.merge_report(...)` is the graph-free diagnostic that never raises on conflicts.
Both are top-level lazy names routed here. Contract of record:
`docs/reference/merged_trace_contract.md` (ordered-list-equality doc-vs-enum gate).

## _presenter.py
- `MergedTrace` is a PRESENTER over the rank cores -- never a `Trace`/`Bundle`
  subclass; rank cores stay the single truth and are never rewritten.
- `merge_ranks()` raises `MergeConflictError` on conflict; `merge_report()` returns
  a `MergeReport` instead. `CollectiveJoin` is the public join row.

## _engine.py
- `derive_merge()` is the ONE pure derivation, run at merge time and again verbatim
  at load rederivation. Audit-first: conflicted memberships never join and never
  become presence gaps.
- Alignment is seq-DELTA from each rank's first recorded key; absolute bases are
  never compared. Witness digests only DEMOTE (`witness_consistency` in
  `_witness.py`, which the engine imports and which never imports the engine).
- Outputs: `MergeDerivation`, `JoinRecord`, `PerRankRef`.
- `derive_merge()` is a short orchestrator over named stages, in normative order:
  scope guard, membership authority, lineage audit, group table, channel
  collection, per-channel joining (`_join_channel` -> `_join_key`: relation table,
  c10d and absolute-seq cross-checks, presence gap, witness verdict), declared
  ranks, order check, stored verdicts. The join assembly (anything that builds a
  `PerRankRef`/`JoinRecord`) stays here; the evidence-only stages live in
  `_derive_steps.py`, the per-kind relation table (`_relation_findings`) in
  `_relation_table.py` and the witness verdict in `_witness.py`; neither imports
  the engine. The order check is `_order_graph` + `_has_cycle` (union-find join
  classes, then Kahn) behind the thin `_check_order`.
- `tests/test_merged_derive_steps.py` pins the whole result (digest) or the exact
  refusal of every stage; a change to findings order, messages or codes fails it.

## _derive_steps.py
- The derivation stages that need no join record: `_guard_scope`,
  `check_membership_authority`, `audit_membership_lineages`, `build_group_table`,
  `collect_channel_records`, `align_seq_deltas`, the per-key finding builders
  (`c10d_delta_mismatch`, `absolute_seq_mismatch`, `presence_gap`,
  `value_divergence`), `declared_rank_gap`, `stored_alignment`,
  `stored_value_status`, and `_STRUCTURAL_KINDS`.
- Never imports `_engine.py` (the engine imports it), so the import graph stays
  acyclic; it defines no `__all__` or other mutable module globals.

## _enums.py
- Frozen vocabularies: `MergeAlignment` (stored vs effective), `BoundaryConsistency`
  (totalized derivation), `MergeValueStatus`, `MergedErrorCode`, plus
  `MERGE_FINDING_KINDS` (exported from `__init__.py`). Release-gated against the
  contract doc; adding a member is a reviewed contract diff.

## _errors.py
- Typed family rooted at `MergedTraceError`: `MergeInputError`,
  `MergeConflictError`, `MergedArtifactError`, `MergedSurfaceUnsupportedError`;
  `MergedFinding` is the structured finding record.

## _evidence.py
- `extract_rank_evidence(trace, source)` / `resolve_rank_inputs(inputs)` pull each
  rank's boundary + ledger evidence into `RankEvidence`; malformed inputs refuse
  typed via `MergeInputError`.
- `_validate_boundary` is the ordered list of per-boundary rule groups; the rules
  themselves live in `_boundary_rules.py` and `_boundary_membership.py`. The
  per-core coherence rules (group-rank presence, the `c10d_group_seq` latch,
  install-epoch agreement, unique correlation keys) are `_CoreCoherence`; the
  boundary walk is `_walk_boundaries`, the ledger parse `_parse_ledger`.

## _rules.py (shared identity, boundary and provenance rules)
- Leaf module, raises nothing, imports only `_enums` and the ledger digest. It is
  the ONE home of every predicate both the evidence validator (`_evidence`,
  `_boundary_rules`, `_boundary_membership`) and the merge derivation
  (`_engine`, `_derive_steps`, `_relation_table`, `_witness`) need: identity
  (`correlation_key`, `group_uid`, `channel_key`, `membership_digest_bound`,
  `ranks_outside_membership`, `group_rank_in_range`, `GroupRankPresence`),
  boundary (role vocabularies and selectors `roles_of` / `role_shapes`,
  `empty_for_tensor_kind`, `is_digest_sequence`) and provenance (completion,
  witness-policy, install-epoch and disclosure vocabularies, `GroupSeqLatch`).
- No duplicated predicate: a rule needed on both sides is added HERE and called
  from both, never re-spelled in a caller. The validator and the engine differ
  only where the contract differs (parse requires digest fields to be JSON
  lists and pins `my_group_rank` to the membership position; the engine belt
  accepts any digest sequence and checks range).

## _boundary_rules.py
- Leaf module: the closed boundary vocabularies (`KNOWN_KINDS`, `P2P_KINDS`, the
  reader vocabularies `_evidence` re-exports), the shared `_refuse` constructor,
  and one `check_*` function per rule group (identity, correlation key, scalar
  records, completion events, disclosures, kind-class records, witness policy
  and digests, op-label back-references, lifetime evidence).
- `_boundary_membership.py` holds the group membership and coherence and the
  roles rule groups; it imports `_refuse` and the vocabularies from
  `_boundary_rules.py`, never the reverse.
- Call order is contract: with several defects the earliest group's refusal
  wins. `tests/test_merged_boundary_rules.py` pins every refusal's class, code
  and message; a new rule appends a row there.

## _artifact.py
- `save_merged(merged, path)` writes the merged-directory artifact: canonical-JSON
  descriptor CACHE (`canonical_json_bytes`), per-member tree hashes (`tree_hash`),
  and byte-identical rank cores.
- `load_merged(path)` (exported as `load`) reruns the derivation and requires EXACT
  cache equality; tamper refuses `MergedArtifactError`, never degrades to a gap.
  An unparseable member enters `degradations` and caps effective alignment at
  `partial`.

## Gotchas
- Refused typed in C1: p2p/pipeline boundaries (C3), DTensor dual geometry (C2),
  merged-level selectors, merged replay/runnable export/validate.
- `distributed_witness="payload"` is a typed construction refusal; digest witnesses
  only.
- Never let load "trust" the cached derivation: the cache is a coherence check, the
  rerun is the authority.
