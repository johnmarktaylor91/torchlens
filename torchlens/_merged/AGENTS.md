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
  never compared. Witness digests only DEMOTE (`_witness_consistency`).
- Outputs: `MergeDerivation`, `JoinRecord`, `PerRankRef`.

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

## _artifact.py
- `save_merged(merged, path)` writes the merged-directory artifact: canonical-JSON
  descriptor CACHE (`canonical_json_bytes`), per-member tree hashes (`tree_hash`),
  and byte-identical rank cores.
- `load_merged(path)` (exported as `load`) reruns the derivation and requires EXACT
  cache equality; tamper refuses `MergedArtifactError`, never degrades to a gap.
  An unparseable member enters `load_degradations` and caps effective alignment at
  `partial`.

## Gotchas
- Refused typed in C1: p2p/pipeline boundaries (C3), DTensor dual geometry (C2),
  merged-level selectors, merged replay/runnable export/validate.
- `distributed_witness="payload"` is a typed construction refusal; digest witnesses
  only.
- Never let load "trust" the cached derivation: the cache is a coherence check, the
  rerun is the authority.
