# Merged Trace Contract (`merged-directory` artifacts, rung C1)

This document is the **frozen specification** for cross-rank merged traces
produced by `tl.merge_ranks()` and persisted as `merged-directory` artifacts.
It plays the same role for `torchlens.merged` that
`runnable_tlspec_contract.md` plays for runnable artifacts: the enum and
finding-kind tables below are release-gated against the code by
ordered-list-equality tests (`tests/test_merged_engine.py::TestContractLockstep`).
Editing either side without the other turns the gate red.

## 1. Scope (rung C1)

`tl.merge_ranks([trace_or_path, ...])` merges N rank-local captures of one
SPMD program at their **explicit in-forward `torch.distributed` python
collectives** on symmetric-issue (`coll`) channels.

Refused typed in this release, with `fields["code"] = "merge_scope_unsupported"`:

- point-to-point boundaries (`send`/`recv`, any `p2p/...` channel) — pipeline
  pairing is rung C3;
- boundaries carrying DTensor dual geometry — sharded-topology capture stays
  refused until rung C2's capture-fidelity census is green (relaxation follows
  fidelity, never the other way around);
- functional-collective boundaries (the documented-unstable
  `functional_collective_boundary_v0` payload recorded by the C2 recording
  slice) — C1 joins the frozen `collective_boundary_v1` payload only; funcol
  boundary joining is rung-C2 merged-side scope behind its own authorizing
  ruling. Refusing the member core typed is strictly fail-closed: the
  pre-recording alternative was a merge with the funcol traffic invisibly
  absent.

Merged replay does not exist (`merge_run_unsupported`); merged runnable
export, merged `validate()`, receptive/projective fields, and intervention
chaining are refused typed. Backward/gradient merging is fork F1 (deferred).

Member cores must carry a mergeable settled capture outcome: a member whose
settled outcome is FAILED, ABORTED_NONFINITE, or UNKNOWN refuses typed at
input resolution (`merge_input_invalid`, reason
`member_outcome_not_mergeable`) for live traces and path-loaded bundles
alike. HALTED (and legacy UNATTESTED) members merge and are DISCLOSED —
`MergedTrace.member_outcomes` maps each rank to its settled status string and
`summary()` carries the non-complete disclosure ahead of the witness-coverage
line, so the presenter never claims `attested_complete` silently over a
halted prefix. The disclosure is presenter-side and never edits the
derivation or the descriptor cache.

## 2. Principles (binding)

- **P1 — rank cores are single truth.** A merged artifact CONTAINS N per-rank
  captures byte-identical to standalone saves; the merge never rewrites a
  per-rank claim.
- **P2 — capture never communicates.** All cross-rank identity is established
  offline from rank-local evidence.
- **P3 — correlation by counting, verified by value.** Alignment authority is
  the issue-ticked seq counters; witness digests are redundant byte-exact
  evidence that can only **demote**, never rescue.
- **P4 — fail closed, typed.** Callers branch on `fields["code"]` /
  `finding.kind`, never message text.

## 3. Frozen vocabularies

### 3.1 `MergedErrorCode`

The exact `MergedErrorCode` values are:

```text
merge_input_invalid
merge_scope_unsupported
group_lifetime_evidence_conflict
merge_conflict
merged_schema_invalid
merged_descriptor_tamper
merge_run_unsupported
merged_selector_unsupported
merged_surface_unsupported
merged_member_released
```

### 3.2 Merge finding kinds

The exact `MERGE_FINDING_KINDS` values are:

```text
group_lifetime_evidence_conflict
presence_gap
relation_violation
order_contradiction
correlation_delta_mismatch
value_divergence
load_degradation
```

STRUCTURAL kinds (make the merge `conflicted`; `merge_ranks` refuses with
`MergeConflictError`): `group_lifetime_evidence_conflict`,
`relation_violation`, `order_contradiction`, `correlation_delta_mismatch`.
`presence_gap` demotes structure to `partial`. `value_divergence` demotes the
value status only. `load_degradation` caps the EFFECTIVE alignment at load.

### 3.3 `MergeAlignment` (structural only)

```text
aligned
partial
conflicted
```

`merged.stored_alignment` is frozen at merge time in the descriptor;
`merged.alignment` is the EFFECTIVE value — stored, lowered to `partial` by a
non-empty `load_degradations` ledger. An unparseable rank core can never
support an effective `aligned`. Every presentation of `aligned` carries the
witness-coverage line.

### 3.4 `BoundaryConsistency` (per join, totalized)

```text
attested
mismatched
not_applicable
not_present
```

Kind capability evaluates BEFORE the capture setting:

1. Intrinsically unwitnessable kind → `not_applicable` at every witness level.
2. Applicable kind, witness absent/incomplete (level `"none"`, async
   completion unobserved, member rank missing) → `not_present`.
3. Any byte-exact digest mismatch → `mismatched`.
4. Otherwise → `attested`.

### 3.5 `MergeValueStatus` (from the join ledger)

```text
divergent
attested_complete
attested_partial
unwitnessed
```

Any `mismatched` join → `divergent` (never presented as any flavor of
attested). All applicable joins attested (≥ 1 applicable) →
`attested_complete`; some attested → `attested_partial`; zero attested
applicable → `unwitnessed`. Value divergence is NOT a structural conflict:
the artifact constructs and saves; surfaces that would treat rank values as
interchangeable refuse.

## 4. Correlation and joining

- Join identity: `(membership_digest, lifetime_ordinal, channel, seq_delta)`.
  `membership_digest` = SHA-256 over sorted global ranks;
  `lifetime_ordinal` = the TorchLens-owned ever-created group generation
  (see the C0 lifecycle ledger). `ordinal_source` is diagnostic, never keyed.
- **Seq counters are absolute from arm time and rank-local.** Ranks align as
  seq DELTAS from each rank's first recorded key per `(group_uid, channel)`;
  absolute bases are never compared (arming histories legitimately differ).
- **PRE-JOIN membership-lineage audit** (`torchlens.distributed.
  audit_membership_lineages`) runs BEFORE any joining or gap derivation, at
  merge time and again at load rederivation. A conflicted membership refuses
  STRUCTURALLY: no uid of it joins and no unmatched key of it becomes a
  presence gap. The named remedy is `torchlens.distributed.arm()` at process
  start on every rank.
- The redundant `c10d_group_seq` cross-check compares as deltas from the
  first joined key; disagreement is the structural
  `correlation_delta_mismatch` finding (the backend's own ordering contradicts
  the join alignment). Absence of the probe never demotes anything. Within
  one rank core, recorded `c10d_group_seq` presence must follow the
  recorder's latch shape — a prefix of values, at most one boundary
  disclosing `c10d_group_seq_read_failed`, then an all-null suffix — so
  selective nulling of a value refuses at parse (`merged_schema_invalid`)
  instead of silently deleting a cross-check finding; a uniformly-absent
  core remains the honest capability-absent shape and stays finding-free.
  The journal's `boundaries` container itself must be a list (or tuple) of
  boundary records: an int/bool/mapping/string container refuses
  `merged_schema_invalid` at merge entry, live and loaded cores alike, with
  the container named (W051-CAPT3; previously a bare `TypeError`).
- Presence expectations and gaps derive from the group memberships recorded
  INSIDE the surviving rank cores' boundary records. A declared
  `expected_ranks` input can only WIDEN expectations, never narrow them.

## 5. Witness capability allowlist (semantic authority)

Semantic authority is this versioned public-contract table, not CI probes
(probes remain regression tests). Verdict-grade backends: `gloo`, `nccl`,
`mpi`. Any other backend renders every witness `not_applicable`.

| kind | witness | authority |
|---|---|---|
| `all_reduce` | destination digests identical on every member rank | torch documents bitwise-identical results across processes |
| `all_gather`, `all_gather_into_tensor` | destination digests identical on every member rank | definitional data movement |
| `broadcast` | every destination digest equals the root's contribution digest (the root's bytes are the source) | definitional data movement |
| `gather` | root's destination list slice *i* equals member *i*'s contribution (group-rank indexed) | definitional data movement |
| `scatter` | member *i*'s destination equals root's contribution list slice *i* | definitional data movement |
| `all_to_all` | rank *j*'s destination *i* equals rank *i*'s contribution *j* | definitional data movement |
| `reduce`, `reduce_scatter`, `reduce_scatter_tensor` | none | no cross-rank identity property exists |
| `all_to_all_single` | none in C1 | whole-tensor digests cannot witness intra-tensor slice reciprocity |
| `barrier`, object collectives | none | no digestable tensor payload |

`distributed_witness="payload"` remains a typed construction refusal in this
release: the C1 artifact story ships digest witnesses only; payload witnesses
are tensor-valued and require their own blob family (deferred; a future rung
must define it as a schema change here).

## 6. Artifact (`merged-directory`)

```
<root>/
  manifest.json            # integrity root (bundle_format: "merged-directory",
                           # tlspec_version: 7, descriptor sha256, member tree hashes,
                           # provenance, created_at)
  merge/descriptor.json    # canonical-JSON derivation CACHE
  members/rank_NNNN.tlspec # ordinary rank-core bundles (byte-identical to standalone saves)
```

- Canonical JSON (`torchlens-canonical-json-v1`): UTF-8, sorted keys, no
  NaN/Infinity, LF, no insignificant whitespace, trailing newline.
- Canonical tree hash per rank core: sorted POSIX-relative paths; each entry is
  an 8-byte big-endian path length, the UTF-8 path bytes, 8-byte big-endian file
  size, and 32 raw SHA-256 bytes. Framed entries are joined by one LF byte;
  the tree hash is SHA-256 of that concatenation. Symlinks are rejected at hash
  time.
- The merged ROOT `manifest.json` is a different discriminated object from a
  trace bundle's manifest (no tensor table) and carries `tlspec_version: 7`,
  gated independently in the merged loader. Runtimes without the merged
  loader refuse the format typed at the closed `bundle_format` vocabulary.
- Descriptor: `descriptor_kind: "cross_rank_merge"`, `schema_version: 1`,
  `encoding`, the member table (`rank`, `path`, `tree_sha256`), and the full
  `derivation` payload. Required v1 fields are exactly those; unknown
  `descriptor_kind`/`schema_version` values refuse (`merged_schema_invalid`).
  The descriptor deliberately contains NO timestamps.

### 6.1 Load: full rederivation (the descriptor is a CACHE)

At every load:

1. **Integrity.** Descriptor bytes vs the root-manifest SHA-256; each member's
   canonical tree hash vs BOTH recorded copies. Mismatch = typed
   `merged_descriptor_tamper` refusal. Tamper is never a gap.
2. **Independent rederivation.** The same `derive_merge` function reruns over
   the parsable rank cores' boundary records alone; when every member parses,
   canonical-JSON EXACT equality with the descriptor cache is required — any
   inequality is tamper.
3. **Environment degradations.** A member that no longer parses on this
   runtime enters `load_degradations`; the rederivation covers the parsable
   subset, the STORED verdicts stay visible as stored, and the effective
   alignment is capped at `partial`. Exact-equality is not evaluable in this
   case and is not claimed.

**Threat-model boundary** (same as the runnable contract): whole-artifact
coherent reauthoring — rewriting the rank cores AND every hash AND the
descriptor into a byte-for-byte honest artifact of a *different* merge — is
out of scope. Note the rederivation still runs on the reauthored inputs, so
such an artifact can never present verdicts its own cores do not support
(deleting a rank's core coherently still rederives as `partial` with gaps,
because the surviving cores' boundary records declare the full membership).

### 6.2 Determinism

The merge derivation is a pure function of the input rank-core content. The
named nondeterministic field set for the artifact is exactly:
`manifest.json:created_at` (plus member tree-hash entries when the members
are re-saved from live traces rather than copied, since rank cores embed
their own `created_at`). The descriptor of a merge over identical on-disk
members is byte-identical across runs; the determinism test pins this.

## 7. Release gate

- `MergedErrorCode` and `MERGE_FINDING_KINDS` above must equal the code
  enums exactly (ordered) — `TestContractLockstep`.
- The `_distributed.py` refusal suggestion strings must reference
  `tl.merge_ranks` truthfully (dense explicit-collective captures merge;
  DTensor/TP/PP capture stays refused pending C2/C3).

## 8. Lifetime and release

`MergedTrace` holds each rank-core handle strongly for as long as the presenter
is reachable. Dropping the presenter releases those handles; like ordinary
`Trace` objects, member traces can contain cycles and may require Python's cyclic
collector before memory is reclaimed.

`MergedTrace.release()` is the explicit counterpart of `Trace.cleanup()` for the
presenter: it drops every rank-core handle so member traces (and their retained
activations) can be reclaimed without waiting for the presenter itself to become
unreachable. After `release()`:

- The derivation record stays readable: `alignment` / `value_status` (stored and
  effective), `rank_ids`, `expected_ranks`, `joins`, `gaps`, `findings`,
  `report`, `load_degradations`, `summary()`, `to_pandas()`, and
  `happens_before()` all keep working.
- Every surface that must resolve a member trace — `ranks`, `__getitem__` (all
  spellings), `super_op()`, `join_ops()`, and `save()` — refuses typed with
  `fields["code"] = "merged_member_released"`. A released presenter never
  presents as a zero-member merge and never leaks a bare `KeyError`.
- `release()` is idempotent.

`tl.distributed.arm()` intentionally keeps the process-wide group-lifecycle
ledger for the whole armed interval because old generations are evidence needed
to reject ambiguous ordinal reuse. Long-lived processes that have finished all
distributed capture work should call `torchlens.distributed.disarm()`. A
successful disarm restores every wrapper and releases the ledger; if any restore
fails, disarm raises and retains the unresolved original-function entries so the
operation can be retried rather than discarding repair state.
