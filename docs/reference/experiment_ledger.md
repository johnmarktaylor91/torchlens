# Experiments remember: lineage, vary, site sweeps, and the experiment ledger

**Doc of record for the F03 surface. Every spelling DOCUMENTED-UNSTABLE
pending naming-session ratification.** The design law: *the Bundle proves,
the ledger remembers* — the Bundle is the MATERIAL record of an experiment
(always on, travels with the artifact); the experiment ledger is the
SEMANTIC record (hypothesis, metric choice, verdict — opt-in, references the
material record, never duplicates it).

## The material record

- **Container lineage.** Every `Bundle` carries a persisted random
  `bundle_id` (never a content hash), per-member `member_construction`
  origin anchors (closed vocabulary: constructed / added / forked / varied /
  swept / loaded), and a hash-chained `operations` chronology
  (`bundle_operation_v1` rows: seq + previous-row digest, the same spelling
  as the v9 EVENT audit rows). `Bundle.fork()` carries the relation table
  and preserved sections, mints a NEW id, and anchors the source on both
  containers. All of it persists in `bundle.json` (optional writer-owned
  sections, fail-closed load validation, `bundle_lineage_invalid`).
- **Transaction envelopes.** Every intervention transaction writes one
  `intervention_event_v2` envelope (the C03 substrate); experiment-layer
  doors additionally land the canonical `kind="EVENT"` audit row with its
  `seq`/`prev_event_digest` hash chain (the tlspec-v9 admission's writer).
- **The provenance join.** `bundle.why(member[, relative_to])` walks ORDERED
  event ids (never set arithmetic) and answers "member X = reference + THESE
  edits" with four orthogonal honesty axes: lineage status (exact / partial /
  diverged / unrelated / unattested), payload fidelity (declared / opaque —
  a bare user callable names the site and the fact, never guessed content),
  value residual (identical recorded chains with differing outputs read
  `unexplained` — the unrecorded-write detector), and comparability
  (structural relationship + lanes, in their own columns). Additive wording
  is licensed ONLY when the reference-side suffix is empty.
  `bundle.provenance()` is the one-row-per-member view. Both are derived,
  never stored.
- **vary.** `bundle.vary(mapping)` — *do = one edit to all members; vary =
  one explicit edit or identity per member; both take the same spec
  objects.* Complete coverage by default (`unmentioned="unchanged"` opts a
  subset in); `None` is an explicit recorded identity; donor plans normalize
  ONCE across the whole mapping (one reused SamplingPlan = one persisted
  donor group); a mid-apply failure claims NO rollback and lands per-member
  outcomes in the chronology row (`vary_partial_failure`,
  `material_action_completed=True`).

## Site sweeps and the effect table

`torchlens.experiment.site_sweep(baseline, candidates=, edit=, metric=,
retain=, engine="live_hook", model=, x=)` runs ONE candidate engine (the
occlusion loop generalized): preflight once, serial execution in stable
order, EXTRACT BEFORE CLEANUP, retain/release under policy (`"all"`,
`"none"`, `top_k(k)` rolling by |effect|; `retain_ceiling_bytes=` arms the
BEFORE-the-first-candidate byte preflight over the lower-bound aggregation
in `torchlens.experiment.retained_bytes`). `edit=`, `metric=`, and `retain=`
are REQUIRED — which ablation is sound, what to measure, and what evidence
to destroy are the user's scientific choices.

Candidates ENUMERATE however spelled (labels, Selections, scoped facet
selectors) and EXECUTE in structural site-key coordinates; an undeclared
multi-site candidate (including the unscoped `tl.head(i)` broadcast) is a
REFUSED table row, never a silently-wide member. The complete per-candidate
effect table — released, refused, and failed rows included — persists in the
bundle artifact (`member_effect_table_v1`) and survives member release and
save/load. `bundle.effects()` serves it; `.most_changed()` is the MEMBER
axis; `.top(k).selection()` is the explicit tier-c conversion (session-only);
`bundle.measure_members(metric=)` recomputes a NEW metric over members that
still exist and names the unmeasured.

`torchlens.experiment.head_ablation_candidates(baseline, module, heads=)` is
the head-ablation sugar: mandatory module scoping, v-facet lowering gated on
derived standard-MHA geometry (GQA/MQA refuses
`head_ablation_equivalence_undeclared`), proven against a hand-written
pre-projection hook oracle in its own test file.

## The experiment ledger

`torchlens.experiment.ledger(path, hypothesis=, metric=)` arms a
ContextVar-scoped, file-backed, single-writer ledger and opens entry 1
(visibly DRAFT without a hypothesis). One entry = one semantic question;
events are append-only and hash-chained; every event is fsynced before it is
reported. Kill -9 after event k recovers exactly k events (a torn tail line
is discarded AND disclosed; an interior break refuses). The three-layer
honesty law: MECHANICAL fields are engine-derived, INTERPRETIVE fields
(hypothesis, metric choice, note, verdict) are actor-stamped, and DERIVED
projections never masquerade as either — a verdict without basis renders
"asserted; no recorded basis", `declared_at_seq` makes pre-registration a
checkable fact, and TorchLens LINTS (never blocks, never computes a
verdict). Emission follows MUTATION: one material step per top-level
operation (a 144-candidate sweep is ONE step); reads are quiet; observations
are explicit. Write failures after material work QUARANTINE the entry and
return the material result unharmed (`on_record_error="raise"` is the strict
opt-in). Contents are references only (EvidenceRefs with digests; a bundle
overwritten at the same path reads `stale`, never silently relinked).

Serving is read-only and artifact-fresh mid-experiment:
`ledger_overview` / `ledger_entry` / `ledger_evidence` (also on the MCP
stdio server as `torchlens_ledger_*`). No tool runs a model, arms a process,
writes a verdict, or executes an opaque metric.

## Deliberately not built

No default metric, edit, or retention; no iterated search (ACDC-style
discovery is user/agent strategy — TorchLens enumerates, executes, records);
no verdict computation ever; no result-wrapper noun (the Bundle is the
return); no MCP write door in v1 (its plumbing — actor ids, event ids,
append-only amendments, one-writer rules — already rides the schema).
