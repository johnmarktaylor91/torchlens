# Experiments remember: lineage, vary, site sweeps and grids

**Doc of record for the F03 surface. Every spelling DOCUMENTED-UNSTABLE
pending naming-session ratification.** The design law: *the Bundle is the
record* — the material record of an experiment, always on, travelling with
the artifact: lineage, the operation chronology, and the complete
per-candidate effect table. The semantic layer (hypothesis, metric choice,
verdict) is the user's: the experiment ledger that once carried it was
removed on 1 Nov 2026 (`deprecations.md`, "Experiment Ledger"), and every
`tl.experiment.ledger`-family spelling now raises the typed redirect.

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

`torchlens.experiment.sweep_sites(baseline, candidates=, edit=, metric=,
retain=, engine="live_hook", model=, inputs=)` runs ONE candidate engine (the
occlusion loop generalized): preflight once, serial execution in stable
order, EXTRACT BEFORE CLEANUP, retain/release under policy (`"all"`,
`"none"`, `k(k)` rolling by |effect|; `retain_ceiling_bytes=` arms the
BEFORE-the-first-candidate byte preflight over the lower-bound aggregation
in `torchlens.experiment.retained_bytes`). `edit=`, `metric=`, and `retain=`
are REQUIRED — which ablation is sound, what to measure, and what evidence
to destroy are the user's scientific choices.

Candidates ENUMERATE however spelled (labels, Selections, scoped facet
selectors) and EXECUTE in structural site-key coordinates; an undeclared
multi-site candidate (including the unscoped `tl.head(i)` broadcast) is a
REFUSED table row, never a silently-wide member. A unit selection
(`tl.units(site, mask)`) executes at its site's structural key with the edit
masked to the selected units, exactly as `trace.fork().do(selection, edit)`
masks it, and its identity includes the units: distinct units at one site are
distinct candidates, and the same units spelled twice (a mask or coordinate
tuples) refuse as `site_sweep_duplicate_candidates` unless `replicates=True`.
The complete per-candidate
effect table — released, refused, and failed rows included — persists in the
bundle artifact (`member_effect_table_v1`) and survives member release and
save/load. `bundle.effects()` serves it; `.rank_changes()` is the MEMBER
axis; `.top(k).selection()` is the explicit tier-c conversion (session-only);
`bundle.map(fn)` recomputes a NEW metric over members that still exist and
reports every effect-table candidate the bundle no longer holds (released,
refused or failed) under its candidate id with the value `"unmeasured"`,
never silently re-scored and never silently dropped.

`torchlens.experiment.head_ablation_candidates(baseline, module, heads=)` is
the head-ablation sugar: mandatory module scoping, v-facet lowering gated on
derived standard-MHA geometry (GQA/MQA refuses
`head_ablation_equivalence_undeclared`), proven against a hand-written
pre-projection hook oracle in its own test file.

## Grids

`torchlens.experiment.sweep_grid(baseline, axes=, metric=, retain=, edit=,
engine="live_hook", model=, inputs=)` is the second face of the SAME engine:
named axes instead of a flat candidate list, and a labeled N-dimensional
result instead of the bare Bundle. `axes` is a plain ordered mapping of axis
name to values, each a list (values named `"0"`, `"1"`, ...) or a mapping of
value name to value (`axes={"layer": {"L3": tl.module("h.3")},
"position": {"t7": tl.positions(tl.zero_ablate(), 7)}}`). An axis holds
sites (selectors, Selections, labels) or edits; at most one axis holds edits,
and it replaces `edit=` (both at once refuse `grid_edit_conflict`, neither
`grid_edit_missing`). A cell's site is its site values combined: one value
as is, a module-free facet plus exactly one module value as
`facet.in_module(address)`, otherwise the `&` intersection. Cells are named
by their comma-joined value names in axis order (`"L3,t7"`), which is also
the retained member's name.

The grid preflights the axes' shape, forms and resolves every cell, checks
integer `tl.positions` against the recorded shapes
(`grid_position_out_of_range`) and refuses duplicates
(`site_sweep_duplicate_candidates`, the edit included in the identity), all
BEFORE any forward; then the cells run through `sweep_sites`' serial loop
with retention under the same policies. A hole in the grid is a refused cell,
never a refused grid (ruled 1 Nov 2026): a cell resolving to no site
(`grid_cell_unresolved`) or to several without a `multi_site` declaration
(`grid_cell_multi_site`) lands as a refused effect row, NaN in `values`,
while every other cell runs. Axis-shape mistakes (`grid_axes_invalid`,
`grid_axis_mixed`, `grid_site_axis_missing`, `grid_edit_axis_multiple`,
`grid_cell_uncombinable`, `grid_too_large`) refuse the whole grid.

The result is a `SweepGrid`: `values` is a float64 tensor shaped by the axes
with NaN for refused or failed cells, `deltas` subtracts `baseline_value`,
`cells` carries one `GridCell` per cell (`candidate_id`, `coords`, `status`,
`value`, `error`, `member_name`), `failed` and `refused` list the cell ids,
`index(**coords)` / `candidate_id(**coords)` / `coords(cell_id)` / `cell(**coords)`
translate between coordinates, cell ids and tensor indices, `to_pandas()` is
the long format (one row per cell, `tabular` extra), and `bundle`, `effects`
and `history_id` are the record. The axes ride in the recorded `site_sweep`
operation's params (`axes`, `edit_axis`, `cells`; the effect table's
`edit_repr` is `axis:<name>` for an edit axis), so a saved and loaded bundle
rebuilds the grid with `bundle.effects().to_grid()`; a table written by
`sweep_sites` refuses `grid_axes_unavailable`. Plain data: no plotting
helper, no threshold, no ranking beyond what `effects` already offers.

## Deliberately not built

No default metric, edit, or retention; no iterated search (ACDC-style
discovery is user/agent strategy — TorchLens enumerates, executes, records);
no verdict computation ever; no semantic record (hypothesis, verdict and
their file are the user's; the Bundle carries the material facts they cite);
no result-wrapper noun for the flat sweep (the Bundle is the return).
