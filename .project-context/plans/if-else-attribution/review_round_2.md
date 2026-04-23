# Adversarial Review Round 2: `if` / `elif` / `else` Attribution Sprint Plan v2

Scope reviewed:
- Current plan: `.project-context/plans/if-else-attribution/plan.md`
- Prior plan: `.project-context/plans/if-else-attribution/plan_v1.md`
- Prior review: `.project-context/plans/if-else-attribution/review_round_1.md`
- Grounding: `.project-context/research/{ast-design,adversarial,prior-art}.md`
- Current code paths named in the plan and round-1 review

## Resolution

| Finding | Status | Why |
|---|---|---|
| F1 | PARTIALLY RESOLVED | D9 correctly changes the intended semantics so AST classification no longer depends on `save_source_context` (`plan.md:61`, `84`, `413-417`). But the plan still assumes `(file, line)` is "always" present while the current capture path still zeros `func_call_stack` when `save_source_context=False` in [output_tensors.py](/home/jtaylor/projects/torchlens/torchlens/capture/output_tensors.py:311), [source_tensors.py](/home/jtaylor/projects/torchlens/torchlens/capture/source_tensors.py:169), and [graph_traversal.py](/home/jtaylor/projects/torchlens/torchlens/postprocess/graph_traversal.py:67). The v2 phases/deliverables never explicitly schedule those ungates. |
| F2 | RESOLVED | D3 now treats `bool(x)` inside `If.test` / `elif` as branch-participating and records `bool_wrapper_kind="bool_cast"` (`plan.md:55`, `194-202`). The new `IfBoolCastModel` row exercises the fix (`plan.md:364-367`). |
| F3 | RESOLVED | D13 changes 5e from "branch-start children only" to diffing stacks on every forward edge (`plan.md:65`, `228-240`). The new `BranchUsesOnlyParameterModel` / `BranchUsesOnlyConstantModel` rows directly cover the original gap (`plan.md:358-362`). |
| F4 | PARTIALLY RESOLVED | v2 adds the missing lifecycle section and includes `labeling.py`, `cleanup.py`, and `interface.py` in phases/deliverables (`plan.md:288-310`, `441`, `459-463`). But the cleanup spec still only mentions pruning "new edge lists" and omits `conditional_edge_passes`, `conditional_events[*].bool_layers`, and surviving parents' new child lists/maps; current cleanup already only knows about `conditional_branch_edges` / `conditional_then_edges` ([cleanup.py](/home/jtaylor/projects/torchlens/torchlens/data_classes/cleanup.py:157), [cleanup.py](/home/jtaylor/projects/torchlens/torchlens/data_classes/cleanup.py:212)). |
| F5 | RESOLVED | Invariant 6 is rewritten to allow either stack to be a prefix of the other, explicitly permitting reconvergence (`plan.md:330-335`). `ReconvergingBranchesModel` now asserts that case (`plan.md:376-379`). |
| F6 | RESOLVED | Invariant 7 has been moved off the per-node `cond_branch_elif_children` map and onto `ConditionalEvent.branch_ranges` (`plan.md:336-337`). That addresses the sparse-per-parent case raised in round 1. |
| F7 | RESOLVED | D15 adds explicit pass-level rolled-edge metadata and the visualization section now defines composite rolled labels (`plan.md:67`, `139-142`, `250-261`). That resolves the original absence of a truthful rolled-edge data source. |
| F8 | RESOLVED | v2 now specifies label precedence, says arm-entry labels suppress `IF`, and moves arg labels off the main `label` field (`plan.md:248-275`). That directly addresses the overwrite/collision problem in current [rendering.py](/home/jtaylor/projects/torchlens/torchlens/visualization/rendering.py:985) and [dagua_bridge.py](/home/jtaylor/projects/torchlens/torchlens/visualization/dagua_bridge.py:342). |
| F9 | PARTIALLY RESOLVED | D14 adds `code_firstlineno` and `code_qualname` and keys `FileIndex` by `(code_firstlineno, qualname)` (`plan.md:66`, `71-84`, `187-193`). But `attribute_op()` still says resolution is by `(filename, code_firstlineno, func_name)` rather than `code_qualname` (`plan.md:204-207`), so the new identity field is not actually in the primary runtime match key. |

## New

## Finding 1: D10 and Step 5 disagree on when `conditional_id` becomes a dense int
- Severity: blocker
- What's wrong: D10 says `conditional_id` is a dense per-`ModelLog` integer and that the structural AST key is internal only (`plan.md:62`). But 5b has `classify_bool()` already returning `maybe_cond_id` and immediately writing `bool_conditional_id` plus appending into `ModelLog.conditional_events[cond_id].bool_layers` (`plan.md:202`, `216-223`), even though 5c is the step that materializes `conditional_events`.
- Why it matters: as written, the identifier can’t be both a dense external int and an AST-internal structural key at the same moment. The spec also asks `attribute_op()` to return `List[Tuple[int, str]]` before the dense-id materialization exists (`plan.md:183`, `204-209`). An implementation worker would have to invent an unstated temporary-key rewrite pass.
- Fix: introduce an explicit temporary `ConditionalKey` type in `ast_branches.py`, let 5a/5b/5e operate on keys, and make 5c the sole "key -> dense int" translation step that rewrites `bool_conditional_id`, `conditional_branch_stack`, and all emitted edge records in one pass. Alternative: move event materialization and dense-id assignment before 5b.
- Test: add a model with one `if a and b:` plus one nested `elif` so that multiple bool layers and nested stack entries must all map to the same final dense IDs.

## Finding 2: D13 allows multi-entry edges, but the new edge structures are still cond-id-blind
- Severity: high
- What's wrong: 5e now says a single forward edge can gain every stack item present in `child_stack` but not `parent_stack` (`plan.md:231-238`). That makes one edge legitimately enter multiple branches at once, e.g. outer `then` and inner `then` for a parameter-fed op inside a nested `if`. But the preserved THEN structures are still only `(parent, child)` and `List[str]` with no `cond_id` (`plan.md:108-111`, `163-165`), and the new per-parent `elif` / `else` child maps are also not keyed by `cond_id` (`plan.md:103-105`).
- Why it matters: the first new case D13 unlocks is also the first case the data model cannot represent losslessly. On one `parent -> child` edge you can have two simultaneous THEN entries, or two different conditionals each with `elif_1`, and the current primary structures collapse them. The visualization API also still returns a single branch label per edge (`plan.md:252-257`, `269`, `275`), so even a correct implementation has nowhere truthful to put the second entry.
- Fix: add a cond-id-aware primary arm-entry structure for all branch kinds, e.g. `conditional_arm_edges[(cond_id, branch_kind)] -> List[(parent, child)]` and a parent-side `cond_branch_children_by_cond: Dict[int, Dict[str, List[str]]]`. Keep `conditional_then_edges` and `cond_branch_then_children` as backward-compatible derived unions only.
- Test: add a nested-`if` model where `self.bias -> add` enters outer THEN and inner THEN on the same pass, and assert both cond IDs survive in the primary structure.

## Finding 3: D14 captures `code_qualname`, but the algorithm never actually matches on it
- Severity: medium
- What's wrong: the plan says the file index is keyed by `(code_firstlineno, qualname)` (`plan.md:187-193`) and D14 explicitly adds `code_qualname` to `FuncCallLocation` (`plan.md:66`, `71-84`). But `attribute_op()` still resolves scopes by `(filename, code_firstlineno, func_name)` (`plan.md:204-207`).
- Why it matters: the new field that was added to fix same-name nested helpers and lambda/decorator ambiguity is not in the actual primary lookup key. In practice that means the v2 fix still relies mostly on `co_firstlineno`, with `code_qualname` present but inert. That is weaker than the decision text promises.
- Fix: make the exact-match path use `(filename, code_firstlineno, code_qualname)` and only fall back to `func_name` / line-based heuristics when `code_qualname` is unavailable.
- Test: keep `NestedHelperSameNameModel`, but replace `LambdaBranchModel` with a case that actually exercises D14’s matching logic under a real branch-attribution path. The current `lambda x: op(x) if cond else op2(x)` example is `IfExp`, which the plan explicitly classifies but does not attribute (`plan.md:20`, `45`, `393`).

## Fresh

## Finding 4: The D15 cleanup and validation story is still incomplete
- Severity: medium
- What's wrong: the lifecycle section says cleanup should prune "new edge lists" (`plan.md:304-305`), and the keep-unsaved-layers test only mentions those edge lists (`plan.md:419-423`). Nothing there covers `conditional_edge_passes`, `conditional_events[*].bool_layers`, or the new aggregate pass maps on `LayerLog`. The current cleanup code only filters `conditional_branch_edges` and `conditional_then_edges` ([cleanup.py](/home/jtaylor/projects/torchlens/torchlens/data_classes/cleanup.py:157), [cleanup.py](/home/jtaylor/projects/torchlens/torchlens/data_classes/cleanup.py:212)).
- Why it matters: once D15 exists, stale rolled-edge keys are user-visible wrong metadata, not just dead internal state. The same applies to stale `bool_layers` references inside `ConditionalEvent`: they let a pruned bool layer continue to "drive" a surviving event.
- Fix: extend the cleanup spec to scrub `conditional_edge_passes`, `conditional_events[*].bool_layers`, and every surviving parent-side conditional child list/map. Add reverse-link invariants for `bool_layers` and exactness invariants for the pass maps.
- Test: strengthen `KeepUnsavedLayersFalseModel` so it asserts that no removed label appears anywhere in `conditional_edge_passes`, `conditional_events.bool_layers`, `cond_branch_*_children`, or `conditional_branch_stack_passes`.

## Tests

- `BranchUsesOnlyParameterModel`: good. It directly exercises the D13 fix the round-1 review requested.
- `IfBoolCastModel`: good. It directly exercises D3’s wrapped-bool normalization and the new `bool_wrapper_kind`.
- `RolledMixedArmModel`: partial. It checks renderer output, but not whether `conditional_edge_passes` is itself correct, stable, or pruned after layer removal.
- `NestedHelperSameNameModel`: partial. It exercises `code_firstlineno`; it does not prove that `code_qualname` is used in matching.
- `LambdaBranchModel`: not a real coverage row for D14. As specified, it uses `IfExp`, which the plan marks as classified-but-not-attributed (`plan.md:19-25`, `43-46`, `393`).
- `SaveSourceContextOffModel`: partial. It should assert that `func_call_stack` is non-empty and carries `(file, line, code_firstlineno)` under `save_source_context=False`, otherwise it misses the live gating bug in [output_tensors.py](/home/jtaylor/projects/torchlens/torchlens/capture/output_tensors.py:311).
- `KeepUnsavedLayersFalseModel`: partial. It currently covers only "new edge lists"; it should cover `conditional_edge_passes`, `conditional_events.bool_layers`, and surviving parents’ conditional child lists/maps.
- Missing entirely: one test where a branch-entry edge is also argument-labeled, to prove the D11 precedence rule and the `headlabel` / `xlabel` move actually solve the current [rendering.py](/home/jtaylor/projects/torchlens/torchlens/visualization/rendering.py:1087) overwrite path.
- Missing entirely: one nested parameter-only model where a single edge gains two branch-stack items at once, which is the first correctness cliff introduced by D13.

## Phases

- The phase ordering is not sound as written. `plan.md:310` says Phase 1 cannot be declared green until rename and cleanup are verified, but rename/cleanup are not scheduled until Phase 5 (`plan.md:441-442`). That gate is impossible to satisfy literally.
- D9’s "always capture `(file, line)`" semantics must be a Phase 1 deliverable, and the file list needs to explicitly include `torchlens/postprocess/graph_traversal.py`; the live save-source gate there ([graph_traversal.py](/home/jtaylor/projects/torchlens/torchlens/postprocess/graph_traversal.py:67)) otherwise survives past the foundation phase.
- The dense-id design from Finding 1 has to be resolved before Phase 3. Otherwise Step 5 is blocked on an identifier contract that is still contradictory.
- Recommended order: Phase 1 should include `FuncCallLocation` augmentation, ungated stack capture plumbing, and the temporary-vs-dense conditional-key design; only then is the AST module / Step 5 work in Phases 2-3 well-specified.

## Invariants

- The 8 invariants are not yet complete for the new state.
- Missing invariant: every label in `ConditionalEvent.bool_layers` exists and each referenced bool layer points back to the same `bool_conditional_id`.
- Missing invariant: `LayerLog.conditional_branch_stacks` and `conditional_branch_stack_passes` must equal the exact set/map induced by the underlying `LayerPassLog.conditional_branch_stack` values. D5 adds those aggregates, but none of invariants 1-8 checks them.
- Missing invariant: `conditional_edge_passes` should be exact, not just plausible. It should enforce unique sorted pass numbers and prove that each `(parent_no_pass, child_no_pass, cond_id, branch_kind, pass_num)` corresponds to a real unrolled edge classified that way on that pass.
- Missing invariant: if D13 continues to allow an edge to gain more than one stack item, there must be a cond-id-aware primary representation that proves no information was lost. Right now the invariant set cannot detect that loss because the primary structures are already collapsed.

## Summary: verdict = RED

- v2 fixes most of the round-1 conceptual objections.
- It still contains one specification-level blocker: the conditional-ID contract is internally inconsistent.
- The largest new hole introduced by v2 is D13’s multi-entry edge case, which the current edge data model and visualization contract cannot represent losslessly.

Count: 1 blocker / 1 high / 2 mediums / 0 lows
