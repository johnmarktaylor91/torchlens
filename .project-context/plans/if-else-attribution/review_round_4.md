## Resolution

| Item | Status | Evidence |
|---|---|---|
| F1 (`5e` can attribute via unmaterialized keys) | RESOLVED | v4 makes the drop rule explicit in Step `5e`: translate only keys present in `events_by_key`, otherwise drop, with rationale tied to the documented false-negative scope. See D10 + Step `5e` in `.project-context/plans/if-else-attribution/plan.md:59-60,217-220`. Tests also now enumerate the false-negative models explicitly at `.project-context/plans/if-else-attribution/plan.md:411-412`. |
| F2 (D17 ungated stack vs eager lazy props) | PARTIAL | The core fix is present: D17 adds `source_loading_enabled` and requires fail-closed behavior for lazy props plus `__repr__` / `__len__` / `__getitem__` in `.project-context/plans/if-else-attribution/plan.md:67,71-84`, with Phase 1 gate wording corrected at `.project-context/plans/if-else-attribution/plan.md:445-449`. But v4 still does not cover all current lazy surfaces or the retained `_frame_func_obj` serialization/pickle risk; see New-in-v4 Finding 1. |
| F3 (D14 fallback can still silently pick wrong scope) | PARTIAL | The decision text is correct now: D14 v2 requires unique step-2 match and otherwise fail-closed, with a dedicated ambiguity test at `.project-context/plans/if-else-attribution/plan.md:64,399-403`. But the later `attribute_op` subsection still reintroduces the forbidden “smallest containing scope” heuristic at `.project-context/plans/if-else-attribution/plan.md:276-280`; see New-in-v4 Finding 2. |
| F4 (invariants incomplete) | RESOLVED | v4 adds exactly the missing invariants Round 3 asked for: aggregate exactness for `LayerLog.cond_branch_children_by_cond` and bidirectional legacy IF-view consistency. See `.project-context/plans/if-else-attribution/plan.md:371-374`. |
| F5 (Phase 1 gate imprecise) | RESOLVED | The Phase 1 gate now distinguishes conditional-branch defaults from populated `FuncCallLocation` identity fields, and it explicitly calls out the lazy gated behavior under `save_source_context=False`. See `.project-context/plans/if-else-attribution/plan.md:445-449`. |
| F6 (`cond_id` ordering underspecified / nondeterministic) | RESOLVED | v4 now defines deterministic ordering in both collection and materialization: iterate bools in `ModelLog.layer_list` order, collect keys in first-seen insertion order, then assign dense IDs in that order. See `.project-context/plans/if-else-attribution/plan.md:202-210`. The underlying `layer_list` order is a real codebase contract, not implied hand-waving: `torchlens/data_classes/model_log.py:177`, `torchlens/postprocess/labeling.py:360-377`, `torchlens/validation/invariants.py:915-945`. |
| F7 (docs missed `user_funcs.py`) | RESOLVED | Phase 10 now explicitly includes `torchlens/user_funcs.py` docstrings and README/API references for the changed `save_source_context` contract. See `.project-context/plans/if-else-attribution/plan.md:484-488`. |
| R3 partial: save-source capture-path ungating | PARTIAL | The ungating points are now explicitly named and the Phase 1 gate is coherent: `.project-context/plans/if-else-attribution/plan.md:67,314-316,445-449`. Remaining issue is not capture-path coverage anymore; it is the incomplete disabled-source contract on `FuncCallLocation` itself. |
| R3 partial: lifecycle wiring for new fields | RESOLVED | Rename, cleanup, and export wiring are now enumerated concretely, including the new cond-id-aware structures and pass maps, at `.project-context/plans/if-else-attribution/plan.md:318-339`, with matching cleanup-strengthening tests at `.project-context/plans/if-else-attribution/plan.md:418-426`. |
| R3 partial: runtime identity too weak for scope resolution | PARTIAL | D14 materially improves the identity tuple and adds fail-closed ambiguity handling at `.project-context/plans/if-else-attribution/plan.md:64,399-403`, but the contradictory fallback text at `.project-context/plans/if-else-attribution/plan.md:276-280` means the plan is still internally inconsistent on this exact point. |

## New-in-v4 findings

### Finding 1
- Severity: high
- Description: `source_loading_enabled` still does not cover the full disabled-source contract, and the plan now risks breaking pickle/serialization under `save_source_context=False`.
- Why it matters: v4 fixes the core Round 3 issue conceptually, but it only names four lazy properties plus `__repr__` / `__len__` / `__getitem__` in D17 and the `FuncCallLocation` data-model note (`.project-context/plans/if-else-attribution/plan.md:67,71-84`). The current class has three additional lazy accessors that also call `_load_source()` today: `code_context_labeled`, `call_line`, and `num_context_lines` (`torchlens/data_classes/func_call_location.py:194-218`). More importantly, the current object releases `_frame_func_obj` only inside `_load_source()` (`torchlens/data_classes/func_call_location.py:156-169`), and the repo already records this as a known gotcha (`torchlens/data_classes/AGENTS.md:35-36`). Under the v4 disabled-source design, `_load_source()` is intentionally never reached, so the function object can stay attached indefinitely. That conflicts with the plan’s own pickle-compat story at `.project-context/plans/if-else-attribution/plan.md:439`, and nested/local function objects are exactly the kind of thing that make default pickling fragile.
- Additional note: there is no `__iter__` on `FuncCallLocation`, so that specific accessor path is not relevant. `copy.deepcopy()` is not the loading risk either; it copies object state directly. The unresolved problem is the retained `_frame_func_obj` state and the still-uncovered lazy properties.
- Concrete fix: make the disabled path a fully-specified state, not just a read-time guard. The plan should require that when `source_loading_enabled=False`, all lazy backing fields are initialized to stable empty/None values without disk access, and `_frame_func_obj` is cleared immediately or excluded from pickle via explicit state handling. The same section should enumerate `code_context_labeled`, `call_line`, and `num_context_lines` alongside the other lazy properties.

### Finding 2
- Severity: medium
- Description: D14 is still internally contradictory inside v4.
- Why it matters: the top-level decision and the dedicated ambiguity test both say step 3 must fail closed when step 2 is non-unique or missing (`.project-context/plans/if-else-attribution/plan.md:64,399-403`). But the later `attribute_op` subsection still says the fallback order ends with “smallest function scope containing `line_number`” (`.project-context/plans/if-else-attribution/plan.md:276-280`). That is exactly the heuristic Round 3 rejected. An implementation worker can reasonably follow either section.
- Assessment of the fail-closed policy itself: the policy is otherwise defensible. I do not see a new spec-level case where multiple step-2 matches are common and must be disambiguated rather than failed closed; the realistic ambiguous cases here are pathological same-line duplicated defs/lambdas/metaprogramming, and v4 already scopes those toward graceful degradation.
- Concrete fix: replace `.project-context/plans/if-else-attribution/plan.md:276-280` with the D14 v2 wording verbatim so the algorithm subsection and decision table match.

## Fresh findings

### Finding 3
- Severity: low
- Description: the strengthened disabled-source test now encodes a type contract that conflicts with the current public API.
- Why it matters: `SaveSourceContextOffModel` now expects `loc.source_context == []` in `.project-context/plans/if-else-attribution/plan.md:415`, but `source_context` is currently a `str` field/property (`torchlens/data_classes/func_call_location.py:63,185-191`) and existing tests already assert that contract (`tests/test_metadata.py:896`). This is not a blocker, but it is the kind of ambiguity that invites avoidable churn during implementation.
- Concrete fix: decide whether disabled `source_context` should remain an empty string / sentinel string for backward compatibility, or whether the API is intentionally changing. Then align D17, the test text, and the `FuncCallLocation` type hints/docstrings to one answer.

## Tests

- The v4 test strengthening is materially better and mostly closes the Round 3 coverage gaps. In particular, `SameLineNestedDefModel`, the strengthened `RolledMixedArmModel`, `KeepUnsavedLayersFalseModel`, and `BranchEntryWithArgLabelModel` now cover the intended behaviors they were added for. See `.project-context/plans/if-else-attribution/plan.md:394,402,418-429`.
- The deterministic `cond_id` path is adequately covered at the spec level now because ordering is defined in 5b/5c, and the codebase already has a stable `layer_list` execution-order contract (`torchlens/data_classes/model_log.py:177`, `torchlens/postprocess/labeling.py:360-377`, `torchlens/validation/invariants.py:915-945`).
- The main remaining test gap is the disabled-source contract. v4 still needs explicit assertions for `code_context`, `code_context_labeled`, `call_line`, and `num_context_lines` under `save_source_context=False`, plus a pickle round-trip on a log captured with `save_source_context=False` so the `_frame_func_obj` retention problem cannot slip through.
- I do not see a remaining Round 3-style gap in the 5e drop rule tests. The documented false-negative cases are now explicitly listed, and the drop behavior is specified rather than left implicit.

## Summary: verdict = RED

v4 is close, and most Round 3 items are either resolved or reduced to cleanup-level inconsistencies. But one genuine HIGH remains: the `source_loading_enabled` design is still incomplete for the real `FuncCallLocation` object model, especially around retained `_frame_func_obj` state and pickle/serialization behavior under `save_source_context=False`. There is also one medium internal contradiction in D14, but that alone would not have blocked convergence.

If Finding 1 is fixed and Finding 2 is textually reconciled, this should likely move to YELLOW immediately.

Counts: 1 high / 1 medium / 1 low
