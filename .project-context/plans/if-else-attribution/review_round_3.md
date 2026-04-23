## Resolution

| Finding | Status | Why |
|---|---|---|
| R2-F1 Conditional ID contract | RESOLVED | D10 + 5b/5c/5e now consistently use `ConditionalKey` before 5c and dense `cond_id` after 5c. The v2 contradiction between early `bool_conditional_id` writes and late event materialization is gone. |
| R2-F2 Multi-entry edges cond-id-blind | RESOLVED | D16 adds cond-id-aware primaries: `ModelLog.conditional_arm_edges` and `LayerPassLog.cond_branch_children_by_cond`. That directly fixes the v2 lossiness for one edge entering multiple arms. |
| R2-F3 `code_qualname` not used in matching | RESOLVED | D14 now makes the exact-match path `(filename, code_firstlineno, code_qualname)` first, with explicit fallback order after that. |
| R2-F4 Cleanup incomplete for D15 | RESOLVED | Phase 3 now explicitly wires rename/cleanup through `conditional_edge_passes`, `conditional_events[*].bool_layers`, and `cond_branch_children_by_cond`, which are missing in current code at `torchlens/postprocess/labeling.py:216-228`, `torchlens/postprocess/labeling.py:551-562`, and `torchlens/data_classes/cleanup.py:157-166`, `torchlens/data_classes/cleanup.py:212-217`. |
| R1 partial: save-source capture-path ungating | PARTIAL | D17 correctly schedules ungating at the three live gates in `torchlens/capture/source_tensors.py:169-171`, `torchlens/capture/output_tensors.py:67-69` and `311-312`, and `torchlens/postprocess/graph_traversal.py:67-68`. But v3 also claims `source_context` stays empty under `save_source_context=False`, and it never specifies the required `FuncCallLocation` gating despite current unconditional loading in `torchlens/data_classes/func_call_location.py:107-169`. |
| R1 partial: lifecycle wiring for new fields | RESOLVED | Capture defaults, Phase 3 rename, Phase 3 cleanup, and `to_pandas()` are all now called out explicitly. That closes the earlier “new field exists but is not renamed/cleaned/exported” planning gap. |
| R1 partial: runtime identity too weak for robust scope resolution | PARTIAL | D14 is materially better than v2, but the broader concern is only partially resolved because TorchLens supports Python 3.9/3.10 (`pyproject.toml:11`, `pyproject.toml:20-23`), where `code_qualname` is absent and the plan falls back to heuristics. |

## New-in-v3

### Finding 1
- Severity: blocker
- Description: 5e can discover enclosing conditionals that 5c never materialized.
- Why it matters: D10/5c creates `conditional_events` only from bool-driven `ConditionalKey`s collected in 5b, but 5e re-runs `attribute_op(func_call_stack)` for every op and then translates each returned key through `events_by_key`. That does not compose with the plan’s own documented false negatives (`if self.training:`, `if x.shape[0] > 0:`, `if tensor.item() > 0:`): those ops are lexically inside an `if`, so 5e can still see a branch interval even though no branch-driving bool was captured. As written, that is either a lookup miss or silent over-attribution. Current Step 5 is still rooted in observed terminal bools only (`torchlens/postprocess/control_flow.py:30-76`).
- Concrete fix: make 5e explicitly drop any `ConditionalKey` not present in `events_by_key`, or define `attribute_op()` to return only keys already materialized from 5b. Do not leave this as an implicit implementation choice.
- Optional test: strengthen `PythonBoolModel` and `ShapePredicateModel` to assert `conditional_branch_stack == []` on branch-local ops and no failure during 5e translation.

### Finding 2
- Severity: high
- Description: D17 is not implementable as written without changing `FuncCallLocation` semantics.
- Why it matters: v3 says core call-site identity is always captured but lazy source text remains gated by `save_source_context`. Current `FuncCallLocation` cannot do that. Once a stack entry exists, `_load_source()` will eagerly load `code_context`, `source_context`, `func_signature`, and `func_docstring` on first access regardless of `save_source_context` (`torchlens/data_classes/func_call_location.py:107-169`), and `__repr__`, `__len__`, and `__getitem__` all trigger that path (`torchlens/data_classes/func_call_location.py:238-263`). The public docstring also still says call stacks are recorded only when `save_source_context=True` (`torchlens/user_funcs.py:231-233`).
- Concrete fix: add an explicit “source loading enabled” flag to `FuncCallLocation`, have `_load_source()` fail closed when disabled, and update repr/accessor behavior plus docs accordingly.
- Optional test: in `SaveSourceContextOffModel`, assert `repr(loc)` says source unavailable and `len(loc) == 0`, not just that the stack is non-empty.

### Finding 3
- Severity: medium
- Description: D14’s fallback can still select the wrong scope on supported runtimes.
- Why it matters: exact matching is good on Python 3.11+, but TorchLens supports 3.9/3.10 (`pyproject.toml:11`, `pyproject.toml:20-23`) where `code_qualname` may be unavailable. The plan then falls back to `(filename, code_firstlineno, func_name)` and finally “smallest containing scope”. That is still wrong for same-line nested defs/lambdas and for wrapper/decorator frames where the exact scope is not the user branch scope.
- Concrete fix: state a fail-closed rule when the exact match misses on 3.9/3.10 or when multiple candidate scopes remain. Do not silently choose the smallest containing scope in ambiguous cases.
- Optional test: add a 3.9/3.10-targeted same-line nested-helper case, or explicitly mark that class of ambiguity unsupported.

### Finding 4
- Severity: medium
- Description: Invariants 8-11 are helpful but still not sufficient for the new state.
- Why it matters: they cover `bool_layers`, stack aggregate exactness, `conditional_edge_passes`, and orphan temporary keys. They do not explicitly prove that `LayerLog.cond_branch_children_by_cond` equals the pass-stripped union of its passes, and they do not prove legacy IF-view exactness (`conditional_branch_edges` ↔ `cond_branch_start_children`). Those are exactly the kinds of aggregate/view drifts the current multi-pass code is prone to; today `_build_layer_logs` merges only three fields and otherwise keeps first-pass data (`torchlens/postprocess/finalization.py:536-580`, `torchlens/data_classes/layer_log.py:135-142`).
- Concrete fix: add one invariant for `LayerLog.cond_branch_children_by_cond` exactness and one invariant for `conditional_branch_edges`/`cond_branch_start_children` bidirectional consistency.
- Optional test: deliberately corrupt those aggregates in a validation unit test and assert the invariant failure message is specific.

### Finding 5
- Severity: medium
- Description: the new Phase 1 gate is not literally achievable as written.
- Why it matters: Phase 1 includes D17 ungating plus `FuncCallLocation` augmentation, so not all “new fields” can remain empty. The new conditional-attribution fields can remain at defaults, but the new call-site identity fields cannot. As written, the gate is imprecise enough that an implementation worker has to reinterpret it.
- Concrete fix: rewrite the gate to say “new conditional-branch fields remain default/empty; ungated `func_call_stack` and `FuncCallLocation` core identity fields are populated.”

## Fresh

### Finding 6
- Severity: medium
- Description: `conditional_id` ordering is underspecified and likely nondeterministic.
- Why it matters: 5b says to collect a set of unique `ConditionalKey`s, then 5c assigns dense IDs. If that is implemented literally with a set, event order and IDs will drift across runs and Python implementations. That is a bad contract for tests, serialization, and any user-facing display keyed by `cond_id`.
- Concrete fix: define a deterministic order for 5c, either first-seen order from 5b or a structural sort by `(source_file, func_firstlineno, if_stmt_lineno, if_col_offset)`.

### Finding 7
- Severity: low
- Description: the documentation phase does not include the user-facing `save_source_context` contract that v3 changes.
- Why it matters: D17 changes semantics visible to users, but Phase 10 lists only `AGENTS.md`, `architecture.md`, and a limitations page. The current API doc still says the Python call stack is recorded only when `save_source_context=True` (`torchlens/user_funcs.py:231-233`).
- Concrete fix: add `torchlens/user_funcs.py` docstrings and README/API docs to the documentation checklist.

## Tests

| Test | Assessment | Why |
|---|---|---|
| `MultiArmEntryNestedModel` | Adequate for the core D16 case | It proves the exact lossless case round 2 cared about: one edge enters two THEN arms at once, both cond IDs survive in the primary structures, and the derived THEN view dedupes correctly. It does not cover `elif`/`else` multi-entry or renderer composition. |
| `NestedQualnameModel` | Partial | It proves the happy-path 3.11 `code_qualname` exact match. It does not exercise the ambiguous fallback path that still matters on supported 3.9/3.10, and it does not force a fail-closed outcome when exact matching misses. |
| `BranchEntryWithArgLabelModel` | Partial | It is good for Graphviz, where `label` vs `headlabel`/`xlabel` is the real collision in `torchlens/visualization/rendering.py:985-1090`. It is not enough for dagua as written, because the current bridge only carries one edge `label` plus `type` (`torchlens/visualization/dagua_bridge.py:478-490`); a dagua-specific expected representation needs to be defined. |
| strengthened `SaveSourceContextOffModel` | Partial | It will catch the live capture-path gates and the new core identity fields. It still does not prove the promised “lazy source properties remain gated” behavior unless the test explicitly checks `repr`, `len`, and `source_context`/`code_context` access semantics under `save_source_context=False`. |
| strengthened `KeepUnsavedLayersFalseModel` | Partial | It now covers the major raw-label surfaces round 2 called out. It still should assert that aggregate `LayerLog` conditional state stays clean too, especially `cond_branch_children_by_cond` and any pass maps/view fields derived after pruning. |
| strengthened `RolledMixedArmModel` | Partial | It checks that mixed-pass metadata exists and that the renderer emits some composite label. It does not prove exactness of the `(parent_no_pass, child_no_pass, cond_id, branch_kind) -> pass_nums` mapping, and it does not combine mixed-pass behavior with the D13/D16 multi-entry-on-one-edge case. |

## Summary

Verdict = RED

The main round-2 blockers are mostly fixed, but v3 still has one specification-level blocker: 5e can attribute ops to AST conditionals that 5c never materialized from observed bools. The largest new implementation hole is D17: the plan promises ungated call stacks without ungated source loading, but the current `FuncCallLocation` design cannot express that split.

Counts: 1 blocker / 1 high / 4 mediums / 1 low
