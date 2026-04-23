# Adversarial Review: `if` / `elif` / `else` Attribution Sprint Plan

Scope reviewed:
- Full plan in `.project-context/plans/if-else-attribution/plan.md`
- Research grounding in `.project-context/research/ast-design.md`, `adversarial.md`, `prior-art.md`
- Current implementation surfaces in `torchlens/postprocess/control_flow.py`, data classes, constants, visualization, labeling, cleanup, and validation

## Finding 1: `save_source_context=False` is internally inconsistent with D3/D9/D12
- Severity: blocker
- What's wrong: D3 and D12 say Step 5 should pre-classify bools and remove the old "mark first, clear later" behavior, but D9 keeps AST attribution gated on `save_source_context=True`, and the test matrix's `SaveSourceContextOffModel` still expects legacy IF-edge behavior with no branch attribution. Those three positions do not compose.
- Why it matters: current Step 5 floods backward from **all** terminal bools and only does AST validation when `save_source_context` is on (`torchlens/postprocess/control_flow.py:51-77`, `164-193`). If the new classifier is also gated off, 5d has no way to distinguish real `if`/`elif` tests from `assert`, `bool()`, `while`, or comprehension filters. Either false positives come back, or IF edges disappear entirely.
- Suggested fix: change D9 to say: "AST classification needed for Step 5 runs whenever `func_call_stack` has `(file, line)`; `save_source_context` only gates rich source snippets and user-facing source display." Then update Step 5b/5d and the `SaveSourceContextOffModel` row to match that rule.
- Test: add a `save_source_context=False` case with `assert tensor_cond` and verify no `conditional_branch_edges` are emitted.

## Finding 2: `if bool(tensor_cond):` is a real branch that the classifier would suppress
- Severity: high
- What's wrong: Step 5b says to choose the innermost non-`unknown` bool consumer, and D3 treats `bool_cast` as non-branch. For `if bool(tensor_cond):`, the innermost consumer is the `bool(...)` call, not the enclosing `If.test`, so the plan classifies a genuine branch as non-branch.
- Why it matters: this is not the same case as `flag = bool(tensor)` with no branch. The runtime bool conversion still drives an `if`, but the proposed rule would set `bool_is_branch=False` and drop attribution. The problem follows directly from the plan's `classify_bool()` rule plus the explicit bool-consumer index in `ast_branches.py`.
- Suggested fix: change D3/Step 5b so that a `bool(...)` consumer nested inside an `If.test` or flattened `elif` test is normalized to branch participation. If you still want to preserve the cast information, record it separately, e.g. `bool_context_kind="if_test"` plus `bool_wrapper_kind="bool_cast"`.
- Test: add `IfBoolCastModel` with `if bool(x.sum() > 0): ...` and assert `bool_is_branch=True`.

## Finding 3: Step 5e only looking at branch-start nodes misses valid branch-entry edges
- Severity: blocker
- What's wrong: Step 5e says "For each branch-start node's children, diff parent vs child stack to determine which branch was entered." That preserves the current assumption that branch entry must appear on an edge out of a branch-start node.
- Why it matters: that is false when the first op inside a branch depends only on a parameter, buffer, or constant rather than on the predicate's dataflow ancestors. Current THEN detection already has this limitation because it only inspects `start_node.child_layers` (`torchlens/postprocess/control_flow.py:146-163`). Example:

```python
if x.sum() > 0:
    y = self.bias + 1
else:
    y = self.bias - 1
```

The branch-local ops should get THEN/ELSE attribution, but there may be no edge from the branch-start node to those ops, so no labeled entry edge is ever emitted.
- Suggested fix: change Step 5e to diff stacks on **all** forward edges `(parent, child)`, not just edges whose parent is a branch-start node. Keep `conditional_branch_edges` as the legacy IF-edge structure, but derive THEN/ELIF/ELSE entry edges from the first gained stack item on any edge.
- Test: add `BranchUsesOnlyParameterModel` and assert that an arm label is emitted on the parameter-to-op edge.

## Finding 4: The plan misses Step 11/cleanup/export integration for the new fields
- Severity: high
- What's wrong: the plan adds new raw-label-bearing fields (`cond_branch_else_children`, `cond_branch_elif_children`, `conditional_elif_edges`, `conditional_else_edges`, `conditional_events.bool_layers`) but does not explicitly update the rename, cleanup, and export surfaces that currently hardcode the old conditional fields only.
- Why it matters: current rename logic only knows about `cond_branch_start_children`, `cond_branch_then_children`, `conditional_branch_edges`, and `conditional_then_edges` (`torchlens/postprocess/labeling.py:214-228`, `551-562`). Cleanup only filters the two existing edge lists (`torchlens/data_classes/cleanup.py:157-166`, `212-217`). Capture-time field initialization only sets the current conditional fields (`torchlens/capture/source_tensors.py:230-248`, `torchlens/capture/output_tensors.py:171-175`). `to_pandas()` does not even expose today's conditional fields (`torchlens/data_classes/interface.py:406-507`).
- Suggested fix: change Phase 1 and Deliverables to explicitly include `torchlens/postprocess/labeling.py`, `torchlens/data_classes/cleanup.py`, `torchlens/capture/source_tensors.py`, `torchlens/capture/output_tensors.py`, and `torchlens/data_classes/interface.py`. Add a rule: every new label-bearing field must be wired through rename, cleanup, and export before Phase 1 is considered green.
- Test: run a keep/trim scenario (`keep_unsaved_layers=False`) and assert no new conditional field contains a removed raw label after Step 12.

## Finding 5: Invariant 6 rejects legitimate reconvergence
- Severity: high
- What's wrong: invariant 6 says "For a parent→child edge, the child's stack contains the parent's stack as a prefix (or enters a new branch)." That only allows equality or suffix growth.
- Why it matters: legal control-flow exits do the opposite. In `ReconvergingBranchesModel`, the edge from a branch-local op to a post-merge op should drop the innermost `(cond_id, branch_kind)` suffix. The plan's own test matrix includes `ReconvergingBranchesModel`, so invariant 6 would false-positive on a case the plan claims to support.
- Suggested fix: rewrite invariant 6 to allow suffix pops as well as suffix pushes. Concrete replacement: "For any parent→child edge, one stack must be a prefix of the other; gains represent branch entry, drops represent branch exit/reconvergence."
- Test: explicitly assert that reconverging edges pass invariants while a non-prefix reordering fails.

## Finding 6: Invariant 7 is attached to the wrong data structure
- Severity: medium
- What's wrong: invariant 7 requires `cond_branch_elif_children` keys to be contiguous from 1 for a given `cond_id`. That is reasonable for the normalized `ConditionalEvent`, but not for every per-node adjacency dict.
- Why it matters: a particular parent node may legitimately have only an `elif_2` entry because it has no direct edge into `elif_1`, or because `elif_1` contains no captured op reachable from that parent. The same issue gets worse after the LayerLog union step, where per-parent sparse coverage is expected.
- Suggested fix: move the contiguity check from per-node `cond_branch_elif_children` to `ConditionalEvent.branch_ranges` / `branch_test_spans`. For per-node child maps, only enforce that keys are positive ints and children exist.
- Test: add a model where only the second `elif` arm has a direct edge from a given parent, and ensure invariants still pass.

## Finding 7: Rolled-mode branch labels lose pass-level truth
- Severity: high
- What's wrong: the LayerLog merge in the plan unions `cond_branch_*_children` after stripping pass suffixes, but it does not introduce any edge-level pass map for branch arms.
- Why it matters: the same rolled parent→child edge can be THEN on pass 1 and ELSE on pass 2. Current rolled Graphviz and dagua rendering already consume only aggregate child lists plus generic pass labels (`torchlens/visualization/rendering.py:985-996`, `1162-1188`; `torchlens/visualization/dagua_bridge.py:342-354`, `478-490`). With the proposed union semantics, the renderer has no truthful way to label that edge.
- Suggested fix: add an explicit rolled-edge data structure, e.g. `conditional_edge_passes[(parent_no_pass, child_no_pass, cond_id, branch_kind)] -> [pass_nums]`, and update the Visualization section to define how conflicting branch kinds are rendered (`THEN 1 / ELSE 2`, `mixed`, etc.).
- Test: add a recurrent model where the same no-pass edge is THEN on one pass and ELSE on another, and assert the rolled renderer does not collapse it to a single-arm label.

## Finding 8: Visualization label precedence is underspecified and currently collides
- Severity: medium
- What's wrong: D11 says to extend the existing IF/THEN label block for ELIF/ELSE, but the current Graphviz code already uses a single `edge_dict["label"]` and overwrites IF with THEN (`torchlens/visualization/rendering.py:984-996`). Argument labels then append into the same label field (`1049-1090`). Dagua likewise picks exactly one edge type (`torchlens/visualization/dagua_bridge.py:342-354`).
- Why it matters: once ELIF/ELSE are added, the renderer still needs a precedence rule. Is a THEN edge also an IF edge? What wins when the same edge also needs an argument label? What happens in rolled mode when pass annotations are present? The plan does not answer any of those.
- Suggested fix: change the Visualization section to define explicit composition rules. Concrete recommendation:
  - `IF` is only used for condition-chain edges (`conditional_branch_edges`)
  - arm-entry edges use `THEN` / `ELIF n` / `ELSE` and suppress `IF`
  - argument labels move to `headlabel`/`xlabel` so they do not compete with branch labels
  - dagua_bridge must use the same precedence order
- Test: add a model where a branch-entry edge is also argument-labeled and verify the rendered label is deterministic.

## Finding 9: Per-frame attribution does not have enough runtime identity to resolve scopes robustly
- Severity: high
- What's wrong: D4 and the `attribute_op(func_call_stack)` design rely on per-frame scope resolution, but current `FuncCallLocation` only stores `file`, `line_number`, and `func_name` (`torchlens/data_classes/func_call_location.py:50-69`). Stack capture only records `co_filename`, `co_name`, and `f_lineno`, with a best-effort name lookup for the function object (`torchlens/utils/introspection.py:404-464`).
- Why it matters: line-only scope resolution is under-specified for nested helpers with repeated names, lambdas, decorated wrappers, and local functions defined on the same line. The plan's `FileIndex` talks about `function_qualname`, lambdas, and smallest-containing-function lookup, but the runtime evidence currently retained by TorchLens is not enough to do that reliably.
- Suggested fix: amend the Foundation section and `ast_branches.py` API to require `FuncCallLocation` to store `code_firstlineno` (minimum) and preferably a qualname/code-object identity. Then resolve scopes by `(filename, code_firstlineno, func_name)` before falling back to line-only heuristics.
- Test: add `NestedHelperSameNameModel` and `LambdaBranchModel` so the implementation cannot silently pass with line-only scope matching.

## Summary: verdict = RED
- The AST-first direction is still viable.
- The plan is not implementable as-is without correctness regressions.
- The biggest issues are the D3/D9/D12 contradiction, the branch-entry edge restriction in 5e, and the lack of a truthful rolled-edge story.

## Blockers
- `save_source_context=False` semantics contradict D3/D9/D12 and leave Step 5 with no correct bool-selection rule.
- Step 5e only labels edges out of branch-start nodes, so valid branch entries can be missed entirely.

## Highs
- `if bool(tensor_cond):` would be misclassified as non-branch.
- New fields are not wired through rename/cleanup/export surfaces.
- Invariant 6 would reject legitimate reconvergence.
- Rolled-mode visualization has no pass-level arm metadata, so unioned child lists can lie.
- Per-frame runtime metadata is too weak for robust scope attribution in nested helper/lambda/decorator cases.

## Mediums/Lows
- Invariant 7 applies contiguity to the wrong structure.
- Visualization label precedence/composition is not specified, and the current renderer already overwrites IF with THEN.
