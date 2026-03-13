# postprocess/ — Implementation Guide

## Critical Ordering Dependencies
- Step 5 backward-only flood must complete before Step 6 module fixing
- Step 6 appends module addresses to `operation_equivalence_type` → affects Step 8 grouping
- Step 10 renames LayerPassLog internal fields → Step 12 builds lookup dicts with final labels
- Step 12 converts module tuples → strings in `containing_modules`
- Steps 16-17 build LayerLog/ModuleLog from finalized data — must run after all renaming

## Step 5: Conditional Branch Detection (control_flow.py)

**Bug #88 fix**: `_mark_conditional_branches` floods **backward-only** (parent_layers only),
not bidirectional. Prevents non-conditional nodes' children from being falsely marked.

**THEN detection** (when `save_source_context=True`): AST-based. Finds terminal bool's
`ast.If` node via func_call_stack, checks if children's source lines fall in if-body range.

New fields: `cond_branch_then_children` (LayerPassLog), `conditional_then_edges` (ModelLog).

## Step 6 Module Suffix
`control_flow.py:82-86` appends ALL module addresses to every tensor's
`operation_equivalence_type`. Intentionally prevents operations in different modules
from being loop-grouped. Side effect: same equiv group gets processed multiple times
in Step 8. `_rebuild_pass_assignments` handles the resulting stale references.

## Fast-Mode Postprocess
`postprocess_fast()` in `__init__.py`: minimal processing for the second pass.
Skips most steps — reuses graph structure from exhaustive pass, only updates
tensor contents for saved layers.

## Gotchas
- `_build_layer_logs` multi-pass merge: only 3 fields merged (has_input_ancestor OR,
  io_role char-merge, is_leaf_module_output OR). All other 78 fields use first-pass values.
- `_build_module_logs` must NOT be called in `postprocess_fast` — `_module_build_data`
  isn't populated in fast mode.
- `_pass_finished` not reset between passes — intentional for fast-path lookups.
