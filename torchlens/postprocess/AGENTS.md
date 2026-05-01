# postprocess/ - Implementation Guide

## Critical Ordering Dependencies
- Steps 1-3 must run before Step 5 so conditional attribution sees an orphan-free graph.
- Step 6 mutates `operation_equivalence_type`; it must precede Step 8.
- Step 8 must precede Step 9 because label mapping uses recurrent groups.
- Step 10 must precede Step 12 because lookup keys depend on finalized module hierarchy info.
- Step 11 must rename global refs before unwanted-entry removal.
- Step 16.5 must precede Step 17 because `ModuleLog.all_layers` points to `LayerLog` keys.
- Step 17.5 computes `graph_shape_hash` before `_set_pass_finished` changes access behavior.
- Steps 19-20 are only for streamed activation bundles.

## Step 5 Conditional Branch Detection
- Implementation is in `control_flow.py` with AST support from `ast_branches.py`.
- Primary data is cond-id-aware: `conditional_events`, `conditional_arm_edges`,
  `conditional_edge_passes`, `cond_branch_children_by_cond`.
- Legacy THEN/ELIF/ELSE fields are derived compatibility views.
- Backward flood is parent-only; do not make it bidirectional.
- Ternary `IfExp` attribution depends on source `col_offset` when arms share a line.

## Step 6 Module Suffix
`_fix_modules_for_internal_tensors()` appends module-address information to
`operation_equivalence_type` so identical ops in different modules do not get loop-grouped
together. This can leave stale group references until Step 8 rebuilds assignments.

## Step 12 Save Policy
`_remove_unwanted_entries_and_log_remaining()` applies `layers_to_save`,
`keep_unsaved_layers`, intervention-readiness retention, and lookup-key construction. It must
preserve dependencies needed for replay/intervention when those modes request them.

## Steps 19-20 Streaming
Streaming bundle finalization and eviction live in `finalization.py`. These steps coordinate
with `_io.streaming.BundleStreamWriter` and lazy activation refs. Never evict graph-connected
training activations.

## Fast-Mode Postprocess
`postprocess_fast()` only copies output activations from parents, trims/renames as needed,
removes unwanted entries, undecorates tensors, builds `LayerLog` aggregates, and sets
pass-finished. It intentionally skips graph traversal, conditionals, loop detection, label
mapping, and module building.

## Gotchas
- `_build_layer_logs()` merges only selected fields across passes; most fields use first pass.
- `_build_module_logs()` must not run in fast mode.
- `_pass_finished` is not reset between exhaustive and fast passes.
- Conditional cleanup must update both primary cond-id structures and derived views.
- Changing label formats requires checking visualization, validation, I/O, intervention, and
  bundle supergraph code.
