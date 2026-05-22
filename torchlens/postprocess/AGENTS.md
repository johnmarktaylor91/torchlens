# postprocess/ - Implementation Guide

## Critical Ordering Dependencies
- Steps 1-3 must run before Step 5 so conditional attribution sees an orphan-free graph.
- Module suffixes must already be present on `equivalence_class` before Step 7.
- Step 7 must precede Step 8 because label mapping uses recurrent groups.
- Step 9 must precede Step 11 because lookup keys depend on finalized module hierarchy info.
- Step 10 must rename global refs before unwanted-entry removal.
- Step 15.5 must precede Step 16 because `Module.layers` points to `Layer` keys.
- Step 16.5 computes `graph_shape_hash` before `_set_tracing_finished` changes access behavior.
- Steps 18-19 are only for streamed out bundles.

## Step 5 Conditional Branch Detection
- Implementation is in `control_flow.py` with AST support from `ast_branches.py`.
- Primary data is cond-id-aware: `conditional_records`, `conditional_arm_entry_edges`,
  `conditional_edge_call_indices`, `conditional_arm_children`.
- Legacy THEN/ELIF/ELSE fields are derived compatibility views.
- Backward flood is parent-only; do not make it bidirectional.
- Ternary `IfExp` attribution depends on source `col_offset` when arms share a line.

## Module Suffixes
Capture-time op creation appends module-address information to `equivalence_class`
so identical ops in different modules do not get loop-grouped together. Loop
detection still rebuilds assignments after expansion to clear stale group references.

## Step 11 Save Policy
`_remove_unwanted_entries_and_log_remaining()` applies lookup-key construction while
preserving dependencies needed for replay/intervention when those modes request them.

## Steps 18-19 Streaming
Streaming bundle finalization and eviction live in `finalization.py`. These steps coordinate
with `_io.streaming.BundleStreamWriter` and lazy out refs. Never evict graph-connected
training outs.

## Fast-Mode Postprocess
`postprocess_fast()` only copies output outs from parents, trims/renames as needed,
removes unwanted entries, undecorates tensors, builds `Layer` aggregates, and sets
pass-finished. It intentionally skips graph traversal, conditionals, loop detection, label
mapping, and module building.

## Gotchas
- `_build_layer_logs()` merges only selected fields across ops; most fields use first pass.
- `_build_module_logs()` must not run in fast mode.
- `_tracing_finished` is not reset between exhaustive and fast ops.
- Conditional cleanup must update both primary cond-id structures and derived views.
- Changing label formats requires checking visualization, validation, I/O, intervention, and
  bundle supergraph code.
