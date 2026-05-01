# postprocess/ - Graph Cleanup and Finalization

## What This Does
Transforms raw capture records into user-facing `ModelLog` state. The current full pipeline
has 20 ordered steps: graph traversal, conditional attribution, module/buffer fixes, loop
detection, labeling, finalization, streaming bundle finalization, and optional activation
eviction. Step order is load-bearing.

## Files

| File | Steps | Purpose |
|------|-------|---------|
| `__init__.py` | orchestrator | Full `postprocess()` and `postprocess_fast()` |
| `graph_traversal.py` | 1-4 | Output nodes, output ancestors, orphan removal, distances |
| `ast_branches.py` | Step 5 support | AST indexing and branch-scope records for conditionals |
| `control_flow.py` | 5-7 | Conditional attribution, module containment fixes, buffer dedup |
| `loop_detection.py` | 8 | Recurrent/loop grouping and shared-param grouping |
| `labeling.py` | 9-12 | Final labels, renaming, removal, lookup keys, field ordering |
| `finalization.py` | 13-20 | Undecorate, params, layers, modules, hash, streaming finalization/eviction |
| `incremental.py` | fastlog enrichment | Adds module paths and param addresses to sparse recordings |

## The Ordered Steps

| Step | Function | What |
|------|----------|------|
| 1 | `_add_output_layers` | Create dedicated output nodes |
| 2 | `_find_output_ancestors` | Mark nodes connected to model output |
| 3 | `_remove_orphan_nodes` | Drop unconnected raw nodes |
| 4 | `_mark_input_output_distances` | Optional input/output distance metadata |
| 5 | `_mark_conditional_branches` | AST/bool/event/edge conditional attribution |
| 6 | `_fix_modules_for_internal_tensors` | Infer module containment for internal tensors |
| 7 | `_fix_buffer_layers` | Deduplicate and reconnect buffers |
| 8 | `_detect_and_label_loops` or `_group_by_shared_params` | Recurrent grouping |
| 9 | `_map_raw_labels_to_final_labels` | Build raw-to-final label map |
| 10 | `_log_final_info_for_all_layers` | Write final layer/module fields |
| 11 | `_rename_model_history_layer_names` and `_trim_and_reorder_model_history_fields` | Rename global refs |
| 12 | `_remove_unwanted_entries_and_log_remaining` | Apply save policy and lookup keys |
| 13 | `_undecorate_all_saved_tensors` | Strip TorchLens attrs from saved tensors |
| 14 | `torch.cuda.empty_cache` | Optional CUDA cache clear |
| 15 | `_log_time_elapsed` | Capture timing |
| 16 | `_finalize_param_logs` | Build and complete ParamLogs |
| 16.5 | `_build_layer_logs` | Build aggregate LayerLogs |
| 17 | `_build_module_logs` | Build ModuleLogs |
| 17.5 | `compute_graph_shape_hash` | Hash graph shape before pass-finished behavior changes |
| 18 | `_set_pass_finished` | Switch ModelLog to user-facing behavior |
| 19 | `_finalize_streamed_bundle` | Finalize streamed activation bundle |
| 20 | `_evict_streamed_activations` | Optional in-memory activation eviction |

## Step 5: Conditional Attribution
Step 5 builds AST indexes, classifies terminal scalar bools, materializes dense
`conditional_events`, runs a backward flood from branch bools, attributes forward arm edges,
then derives legacy THEN/ELIF/ELSE views. Canonical structures are:
- `ModelLog.conditional_events`
- `ModelLog.conditional_arm_edges`
- `ModelLog.conditional_edge_passes`
- `cond_branch_children_by_cond` on `LayerPassLog` and `LayerLog`

## Loop Detection
Loop detection groups repeated operations by operation equivalence, expands isomorphic
subgraphs, refines groups by neighbor connectivity, merges adjacent/shared-param groups, and
rebuilds pass assignments. Step 6 appends module suffixes to equivalence types, so the rebuild
is necessary, not defensive.

## Fast Mode
`postprocess_fast()` is for second-pass selective activation saves. It reuses graph structure,
labels, module data, and loop groupings from the exhaustive pass. It must not call
`_build_module_logs()` because `_module_build_data` is not repopulated in fast mode.
