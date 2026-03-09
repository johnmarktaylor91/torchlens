# postprocess/ — 18-Step Pipeline

## What This Does
After the forward pass captures raw tensor operations, this package runs an 18-step
pipeline to clean up the graph, detect loops, assign human-readable labels, and build
the final data structures (ModuleLog, ParamLog, LayerLog).

**Order is critical** — many steps depend on prior steps' output.

## Files

| File | Steps | ~Lines | Purpose |
|------|-------|--------|---------|
| `__init__.py` | orchestrator | 228 | Dispatches to submodules, fast-mode postprocess |
| `graph_traversal.py` | 1-4 | 414 | Output layers, ancestor marking, orphan removal, distance flood |
| `control_flow.py` | 5-7 | 504 | Conditional branches (Bug #88 fix + THEN detection), module fixing, buffer cleanup |
| `loop_detection.py` | 8 | 826 | Isomorphic subgraph expansion, loop detection, layer assignment |
| `labeling.py` | 9-12 | 599 | Label generation, rename, trim/reorder, lookup keys |
| `finalization.py` | 13-18 | 608 | Undecorate, ParamLog, ModuleLog, LayerLog, mark complete |

## The 18 Steps

| Step | Function | What |
|------|----------|------|
| 1 | `_add_output_layers` | Create output nodes from model's final tensors |
| 2 | `_find_output_ancestors` | DFS from outputs — marks `is_output_ancestor` |
| 3 | `_remove_orphan_nodes` | Remove tensors not connected to any output |
| 4 | `_mark_input_output_distances` | Flood distances from inputs/outputs (conditional on flag) |
| 5 | `_mark_conditional_branches` | Find terminal booleans → backward-only flood (Bug #88 fix) + AST-based THEN detection |
| 6 | `_fix_modules_for_internal_tensors` | Infer module assignments for internally-initialized tensors |
| 7 | `_fix_buffer_layers` | Deduplicate buffers, reconnect graph edges |
| 8 | `_detect_and_label_loops` | **Loop detection** — group repeating operations into layers |
| 9 | `_map_raw_labels_to_final_labels` | Generate human-readable labels |
| 10 | `_log_final_info_for_all_layers` | Rename internal fields (raw→final labels) |
| 11 | `_rename_model_history_layer_names` | Replace raw labels in ModelLog-level fields |
| 12 | `_trim_and_reorder_layer_entry_fields` | Reorder per FIELD_ORDER, build lookup dicts |
| 13 | `_remove_unwanted_entries_and_log_remaining` | Remove entries user didn't request |
| 14 | `_undecorate_all_saved_tensors` | Remove `tl_*` attrs from saved tensors |
| 15 | `_log_time_elapsed` | Record total timing |
| 16 | `_finalize_param_logs` | Create ParamLog/ParamAccessor |
| 16.5 | `_build_layer_logs` | Group LayerPassLogs into LayerLog aggregates |
| 17 | `_build_module_logs` | Create ModuleLog/ModulePassLog/ModuleAccessor |
| 18 | `_set_pass_finished` | Mark ModelLog as complete |

## Step 5: Conditional Branch Detection (control_flow.py)

**Bug #88 fix**: `_mark_conditional_branches` floods **backward-only** (parent_layers only),
not bidirectional. Prevents non-conditional nodes' children from being falsely marked.

**THEN detection** (when `save_source_context=True`): AST-based. Finds terminal bool's
`ast.If` node via func_call_stack, checks if children's source lines fall in if-body range.
Post-validation only clears IF markings when no `ast.If` found (bool not used for control flow).

New fields: `cond_branch_then_children` (LayerPassLog), `conditional_then_edges` (ModelLog).

## Loop Detection (Step 8) — The Most Complex Step

Algorithm groups operations into "layers" (e.g., 8 iterations of sin → 1 layer, 8 passes):

1. **BFS expansion**: All nodes with same `operation_equivalence_type` start in one iso group.
   BFS explores children/parents, finding isomorphic matches across subgraphs.
2. **Iso group refinement**: Splits groups using direction-aware neighbor connectivity
   (prevents false merging of structurally different operations sharing same equiv type).
3. **Layer assignment**: Groups merged if subgraphs share parameter equiv types OR are
   adjacent (connected via equivalent operation chains).
4. **Rebuild pass assignments**: Safety net — rebuilds `same_layer_operations`, `pass_num`,
   `layer_passes_total` from scratch after all rounds.

**Core invariant**: Two operations → same layer ONLY if subgraphs share params OR are adjacent.

## Critical Ordering Dependencies
- Step 5 backward-only flood must complete before Step 6 module fixing
- Step 6 appends module addresses to `operation_equivalence_type` → affects Step 8 grouping
- Step 10 renames LayerPassLog internal fields → Step 12 builds lookup dicts with final labels
- Step 12 converts module tuples → strings in `containing_modules_origin_nested`
- Steps 16-17 build LayerLog/ModuleLog from finalized data — must run after all renaming

## Step 6 Module Suffix
`control_flow.py:82-86` appends ALL module addresses to every tensor's
`operation_equivalence_type`. Intentionally prevents operations in different modules
from being loop-grouped (e.g., `linear1.relu` != `linear2.relu`). Side effect: same
equiv group gets processed multiple times in Step 8 due to distinct suffixes.
`_rebuild_pass_assignments` handles the resulting stale references.

## Fast-Mode Postprocess
`postprocess_fast()` in `__init__.py`: minimal processing for the second pass.
Skips most steps — reuses graph structure from exhaustive pass, only updates
tensor contents for saved layers.

## Gotchas
- `_build_layer_logs` multi-pass merge: only 3 fields merged (has_input_ancestor OR,
  input_output_address char-merge, is_bottom_level_submodule_output OR). All other
  78 fields use first-pass values.
- `_build_module_logs` must NOT be called in `postprocess_fast` — `_module_build_data`
  isn't populated in fast mode.
- `_pass_finished` not reset between passes — intentional for fast-path lookups.

## Related
- [capture/](../capture/CLAUDE.md) — Produces the raw graph this package processes
- [data_classes/](../data_classes/CLAUDE.md) — LayerLog, ModuleLog, ParamLog built here
- `constants.py` — FIELD_ORDER used by Step 12
