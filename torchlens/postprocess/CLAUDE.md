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
| 5 | `_mark_conditional_branches` | Find terminal booleans → backward-only flood + AST-based THEN detection |
| 6 | `_fix_modules_for_internal_tensors` | Infer module assignments for internally-initialized tensors |
| 7 | `_fix_buffer_layers` | Deduplicate buffers, reconnect graph edges |
| 8 | `_detect_and_label_loops` | Loop detection — group repeating operations into layers |
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

## Loop Detection (Step 8) — The Most Complex Step

Algorithm groups operations into "layers" (e.g., 8 iterations of sin → 1 layer, 8 passes):

1. **BFS expansion**: All nodes with same `operation_equivalence_type` start in one iso group.
   BFS explores children/parents, finding isomorphic matches across subgraphs.
2. **Iso group refinement**: Splits groups using direction-aware neighbor connectivity.
3. **Layer assignment**: Groups merged if subgraphs share parameter equiv types OR are
   adjacent (connected via equivalent operation chains).
4. **Rebuild pass assignments**: Safety net — rebuilds `recurrent_group`, `pass_num`,
   `num_passes` from scratch after all rounds.

**Core invariant**: Two operations → same layer ONLY if subgraphs share params OR are adjacent.

## How It Connects

Receives raw graph from `capture/`. Produces finalized ModelLog with LayerLog/ModuleLog/ParamLog
that `visualization/` renders and `validation/` checks. The pipeline is strictly sequential —
no parallelism within a single postprocess run.
