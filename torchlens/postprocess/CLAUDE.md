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
| `control_flow.py` | 5-7 | 504 | Six-phase conditional attribution, module fixing, buffer cleanup |
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
| 5 | `_mark_conditional_branches` | Six-phase conditional attribution: AST indexing, bool classification, dense event IDs, branch-edge attribution |
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

## Step 5: Conditional Branch Attribution (control_flow.py)

Step 5 is a six-phase pipeline:

1. **5a: Build file indexes.** Parse source files once and cache AST indexes used for
   bool classification and per-op attribution.
2. **5b: Classify terminal bools.** Walk `func_call_stack` frames and classify each
   terminal scalar bool by AST context (`if_test`, `elif_test`, `ifexp`, non-branch kinds).
   This stage produces structural `ConditionalKey`s only.
3. **5c: Materialize events and dense IDs.** `ModelLog.conditional_events` is built in
   first-seen order, dense `cond_id`s are assigned once, and temporary bool keys are
   rewritten to ints. This is the sole key-to-id translation stage.
4. **5d: Backward IF flood.** Starting from branch-participating bools only, flood
   backward through `parent_layers` to mark `cond_branch_start_children` and
   `conditional_branch_edges`. This preserves the PR #127 backward-only fix.
5. **5e: Forward attribution and arm edges.** Attribute each op to its enclosing branch
   stack and diff parent/child stacks across every forward edge. Gained arms populate
   `ModelLog.conditional_arm_edges` and per-node `cond_branch_children_by_cond`.
6. **5f: Derived views and rolled-pass metadata.** Legacy THEN/ELIF/ELSE views are derived
   from the primary cond-id-aware structures, and `ModelLog.conditional_edge_passes`
   records which passes used each rolled arm edge.

Primary conditional structures:
- `ModelLog.conditional_events`: dense event table for `if` / `elif` / `else` chains and
  ternaries (`IfExp`).
- `ModelLog.conditional_arm_edges`: `(cond_id, branch_kind) -> [(parent, child)]`.
- `ModelLog.conditional_edge_passes`: `(parent_no_pass, child_no_pass, cond_id, branch_kind)
  -> [pass_num, ...]` for rolled-mode divergence.
- `cond_branch_children_by_cond` on `LayerPassLog` / `LayerLog`: per-node child mapping keyed
  by condition id, preserving multi-arm entry.

User-visible behavior:
- Graphviz uses the conditional structures to render `THEN`, `ELIF`, and `ELSE` edge labels.
- Ternary attribution is first-class; `col_offset` is load-bearing when both ternary arms
  share a source line.
- `save_source_context=False` does **not** disable conditional attribution. Identity fields on
  `FuncCallLocation` are still captured; only rich source-text accessors stay empty.

Deferred in this sprint:
- `dagua_bridge.py` conditional-edge integration
- `elk_layout.py` conditional rendering
- While-loop body attribution

Reference: `.project-context/plans/if-else-attribution/plan.md` (v7).

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
