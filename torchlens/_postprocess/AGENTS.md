# torchlens/postprocess architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

## What This Does

Transforms raw capture records into user-facing `Trace` state. The current full pipeline
has stable contract steps 0-20: graph traversal, conditional attribution, buffer fixes, loop
detection, labeling, finalization, streaming bundle finalization, and optional out
eviction plus parameter-reference release. Step order is load-bearing.

## Files

| File | Steps | Purpose |
|------|-------|---------|
| `__init__.py` | orchestrator | `postprocess()` prologue/epilogue, audit helpers, re-exports |
| `_call_site_contract.py` | 7 support | Call-site grouping side table (call-site class per raw label, never persisted) and the grouping contract checked at step 7 and by validation |
| `_executor.py` | executor | StepSpec registry (steps 1-20 in run order), `run_pipeline` |
| `_materialize.py` | 0 | Project capture events into raw `Op` state |
| `_buffer_addresses.py` | 0 support | Resolve each buffer op's registered buffer address and the initial buffer snapshots |
| `graph_traversal.py` | 1-4 | Output nodes, output ancestors, orphan removal, distances |
| `ast_branches.py` | 5 support, 11.5 | Conditional AST indexing and source variable names. Hot/cold FileIndex split: parsed ASTs (`_HeavyAst`) are released at the postprocess epilogue (`release_parsed_asts()`); span data + node-free projected calls persist, and unprojected-scope queries re-parse from RETAINED source (never disk), failing closed on anomaly |
| `control_flow.py` | 5-6 | Conditional attribution and buffer version merging |
| `loop_detection.py` | 7 | Call-site grouping (`callsite_v1`, rule in `docs/guides/grouping.md`; keys from `utils/_call_site.py`); falls back to the structural adapter when no op carries call-site evidence |
| `loop_grouping_adapter.py` | 7 fallback | Backend-neutral structural grouper (`grouper_v1`) for captures without call-site evidence (other backends, predicate-mode `tl.record`) |
| `labeling.py` | 8-11 | Final labels, renaming, lookup keys, retained layer lists (field reordering was REMOVED — see step 10 note below) |
| `finalization.py` | 12-20 | Undecorate, params, layers, modules, streaming finalization/eviction. Steps it orchestrates but does NOT implement: the step-16.5 hash lives in `utils/hashing.py` (`compute_graph_shape_hash`), the step-20 ref release in `data_classes/_trace_validation.py` (`_release_param_refs`), and the step-13 CUDA cache clear inline in the executor (`_executor.py`) |
| `_ingest_contract.py` | 0 support | Step-0 ingest contract helpers |
| `_multipass_params.py` | 14 support | Parameter union for a layer whose passes consume different parameters (split from `finalization.py`) |
| `_lazy_param_geometry.py` | 15 support | Finalize deferred geometry for lazy-at-prep params (R43 split from `finalization.py`) |
| `saved_summary.py` | 11 support | Saved-output summary refresh helpers |
| `incremental.py` | fastlog enrichment | Adds module paths to sparse recordings; `add_param_addresses` is DEAD on current builds (ActivationRecord carries no `parent_param_addresses` field, so it always raises `RecordingConfigError`) |

## Step Order

Step order is load-bearing and is the order of `STEP_REGISTRY` in `_executor.py`.
After each step runs, `run_pipeline` checks that step's postcondition (`STEP_POSTCONDITIONS`: steps 1, 8, 11, 15.5, 16, 16.5, 17) on every capture, with no env gate, and raises `MetadataInvariantError` (`postprocess_step_postcondition`) on a violation; step 1's is one output layer per attributable output, so all-unattributable outputs leave none.
Reordering steps requires the byte-identity oracles and a warnings/exception-order
review (those are pinned only by the existing order). Transient
build scratch lives in three named per-phase workspaces (`ir/workspaces.py`),
not a flat `TraceBuildState` (deleted in M10): `RawGraphWorkspace` (capture
ingress + steps 0-11, also the backend `finalize_forward_session` ownership
token), `ModuleCaptureWorkspace` (module prep/stack capture, consumed at step
16), and `WrapperRuntimeWorkspace` (wrapper hot path). All three drop at
step 17.5 (the contracted container-adoption + workspace-drop seam).

## The Ordered Steps

| Step | Function | What |
|------|----------|------|
| pre-0 | `_resolve_output_parent_labels` | Pair each output tensor with its graph parent; late-log returned-but-never-traced buffers as source events |
| 0 | `materialize_from_events` | Rebuild raw `Op` state from sealed capture events |
| 1 | `_add_output_layers` | Create dedicated output nodes (skips unattributable outputs) |
| 2 | `_find_output_ancestors` | Mark nodes connected to model output |
| 3 | `_remove_orphan_nodes` | Drop unconnected raw nodes |
| 4 | `_mark_layer_depths` | Optional input/output distance metadata |
| 5 | `_mark_conditional_branches` | AST/bool/event/edge conditional attribution |
| 6 | `_fix_buffer_layers` | Reconnect buffers; merge only re-logged reads of one recorded buffer version |
| 7 | `_detect_and_label_loops` or `_group_by_shared_params` | Recurrent grouping |
| 8 | `_map_raw_labels_to_final_labels` | Build raw-to-final label map |
| 9 | `_log_final_info_for_layers` | Write final layer/module fields |
| 10 | `_rename_model_history_layer_names` | Rename global refs (field reorder removed — scrub order is now deterministic) |
| 11 | `_build_lookup_keys_and_finalize_retained_layers` | Build lookup keys and finalize retained layer lists |
| 11.5 | `_populate_var_names` | Resolve source assignment names through `ast_branches.py` |
| 11.75 | executor `_run_step_11_75` | Resolve deferred retention decisions through the attached `CaptureSession` (saves selected payloads; runs only when a capture session is attached) |
| 12 | `_undecorate_all_saved_tensors` | Strip TorchLens attrs from saved tensors |
| 13 | `torch.cuda.empty_cache` | Optional CUDA cache clear |
| 14 | `_log_time_elapsed` | Capture timing |
| 15 | `_finalize_param_logs` | Build and complete ParamLogs |
| 15.5 | `_build_layer_logs` | Build aggregate LayerLogs |
| 16 | `_build_module_logs` | Build ModuleLogs |
| 16.5 | `compute_graph_shape_hash` | Hash graph shape before pass-finished behavior changes |
| 17 | `_set_tracing_finished` | Switch Trace to user-facing behavior |
| 17.5 | executor `_run_step_17_5` | Adopt container records; drop the three per-phase workspaces |
| 18 | `_finalize_streamed_bundle` | Finalize streamed out bundle |
| 19 | `_evict_streamed_outs` | Optional in-memory out eviction |
| 20 | `_release_param_refs` | Drop live parameter references after finalization |

## Step 5: Conditional Attribution

Step 5 builds AST indexes, classifies terminal scalar bools, materializes dense
`conditional_records`, runs a backward flood from branch bools, attributes forward arm edges,
then derives legacy THEN/ELIF/ELSE views. Canonical structures are:
- `Trace.conditional_records`
- `Trace.conditional_arm_entry_edges`
- `Trace.conditional_edge_call_indices`
- `conditional_arm_children` on `Op` and `Layer`

## equivalence_class module suffix

Module containment comes from op-creation stack snapshots, and op creation appends
the canonical module path to `equivalence_class`. No postprocess pass infers or
propagates `modules`.

## Loop Detection

Torch captures group by call site (`docs/guides/grouping.md`). At op creation
`_ops_activations._call_site_equivalence_class` keys each op by (innermost module instance,
call-site chain from `utils/_call_site.py`, func_name, output slot) and writes
`equivalence_class = "<type>@cs<N>"`; the key itself is not persisted. Step 7
(`_group_by_call_sites`) makes ops with one class passes of one layer, buffers group by buffer
address, inputs and outputs stay single, and ops with no chain fall back to their site key
position. Whether each module call of one module object repeats the ops inside it (decided
yes, 2026-10-29; repeated calls are module calls, `m:1`, `m:2`) is one predicate,
`_call_site.MODULE_CALL_IS_PASS`. When no op carries a call-site class (other backends,
predicate-mode `tl.record`), `loop_detection.py` builds a backend-neutral
`RecurrenceGroupingGraph` and applies the assignments returned by `loop_grouping_adapter.py`
(`grouper_v1`), which owns the frontier, adjacency, parameter-free false-positive guard,
grouping, and pass assignment behavior. The stamp's `detector` records which ran.

## Refresh Projection

There is no standalone `postprocess_fast()` orchestrator. Refresh captures run through the
full `postprocess()` entry point with the established Trace state; Step 0 reads the sealed
`CapturedRunCore.events` snapshot (cloned with independent mutable dict fields) when a
`CaptureSession` is attached, and `RefreshProjector` applies refreshed payloads onto the
existing graph. Saved-output counters are refreshed after retained layers are finalized;
module aggregation remains part of the ordered full pipeline.

There is no `postprocess_fast()` orchestrator. Step 0 reads the sealed
`CapturedRunCore.events` snapshot (cloned with independent mutable dict fields) when a
`CaptureSession` is attached, `RefreshProjector` applies refreshed payloads onto the
existing graph, and the single `postprocess()` entry point preserves the ordered
Step 0-20 contracts. Saved-output summaries are refreshed after Step 11 finalizes the
retained op list.

## Executor

`postprocess()` keeps the prologue (pre-0 + step-0 materialize block), the
no-layers early exit, and the freeze epilogue; steps 1-20 run through
`_executor.run_pipeline` over `STEP_REGISTRY`. Every step body resolves its
callable through the `torchlens._postprocess` module namespace AT CALL
TIME, so monkeypatching a step function on the module still works
(`tests/test_postprocess_executor.py` proves it). Step 18's `should_run` IS the
streaming snapshot point (context-writing, never trace-writing); step 19
gates on the snapshot; `should_run` evaluates exactly once per step.

## Step 5 Conditional Branch Detection

- Implementation is in `control_flow.py` with AST support from `ast_branches.py`.
- Primary data is cond-id-aware: `conditional_records`, `conditional_arm_entry_edges`,
  `conditional_edge_call_indices`, `conditional_arm_children`.
- Legacy THEN/ELIF/ELSE fields are derived compatibility views.
- Backward flood is parent-only; do not make it bidirectional.
- Ternary `IfExp` attribution depends on source `col_offset` when arms share a line.

## Module Suffixes

Capture-time op creation appends module-address information to `equivalence_class`
so identical ops in different modules do not get loop-grouped together (the structural
grouper's classes; torch captures replace them with call-site classes). Step 7's
`loop_detection.py` seam adapts Trace ops to the live backend-neutral implementation
in `loop_grouping_adapter.py`; do not duplicate grouping logic in the Trace adapter.

## Step 11 Lookup-Key Finalization

`_build_lookup_keys_and_finalize_retained_layers()` applies lookup-key construction while
preserving dependencies needed for replay/intervention when those modes request them.

## Steps 18-20 Streaming and Release

Streaming bundle finalization and eviction live in `finalization.py`. These steps coordinate
with `_io.streaming.BundleStreamWriter` and lazy out refs. Never evict graph-connected
training outs. Step 20 then releases live parameter references.

## Gotchas

- `_build_layer_logs()` merges only selected fields across ops; most fields use first pass.
- `_tracing_finished` is not reset between exhaustive and fast ops.
- Conditional cleanup must update both primary cond-id structures and derived views.
- Changing label formats requires checking visualization, validation, I/O, intervention, and
  bundle supergraph code.
