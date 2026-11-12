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
| `__init__.py` | orchestrator | `postprocess()` prologue/epilogue and the transient-state drop |
| `_call_site_contract.py` | 7 support | Call-site grouping side table (call-site class per raw label, never persisted) and the grouping contract checked at step 7 and by validation |
| `_pipeline.py` | pipeline | `PIPELINE`: the named step records in run order (name, home module, trace fields read and written, gate, postcondition) and `run_pipeline` |
| `_executor.py` | executor | `StepContext` and the per-step adapters (`_run_<name>`), gates (`_should_run_<name>`) and postconditions (`_<name>_violation`) the pipeline points at |
| `_materialize.py` | 0 | Project capture events into raw `Op` state |
| `_column_passes.py` | 2 | Pass bodies ported to the `OpColumns` view (step 2 `find_output_ancestors`), each with its facade body as the fallback |
| `_buffer_addresses.py` | 0 support | Resolve each buffer op's registered buffer address and the initial buffer snapshots |
| `graph_traversal.py` | 1-4 | Output nodes, output ancestors, orphan removal, distances; `_orphan_seeds.py` owns flood seeds, call-group closure and pre-removal identity/payload bookkeeping |
| `ast_branches.py` | 5 support, 11.5 support | Conditional AST indexing and source variable name resolution. Hot/cold FileIndex split: parsed ASTs (`_HeavyAst`) are released at the postprocess epilogue (`release_parsed_asts()`); span data + node-free projected calls persist, and unprojected-scope queries re-parse from RETAINED source (never disk), failing closed on anomaly |
| `_branch_source.py` | 5 support | Source text for `ast_branches.py`: disk files, and the `linecache` text of a notebook cell with no file (`mtime` None entries only). A cell name seen with two different texts fails closed for the process (`_interactive_file_index`). Also the source-range containment predicates |
| `control_flow.py` | 5-6 | Conditional attribution and buffer version merging |
| `loop_detection.py` | 7 | Call-site grouping (`callsite_v2`, rule in `docs/guides/grouping.md`; keys from `utils/_call_site.py`); falls back to the structural adapter when no op carries call-site evidence |
| `loop_grouping_adapter.py` | 7 fallback | Backend-neutral structural grouper (`grouper_v1`) for captures without call-site evidence (other backends, predicate-mode `tl.record`) |
| `labeling.py` | 8-11 | Final labels, renaming, lookup keys, retained layer lists (field reordering was REMOVED — see step 10 note below) |
| `finalization.py` | 12-20 | Undecorate, params, layers, modules, streaming finalization/eviction. Steps it orchestrates but does NOT implement: the step-16.5 hash lives in `utils/hashing.py` (`compute_graph_shape_hash`), the step-20 ref release in `data_classes/_trace_removal.py` (`release_param_refs`), and the step-13 CUDA cache clear inline in the executor (`_executor.py`) |
| `_ingest_contract.py` | 0 support | Step-0 ingest contract helpers |
| `_streamed_tensor_types.py` | 18 support | Bind published chunk identities and resolve streamed payload dtype names |
| `_multipass_params.py` | 14 support | Parameter union for a layer whose passes consume different parameters (split from `finalization.py`) |
| `_lazy_param_geometry.py` | 15 support | Finalize deferred geometry for lazy-at-prep params (R43 split from `finalization.py`) |
| `_column_passes_finalize.py` | 15.5 | Step 15.5 body: leaves one pending state per layer, holding the facts the scan kept, and builds a `Layer` from them on its first lookup (`build_layer_state`); a layer the scan cannot describe is merged inside the step, so no merge runs after the seal; also the merge, conditional and view helpers; `finalization._build_layer_logs` delegates to it (split from `finalization.py`) |
| `_layer_row_scan.py` | 15.5 | The raw scan of the op rows behind step 15.5: groups passes by layer, stamps the op columns the step owns, indexes ops by pass, and keeps the facts of each layer whose passes agree (`pending_states`) |
| `_finished_marks.py` | 17 | Step 17 body: per-op finished flag written on the raw rows of a building store, the facade body as the fallback; `finalization._set_tracing_finished` delegates to it |
| `_replay_template_cells.py` | 9 | Replay templates as a derived on-read family: step 9 keeps each raw `args_template` and `kwargs_template` in a side column indexed by row (`_trace_core/side_columns.py`), with the trace's label map held once, and stores the field's one shared marker in the cell; the first read renames its labels and binds its `ParentRef` display keys |
| `saved_summary.py` | 11 support | Saved-output summary refresh helpers (`_refresh_fast_saved_summary`) |
| `source_provenance.py` | 11, 11.5 | Provenance disclosure warning (`_warn_unattributed_tensor_args`) and source variable names (`_populate_var_names`) |
| `incremental.py` | fastlog enrichment | Adds module path strings to sparse recordings (`Recording.enrich`) |

## Step Order

Step order is load-bearing and is the tuple order of `PIPELINE` in `_pipeline.py`; steps are
named, never numbered (the "Step N" numerals below and in the `_phase_timings` bucket keys are
the historical labels, kept because the bucket text is persisted). After each step runs,
`run_pipeline` checks the step's `postcondition` when it declares one (add_output_layers,
map_final_labels, build_lookup_keys, build_layer_logs, build_module_logs, hash_graph_shape,
mark_tracing_finished) on every capture, with no env gate, and raises `MetadataInvariantError`
(`postprocess_step_postcondition`) on a violation; add_output_layers' is one output layer per
attributable output, so all-unattributable outputs leave none.
Reordering steps requires the byte-identity oracles and a warnings/exception-order
review (those are pinned only by the existing order). Transient
build scratch lives in three named per-phase workspaces (`ir/workspaces.py`),
not a flat `TraceBuildState` (deleted in M10): `RawGraphWorkspace` (capture
ingress + steps 0-11, also the backend `finalize_forward_session` ownership
token), `ModuleCaptureWorkspace` (module prep/stack capture, consumed at step
16), and `WrapperRuntimeWorkspace` (wrapper hot path). All three drop at
step 17.5 (the contracted container-adoption + workspace-drop seam).

## The Ordered Steps

| Label | Pipeline name | Function | What |
|------|------|----------|------|
| pre-0 | (prologue) | `_resolve_output_parent_labels` | Pair each output tensor with its graph parent; late-log returned-but-never-traced buffers as source events |
| 0 | (prologue) | `materialize_from_events` | Rebuild raw `Op` state from sealed capture events |
| 1 | `add_output_layers` | `_add_output_layers` | Create dedicated output nodes (skips unattributable outputs) |
| 2 | `find_output_ancestors` | `_find_output_ancestors` | Mark nodes connected to model output |
| 3 | `remove_orphan_nodes` | `_remove_orphan_nodes` | Drop unconnected raw nodes |
| 4 | `mark_layer_depths` | `_mark_layer_depths` (`_layer_depths.py`: raw-row column body, facade body as the fallback) | Optional input/output distance metadata |
| 5 | `mark_conditional_branches` | `_mark_conditional_branches` | AST/bool/event/edge conditional attribution |
| 6 | `fix_buffer_layers` | `_fix_buffer_layers` | Reconnect buffers; merge only re-logged reads of one recorded buffer version |
| 7 | `group_recurrent_layers` | `_detect_and_label_loops` or `_group_by_shared_params` | Recurrent grouping |
| 8 | `map_final_labels` | `_map_raw_labels_to_final_labels` (`_label_columns.py`: integer label parts and strings by column, facade body as the fallback; tail `_rename_model_history_layer_names` remaps the trace-level references) | Final labels and raw-to-final label maps |
| 9 | `log_final_layer_info` | `_log_final_info_for_layers` | Write final layer/module fields; remap `output_layers` after numbering the outputs |
| 11 | `build_lookup_keys` | `_build_lookup_keys_and_finalize_retained_layers` | Build lookup keys and finalize retained layer lists |
| 11.5 | `populate_source_var_names` | `_populate_var_names` | Resolve source assignment names through `ast_branches.py` |
| 11.75 | `resolve_deferred_retention` | executor `_run_resolve_deferred_retention` | Resolve deferred retention decisions through the attached `CaptureSession` (saves selected payloads; runs only when a capture session is attached) |
| 12 | `undecorate_saved_tensors` | `_undecorate_all_saved_tensors` | Strip TorchLens attrs from saved tensors |
| 13 | `clear_cuda_cache` | `torch.cuda.empty_cache` | Optional CUDA cache clear |
| 14 | `log_time_elapsed` | `_log_time_elapsed` | Capture timing |
| 15 | `finalize_param_logs` | `_finalize_param_logs` | Build and complete ParamLogs |
| 15.5 | `build_layer_logs` | `_build_layer_logs` (body in `_column_passes_finalize.build_layer_logs`) | Group ops into aggregate LayerLogs, each built on first lookup |
| 16 | `build_module_logs` | `_build_module_logs` | Build ModuleLogs |
| 16.5 | `hash_graph_shape` | `compute_graph_shape_hash` | Hash graph shape before pass-finished behavior changes |
| 17.5 | `adopt_containers_and_drop_workspaces` | executor `_run_adopt_containers_and_drop_workspaces` | Adopt container records; drop the three per-phase workspaces (runs before 17) |
| (none) | `seal_op_store` | package `_seal_op_store` | Compaction, primitive-row adoption, relation freeze, Op store freeze and payload owner (untimed) |
| 17 | `mark_tracing_finished` | `_set_tracing_finished` (body in `_finished_marks.mark_ops_finished`: one write per sealed column) | Switch Trace to user-facing behavior |
| 18 | `finalize_streamed_bundle` | `_finalize_streamed_bundle` | Finalize streamed out bundle |
| 19 | `evict_streamed_outs` | `_evict_streamed_outs` | Optional in-memory out eviction |
| 20 | `release_param_refs` | `_release_param_refs` | Drop live parameter references after finalization |

## Step 5: Conditional Attribution

Step 5 builds AST indexes, classifies terminal scalar bools, materializes dense
`conditional_records`, runs a backward flood from branch bools, attributes forward arm edges,
then derives legacy THEN/ELIF/ELSE views. Canonical structures are:
- `Trace.graph.conditional_records`
- `Trace.graph.conditional_arm_entry_edges`
- `Trace.graph.conditional_edge_call_indices`
- `conditional_arm_children` on `Op` and `Layer`

Arm kinds are `then`, `elif_N`, `else` and `fallthrough`. The AST indexer records a
`fallthrough` arm for the statements after an `if` chain with no `else` whose every arm ends
(`_ast_predicates.fallthrough_statements`; conservative: `return`, `raise`, `break`,
`continue`, or a nested `if` ending on both sides). It is walked as an arm, so `if`s in the
after-code nest under it. It has no legacy child view: read `conditional_arm_children`.

## equivalence_class module suffix

Module containment comes from op-creation stack snapshots, and op creation appends
the canonical module path to `equivalence_class`. No postprocess pass infers or
propagates `modules`.

## Loop Detection

Torch captures group by call site (`docs/guides/grouping.md`). At op creation
`_ops_activations._call_site_equivalence_class` keys each op by (innermost stateful module,
call-site chain from `utils/_call_site.py`, func_name, output slot) and writes
`equivalence_class = "<type>@cs<N>"`; the key itself is not persisted. Step 7
(`_step7_rows.group_call_sites`, over op columns) makes ops with one class passes of one layer, buffers group by buffer
address, inputs and outputs stay single, and ops with no chain fall back to their site key
position. A module call repeats the ops inside it only for a stateful module (one that
registers a non-None parameter or buffer, itself or below it; decided once per capture by
`backends/torch/_module_state.py`): a stateless module (`nn.ReLU`, `nn.Dropout`) is read as
the function it is, its identity a token in the chain, so it groups by where the stateful
code called it (`callsite_v2`). Module calls (`m:1`, `m:2`) are recorded either way. When no op carries a call-site class (other backends,
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
no-layers early exit, and the epilogue (the overlay fold of `_op_seal`, FuncCallLocation
adoption, kind-table freezes); the Op store seals inside the pipeline, in step `seal_op_store`,
whose body is the package's `_seal_op_store` (so tests patch its passes on the package);
the named steps run through
`_pipeline.run_pipeline` over `PIPELINE`, each through its `_executor` adapter. Every adapter calls its
function through the function's home module (`graph_traversal`, `labeling`,
`finalization`, ...), so a test that replaces a step replaces it on that home
module, never on the package (`tests/test_postprocess_executor.py` proves it). The
`finalize_streamed_bundle` gate IS the streaming snapshot point (context-writing, never
trace-writing); `evict_streamed_outs` gates on the snapshot; `should_run` evaluates exactly once per step.

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
