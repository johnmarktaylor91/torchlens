# torchlens/data_classes architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

## What This Is

All primary containers for logged TorchLens state. The hierarchy is:

```
Trace
  |- Layer          # aggregate per final layer
  |   +- Op  # one operation/pass tensor record; buffer graph nodes are Op+is_buffer
  |- Module
  |   +- ModuleCall
  |- Param
  |- Buffer         # persistent address-level entity over buffer version nodes
  |- GradFn
  |   +- GradFnCall
  +- FuncCallLocation
```

Accessors (`LayerAccessor`, `ModuleAccessor`, `ParamAccessor`, `BufferAccessor`,
`GradFnAccessor`) provide dict-like lookup by label, index, or substring.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public exports for core data classes and accessors |
| `_accessor_base.py` | Shared ordered dict-like accessor base |
| `trace.py` | `Trace`, conditional event records, save/load/intervention/summary helpers |
| `_trace_accessors.py` | Trace-level typed accessor construction |
| `_trace_export.py` | Trace tabular export, decoded-output helpers, and the `to_agent_json()` agent dump entry point |
| `_trace_intervention.py` | Trace intervention surface; fork dispatch, replay, and rerun helpers |
| `_trace_fork.py` | M11 copy-on-write fork builder (COW shells over `OpStoreView`s) |
| `_compaction.py` | Freeze-seam Op metadata pooling (M11 fold) + M14 duplicate/empty container-cell pooling (`PooledCell`, hydrate-on-read) + singleton label-list compaction (bare str + identity-gated store registry, kind tables only) |
| `_layer_spec.py` | `_LAYER_MIRROR_SPEC` and the Layer mirror-field spec (split out of `layer.py`) |
| `_layer_accessors.py` | `OpAccessor`/`LayerAccessor` dict-like lookup (split out of `layer.py`; re-exported there) |
| `_schema_bindings.py` | GENERATED per-field `StorageBinding` axes — DO NOT EDIT; regenerate with `tools/generate_record_schema.py` |
| `_trace_components.py` | Declared `TRACE_FIELD_OWNERSHIP` component map — 329 entries, pinned equal to the `FIELD_POLICY` key set (the 229-name `MODEL_LOG_FIELD_ORDER` is a strict subset) |
| `_trace_stack.py` | Order-aligned activation stacking for completed traces |
| `_trace_rehydrate.py` | Load-side Trace rehydration |
| `_backend_capability_guards.py` | Backend capability guard helpers |
| `_nonfinite.py` | Nonfinite scan/abort helpers |
| `prehook.py` | Pre-hook effect records |
| `_trace_profile.py` | Trace profiling and timing helpers |
| `_trace_stats.py` | Trace aggregate stats and backward-pass projections |
| `_trace_validation.py` | Trace validation and log-entry removal helpers |
| `_trace_viz.py` | Trace visualization entrypoints |
| `op.py` | `Op` two-word row facade (`_core`/`_row` over `_trace_core`), `TensorLog` alias, tensor save, per-pass fields |
| `_op_transforms.py` | User-transform apply + train-mode/streaming output validation helpers (split from `op.py`; `op.py` re-exposes them) |
| `layer.py` | `Layer` aggregate, pass delegation, graph unions |
| `buffer.py` | `Buffer` and `BufferAccessor` |
| `module.py` | `ModuleCall`, `Module`, `ModuleAccessor` |
| `_call_tree.py` | ModuleCall ASCII call-tree printer + call-scope op resolution/edge-count helpers (split from `module.py`) |
| `param.py` | `Param`, lazy grad access, `ParamAccessor` |
| `backward_pass.py` | Per-invocation backward-pass records and accessor |
| `grad_fn.py` | Backward graph `GradFn` and accessor |
| `grad_fn_call.py` | Per-pass backward graph record |
| `container.py` | Structured output/container specs and reconstruction helpers |
| `derived_grad.py` | Derived-gradient payload records |
| `field_policy.py` | Structural field policy table helpers |
| `func_call_location.py` | Structured call stack frames and lazy source access |
| `interface.py` | Imported `Trace` access/query custom_methods |
| `_lookup_keys.py` | Lookup help and fuzzy key feedback |
| `_module_role_hints.py` | Module input/output role hint helpers |
| `_repr.py` | Shared formatting helpers for user-facing reprs |
| `_runtime_handles.py` | Runtime object handle resolution helpers |
| `_state_adapter.py` | Class-agnostic live-state enumeration/restore adapters (`state_items`/`state_new`/`state_restore`; the flat build-state they once adapted dissolved in M10) |
| `_summary.py` | Small formatting helpers for summaries |
| `internal_types.py` | Internal dataclasses such as `FuncExecutionContext` |
| `cleanup.py` | Cycle breaking and field scrubbing after layer removal |

## Design Decisions

### Layer Delegation
Single-pass layers delegate unknown attrs to `ops[0]`. Multi-pass per-pass fields raise
`ValueError`, not `AttributeError`, to avoid Python falling through to `__getattr__`.

### Trace Surface
`Trace` owns more than storage: lookup, `draw`, `save`,
`find_sites`, `resolve_sites`, `fork`, `run`, `push`, `summary`,
(loading is module-level `tl.load`, never `trace.load`; there is no
`Trace.show_graph` — use `draw` or `torchlens.visualization.show_model_graph`;
`rerun`/`replay` are deprecated aliases of `run`/`push` that warn),
`preview_fastlog`, and validation convenience custom_methods all live here or are attached via
helper modules.

### Conditional Metadata
Primary structures are dense-id based: `conditional_records`, `conditional_arm_entry_edges`,
`conditional_edge_call_indices`, and `conditional_arm_children`. Legacy THEN/ELIF/ELSE
fields are derived views for compatibility and rendering.

### Portable I/O
`Trace.save()` and module-level `tl.load()` delegate to `_io.bundle`. Loaded logs can contain
lazy out refs that materialize on access. `cleanup.py` must preserve manifest and
conditional consistency when removing entries.

### Layer Building
`_build_layer_logs()` merges multiple `Op` entries into one aggregate. Most fields
use first-pass values; only selected graph/role fields are merged across ops.
Since M8, `Layer` no longer COPIES the first-pass fields: they are mirror
descriptors reading through to `ops[0]` on demand, with per-layer `__dict__`
shadows for merged/overwritten values (`_LAYER_MIRROR_SPEC` in `_layer_spec.py`;
`layer.py` imports it).
`in_conditionals`/`terminal_bool_for` remain build-time snapshots because
`_build_conditional_records` rebinds them on the OPS after layers are built.

### M8 record facades (Param/Buffer/FuncCallLocation/ModuleCall/Module)
These classes are row facades over per-trace kind tables
(`_trace_core/record_rows.py`): declared stored fields are row-cell
descriptors; the instance `__dict__` keeps only the store binding, user
extras (FORK-7), and the few names whose properties hardcode `__dict__`
access (template/source-trace/facets slots). Torch build passes adopt records
into `TraceCore.kind_rows`; direct construction, preview backends, pickle
restore, and fork shells stay detached single-row stores.

### M11 COW fork
`Trace.fork()` builds copy-on-write forks (`_trace_fork.build_fork`): fork
`Op`s and record facades are fresh two-word shells bound to per-fork
`OpStoreView`s at the SAME rows; only `Layer` shadow dicts, record instance
extras, and the policy-driven trace-side field remainder are copied (one
shared-memo deepcopy over small trace-side data). Views isolate every
mutation surface: fork writes/deletes land in the view overlay, mutable
builtin containers are eagerly copied into the fork overlay at fork time
(`OpStoreView.isolate_mutable_cells`, a sparse sweep over the base store's
cached mutable-cell index; tensors/callables inside stay shared
by identity), `GroupRef` cells translate to per-fork cloned group tables,
and cell-held records/accessors translate to fork facades. Traces without
a sealed core-backed op store (loaded analysis traces, failed partials)
take the detached fallback (per-record single-row duplication). Mutation
isolation holds in BOTH directions at fork time (deepcopy snapshot
semantics); the one shared residual is mutables nested inside non-builtin
custom objects.

### Module / ModuleCall Fields
`Module.training` mirrors `nn.Module.training`; `Module.layer_labels` stores Layer
labels, while `Module.layers` resolves those labels to Layer records. Module input/output
collections (`input_ops`, `input_layers`, `output_ops`, `output_layers`) are bare label lists.
`ModuleCall.ops`, `input_ops`, `input_layers`, `output_ops`, and `output_layers` are also bare
label lists; resolve through the owning Trace accessor when records are needed.

## Autograd Contract

`Op.save_activation()` is the slow/replay choke point for saved tensor copies.
`backward_ready=True` keeps saved floating tensors graph-connected, rejects contradictory
detaching/disk-only configs, and must restore all flags in `finally` paths.

Transform boundary ops carry `is_transform`, `transform_kind`, `transform_chain`,
`transform_config`, lazy `transform_fn_source`, and diagnostic
`unattributed_tensor_args`. Synthetic output ops should clear transform/provenance
role fields so `Trace.transforms` only reports real transform boundary nodes.

## Back-References

```
Trace -> Op -> source_trace -> Trace          (weakref: Op._source_trace_ref)
Trace -> Module -> _source_trace -> Trace     (weakref: Module._source_trace_ref)
Param -> _param_ref -> nn.Parameter
```

(The two rows above are representative, not exhaustive: `Op`, `Module`,
`Param`, `Buffer`, `GradFn`, and their call-record kinds all carry a
`_source_trace_ref`/`_source_ref` weak back-reference of the same shape.)

The Trace back-references are stored as `weakref.ref` in `_source_trace_ref` slots
(`FieldPolicy.WEAKREF_STRIP`), so these back-references themselves do NOT form strong
cycles; reading one after the Trace dies yields `None` (and consumers that need it, such
as `ModuleCall.module`, raise). Other strong reference cycles remain (core/facade
reference tables and similar internal structure), so a dropped Trace is reclaimed by the
CYCLIC collector, not by refcounting alone: measured on a live capture, `del trace` with
`gc` disabled leaves the object alive until `gc.collect()` runs, with or without a prior
`Trace.cleanup()`. Still call `Trace.cleanup()` when retaining many logs or after
visualization-only workflows.

Retained-Op payload lifetime (fix/fork F4): an `Op` kept past its Trace's death no longer
pins every captured activation. Each owning `TraceCore` (the capture's core plus one per
fork core sharing the sealed base) registers a `weakref.finalize` on the op store; when
the LAST owner is garbage-collected the store evicts top-level tensor cells (and the
snapshot surfaces of surviving fork `OpStoreView`s), so payload reads on the retained
facade return the payload-absent spelling while metadata stays readable. Keep the Trace
alive (or clone the tensor) to keep payloads. Fork views hold the fork's record
translator weakly (anchored on the fork core) so a retained fork Op cannot root the
whole fork graph.

## Key Access Patterns

```python
log["conv2d_1_5"]  # Layer aggregate
log["conv2d_1_5:2"]  # Op for a specific pass
log[3]  # Op by ordinal
log.layers  # LayerAccessor
log.modules  # ModuleAccessor
log.params  # ParamAccessor
log.buffers  # BufferAccessor
```

Single-pass `Layer` values delegate per-pass attributes:

```python
layer = log.layers["linear_1_1"]
layer.out
layer.children  # union across ops
layer.ops  # dict[int, Op]
```

## Field Management

- Add fields to the class definition and the matching FIELD_ORDER list in the
  parent package's `torchlens/constants.py` (not in this directory).
- Add tests for user-facing fields and update `to_pandas()` when the field should export.
- Avoid ad hoc state that is not scrubbed by save/load, cleanup, and postprocess trimming.

## Trace Gotchas

- `_tracing_finished` changes `__len__`, `__getitem__`, iteration, and display behavior.
- Fast-pass postprocess relies on `_tracing_finished` staying true between ops.
- Methods such as `save`, `find_sites`, `fork`, `run`, `push`, and
  `preview_fastlog` bridge into other subpackages; avoid importing them at module top if it
  creates cycles. There is NO `Trace.load` — loading is module-level `tl.load`
  (see the sibling `AGENTS.md`) — and `replay`/`rerun` are deprecated warning
  aliases of `push`/`run`.
- `run(inputs=...)` returns a transactional `RunResult` for live and loaded sparse providers;
  legacy `run(model, x, ...)` remains the intervention-rerun compatibility path.
- `graph_shape_hash` is computed before `_set_tracing_finished`.

## Fork (M11)

- `Trace.fork()` is COPY-ON-WRITE (`_trace_fork.build_fork`): fork records are
  fresh two-word shells over per-fork `OpStoreView`s at the same rows; the
  object-graph forkcopier (typed deepcopy engine) is deleted. Fork writes land
  in view overlays; mutable builtin containers are isolated eagerly at fork time; GroupRefs
  translate to cloned group tables. Modules fork as detached duplicates (their
  cells embed trace-strong accessors); coreless traces take the detached
  fallback. Fork->parent isolation is pinned IN BOTH DIRECTIONS: mutable
  builtin containers are eagerly copied into the fork overlay AT FORK TIME
  (`OpStoreView.isolate_mutable_cells`), so parent in-place container
  mutation after the fork is NEVER visible to the fork — the historical
  before-first-read visibility window is closed; do not weaken the eager
  sweep to restore it.

## Op Gotchas

- `Op.__slots__` is `("_core", "_row")` (M5 seam): fields are generated data
  descriptors over the per-trace `_trace_core` row store (detached single-row store
  for copy/pickle/fork/preview ops). `_OP_SLOT_NAMES` remains the declared stored-field
  universe; `_slot()`, `_internal_set`, and `object.__setattr__` compose over the
  descriptors exactly as they did over slots. Never assume per-instance storage.
- `copy()` deep-copies graph/conditional metadata and SHARES (shallow) the
  tensor-payload/callable set — `fields_not_to_deepcopy` in `op.py` is
  `out`/`transformed_out`/`saved_args`/`saved_kwargs`/`func`/templates/
  `parent_params`/... — i.e. payloads alias the source op; graph fields do not.
- `out` for some output/getitem cases may reference parent saved data directly.
- `grad` is a bare reference; do not mutate it in-place.
- `var_names` records bare source assignment target names for an op when
  `save_code_context=True`; inline, unnameable, dynamic-source, attribute, and
  subscript targets are represented as `[]`.
- `save_activation()` must route through `safe_copy()` and respect `backward_ready`.
- `TensorLog` is a compatibility alias from `op.py`; new docs should prefer
  `Op` unless referring to the alias itself.

## Layer Gotchas

- `__getattr__` delegation must raise `ValueError` for ambiguous multi-pass access.
- Aggregate graph properties are unions across ops.
- Conditional per-cond children need explicit merge handling; do not treat legacy THEN-only
  views as canonical.

## Module/Param/Buffer/Grad Logs

- `Module` and `ModuleCall` are built in postprocess Step 16 from `_module_build_data`.
- `Param` keeps `_param_ref` for lazy grad access; call `release_param_ref()` when
  breaking model references.
- Buffer graph nodes are plain `Op` records with `is_buffer=True`; `Buffer` is the
  persistent address-level entity exposed by `Trace.buffers` and owns versions.
- `GradFn` and `GradFnCall` are populated by backward capture and rendered separately.
- Provisional `.handle` accessors on `Param`, `Buffer`, `Module`, and `GradFn` return
  the live torch/autograd object or `None`; they are computed, non-portable, and not
  dataframe fields. `Param.handle` is the non-caching counterpart to `Param.value`.

## Cleanup

`cleanup.py` removes backrefs, parameter refs, saved outs, conditional edges, and
intervention metadata for removed layers. Keep it in sync with any new cross-reference field.

## Known Risks

- `to_pandas()` can lag new metadata fields; check tests before assuming export coverage.
- `FuncCallLocation` source properties are lazy; avoid keeping live frame/function objects.
- Transform boundary ops use `is_transform`, `transform_kind`, `transform_chain`,
  `transform_config`, and lazy `transform_fn_source`; keep these fields in sync with
  `constants.py`, portable save/load, pandas exclusions, and output-node cleanup.
- `unattributed_tensor_args` is diagnostic provenance metadata. Synthetic output nodes
  must not inherit it or transform role fields from their parent op.
- Removing or renaming labels requires updating conditional, intervention, module, and lookup
  references together.
