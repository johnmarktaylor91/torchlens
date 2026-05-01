# data_classes/ - Logged Data Structures

## What This Is
All primary containers for logged TorchLens state. The hierarchy is:

```
ModelLog
  |- LayerLog          # aggregate per final layer
  |   +- LayerPassLog  # one operation/pass tensor record
  |       +- BufferLog # buffer source node specialization
  |- ModuleLog
  |   +- ModulePassLog
  |- ParamLog
  |- GradFnLog
  |   +- GradFnPassLog
  +- FuncCallLocation
```

Accessors (`LayerAccessor`, `ModuleAccessor`, `ParamAccessor`, `BufferAccessor`,
`GradFnAccessor`) provide dict-like lookup by label, index, or substring.

## Files

| File | Purpose |
|------|---------|
| `model_log.py` | `ModelLog`, conditional event records, save/load/intervention/summary helpers |
| `layer_pass_log.py` | `LayerPassLog`, `TensorLog` alias, tensor save and per-pass fields |
| `layer_log.py` | `LayerLog` aggregate, pass delegation, graph unions |
| `buffer_log.py` | `BufferLog` and `BufferAccessor` |
| `module_log.py` | `ModulePassLog`, `ModuleLog`, `ModuleAccessor` |
| `param_log.py` | `ParamLog`, lazy grad access, `ParamAccessor` |
| `grad_fn_log.py` | Backward graph `GradFnLog` and accessor |
| `grad_fn_pass_log.py` | Per-pass backward graph record |
| `func_call_location.py` | Structured call stack frames and lazy source access |
| `interface.py` | Imported `ModelLog` access/query methods |
| `_lookup_keys.py` | Lookup help and fuzzy key feedback |
| `_summary.py` | Small formatting helpers for summaries |
| `internal_types.py` | Internal dataclasses such as `FuncExecutionContext` |
| `cleanup.py` | Cycle breaking and field scrubbing after layer removal |

## Design Decisions

### LayerLog Delegation
Single-pass layers delegate unknown attrs to `passes[1]`. Multi-pass per-pass fields raise
`ValueError`, not `AttributeError`, to avoid Python falling through to `__getattr__`.

### ModelLog Surface
`ModelLog` owns more than storage: lookup, `render_graph`, `show_graph`, `save`,
`load`, `find_sites`, `resolve_sites`, `fork`, `rerun`, `replay`, `summary`,
`preview_fastlog`, and validation convenience methods all live here or are attached via
helper modules.

### Conditional Metadata
Primary structures are dense-id based: `conditional_events`, `conditional_arm_edges`,
`conditional_edge_passes`, and `cond_branch_children_by_cond`. Legacy THEN/ELIF/ELSE
fields are derived views for compatibility and rendering.

### Portable I/O
`ModelLog.save()` and `ModelLog.load()` delegate to `_io.bundle`. Loaded logs can contain
lazy activation refs that materialize on access. `cleanup.py` must preserve manifest and
conditional consistency when removing entries.

### Layer Building
`_build_layer_logs()` merges multiple `LayerPassLog` entries into one aggregate. Most fields
use first-pass values; only selected graph/role fields are merged across passes.

## Autograd Contract
`LayerPassLog.save_tensor_data()` is the slow/replay choke point for saved tensor copies.
`train_mode=True` keeps saved floating tensors graph-connected, rejects contradictory
detaching/disk-only configs, and must restore all flags in `finally` paths.

## Circular References

```
ModelLog -> LayerPassLog -> source_model_log -> ModelLog
ModelLog -> ModuleLog -> _source_model_log -> ModelLog
ParamLog -> _param_ref -> nn.Parameter
```

These rely on cyclic GC. Use `ModelLog.cleanup()` when retaining many logs or after
visualization-only workflows.
