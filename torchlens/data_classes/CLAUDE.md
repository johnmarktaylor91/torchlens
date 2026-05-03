# data_classes/ - Logged Data Structures

## What This Is
All primary containers for logged TorchLens state. The hierarchy is:

```
Trace
  |- LayerLog          # aggregate per final layer
  |   +- OpLog  # one operation/pass tensor record
  |       +- BufferLog # buffer source node specialization
  |- ModuleLog
  |   +- ModuleCallLog
  |- ParamLog
  |- GradFnLog
  |   +- GradFnCallLog
  +- FuncCallLocation
```

Accessors (`LayerAccessor`, `ModuleAccessor`, `ParamAccessor`, `BufferAccessor`,
`GradFnAccessor`) provide dict-like lookup by label, index, or substring.

## Files

| File | Purpose |
|------|---------|
| `trace.py` | `Trace`, conditional event records, save/load/intervention/summary helpers |
| `op_log.py` | `OpLog`, `TensorLog` alias, tensor save and per-pass fields |
| `layer_log.py` | `LayerLog` aggregate, pass delegation, graph unions |
| `buffer_log.py` | `BufferLog` and `BufferAccessor` |
| `module_log.py` | `ModuleCallLog`, `ModuleLog`, `ModuleAccessor` |
| `param_log.py` | `ParamLog`, lazy grad access, `ParamAccessor` |
| `grad_fn_log.py` | Backward graph `GradFnLog` and accessor |
| `grad_fn_call_log.py` | Per-pass backward graph record |
| `func_call_location.py` | Structured call stack frames and lazy source access |
| `interface.py` | Imported `Trace` access/query custom_methods |
| `_lookup_keys.py` | Lookup help and fuzzy key feedback |
| `_summary.py` | Small formatting helpers for summaries |
| `internal_types.py` | Internal dataclasses such as `FuncExecutionContext` |
| `cleanup.py` | Cycle breaking and field scrubbing after layer removal |

## Design Decisions

### LayerLog Delegation
Single-pass layers delegate unknown attrs to `ops[1]`. Multi-pass per-pass fields raise
`ValueError`, not `AttributeError`, to avoid Python falling through to `__getattr__`.

### Trace Surface
`Trace` owns more than storage: lookup, `render_graph`, `show_graph`, `save`,
`load`, `find_sites`, `resolve_sites`, `fork`, `rerun`, `replay`, `summary`,
`preview_fastlog`, and validation convenience custom_methods all live here or are attached via
helper modules.

### Conditional Metadata
Primary structures are dense-id based: `conditional_events`, `conditional_arm_edges`,
`conditional_edge_ops`, and `cond_branch_children_by_cond`. Legacy THEN/ELIF/ELSE
fields are derived views for compatibility and rendering.

### Portable I/O
`Trace.save()` and `Trace.load()` delegate to `_io.bundle`. Loaded logs can contain
lazy out refs that materialize on access. `cleanup.py` must preserve manifest and
conditional consistency when removing entries.

### Layer Building
`_build_layer_logs()` merges multiple `OpLog` entries into one aggregate. Most fields
use first-pass values; only selected graph/role fields are merged across ops.

## Autograd Contract
`OpLog.save_activation()` is the slow/replay choke point for saved tensor copies.
`train_mode=True` keeps saved floating tensors graph-connected, rejects contradictory
detaching/disk-only configs, and must restore all flags in `finally` paths.

## Circular References

```
Trace -> OpLog -> source_trace -> Trace
Trace -> ModuleLog -> _source_trace -> Trace
ParamLog -> _param_ref -> nn.Parameter
```

These rely on cyclic GC. Use `Trace.cleanup()` when retaining many logs or after
visualization-only workflows.
