# data_classes/ - Logged Data Structures

## What This Is
All primary containers for logged TorchLens state. The hierarchy is:

```
Trace
  |- Layer          # aggregate per final layer
  |   +- Op  # one operation/pass tensor record
  |       +- Buffer # buffer source node specialization
  |- Module
  |   +- ModuleCall
  |- Param
  |- GradFn
  |   +- GradFnCall
  +- FuncCallLocation
```

Accessors (`LayerAccessor`, `ModuleAccessor`, `ParamAccessor`, `BufferAccessor`,
`GradFnAccessor`) provide dict-like lookup by label, index, or substring.

## Files

| File | Purpose |
|------|---------|
| `trace.py` | `Trace`, conditional event records, save/load/intervention/summary helpers |
| `op_log.py` | `Op`, `TensorLog` alias, tensor save and per-pass fields |
| `layer_log.py` | `Layer` aggregate, pass delegation, graph unions |
| `buffer_log.py` | `Buffer` and `BufferAccessor` |
| `module_log.py` | `ModuleCall`, `Module`, `ModuleAccessor` |
| `param_log.py` | `Param`, lazy grad access, `ParamAccessor` |
| `grad_fn_handle.py` | Backward graph `GradFn` and accessor |
| `grad_fn_call_log.py` | Per-pass backward graph record |
| `func_call_location.py` | Structured call stack frames and lazy source access |
| `interface.py` | Imported `Trace` access/query custom_methods |
| `_lookup_keys.py` | Lookup help and fuzzy key feedback |
| `_summary.py` | Small formatting helpers for summaries |
| `internal_types.py` | Internal dataclasses such as `FuncExecutionContext` |
| `cleanup.py` | Cycle breaking and field scrubbing after layer removal |

## Design Decisions

### Layer Delegation
Single-pass layers delegate unknown attrs to `ops[1]`. Multi-pass per-pass fields raise
`ValueError`, not `AttributeError`, to avoid Python falling through to `__getattr__`.

### Trace Surface
`Trace` owns more than storage: lookup, `draw`, `show_graph`, `save`,
`load`, `find_sites`, `resolve_sites`, `fork`, `rerun`, `replay`, `summary`,
`preview_fastlog`, and validation convenience custom_methods all live here or are attached via
helper modules.

### Conditional Metadata
Primary structures are dense-id based: `conditional_records`, `conditional_arm_entry_edges`,
`conditional_edge_call_indices`, and `conditional_arm_children`. Legacy THEN/ELIF/ELSE
fields are derived views for compatibility and rendering.

### Portable I/O
`Trace.save()` and `Trace.load()` delegate to `_io.bundle`. Loaded logs can contain
lazy out refs that materialize on access. `cleanup.py` must preserve manifest and
conditional consistency when removing entries.

### Layer Building
`_build_layer_logs()` merges multiple `Op` entries into one aggregate. Most fields
use first-pass values; only selected graph/role fields are merged across ops.

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

## Circular References

```
Trace -> Op -> source_trace -> Trace
Trace -> Module -> _source_trace -> Trace
Param -> _param_ref -> nn.Parameter
```

These rely on cyclic GC. Use `Trace.cleanup()` when retaining many logs or after
visualization-only workflows.
