# data_classes/ — Data Structures

## What This Is
All data structures for storing logged forward-pass information. Organized as a hierarchy:

```
ModelLog (top-level container)
  |- LayerLog (aggregate per-layer, groups passes)
  |   +- LayerPassLog (per-pass tensor operation entry, ~85+ fields)
  |       +- BufferLog (extends LayerPassLog for buffer tensors)
  |- ModuleLog (per module in model)
  |   +- ModulePassLog (per invocation of a module)
  |- ParamLog (per model parameter)
  +- FuncCallLocation (structured call stack frame)
```

Each main class has a companion Accessor (LayerAccessor, ModuleAccessor, ParamAccessor,
BufferAccessor) providing dict-like indexing by name, index, or substring.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `model_log.py` | 497 | ModelLog: top-level container, 70+ attrs, FLOPs properties, `conditional_then_edges` |
| `layer_pass_log.py` | 677 | LayerPassLog: per-pass entry (~85+ fields, 18 @properties), `cond_branch_then_children`, `func_config` |
| `layer_log.py` | 671 | LayerLog: aggregate class, 13 direct + 38 @properties, `__getattr__` delegation |
| `buffer_log.py` | 132 | BufferLog(LayerPassLog): `name`/`module_address` computed properties, BufferAccessor |
| `module_log.py` | 525 | ModuleLog, ModulePassLog, ModuleAccessor, shared alias support, FLOPs aggregation |
| `param_log.py` | 248 | ParamLog (lazy grad via `_param_ref`), `release_param_ref()` for GC, ParamAccessor |
| `func_call_location.py` | 265 | FuncCallLocation: lazy properties via linecache, dual construction paths, `_SENTINEL` |
| `internal_types.py` | 61 | FuncExecutionContext + VisualizationOverrides (`@dataclass(slots=True)`) |
| `interface.py` | 508 | ModelLog query methods: `__getitem__`, `__str__`, `to_pandas()`, 7-step lookup cascade |
| `cleanup.py` | 237 | Post-session teardown: O(N+M) batch removal, `conditional_then_edges` filtering, `release_param_refs` |

## Design Decisions

### LayerLog Delegation
- Single-pass layers: `__getattr__` delegates unknown attrs to `passes[1]`
- Multi-pass per-pass fields: raises **ValueError** (not AttributeError) to avoid
  Python's property/__getattr__ trap
- Aggregate graph properties (child_layers, parent_layers) return union across passes

### ModelLog Method Importation
Methods defined in `interface.py` and assigned as class attributes on ModelLog.
Keeps `model_log.py` small while class has 20+ methods.

### `_pass_finished` Behavioral Switch
`__len__`, `__getitem__`, `__iter__`, `__str__` change behavior before vs after
postprocessing. Pre-pass uses `_raw_tensor_dict`; post-pass uses `layer_dict_all_keys`.
**Not reset during fast pass** — intentional, fast-path postprocessing depends on True.

### BufferLog
Subclass of LayerPassLog. `name` and `module_address` live only on BufferLog, not on
LayerLog (too generic for the aggregate). Single-pass buffer LayerLogs access them
via `__getattr__` delegation.

### _build_layer_logs Multi-Pass Merge
Only 3 fields merged across passes (has_input_ancestor OR, io_role char-merge,
is_leaf_module_output OR). All other 78 fields use first-pass values.

### Saved Activation Autograd Contract
`LayerPassLog.save_tensor_data()` is the slow/replay chokepoint for saved activation
copies. Legacy `detach_saved_tensors=False` keeps the saved tensor copy attached to
autograd, and existing explicit uses continue to work. `train_mode=True` is the
preferred training API because it also rejects contradictory detaching and disk saves,
preserves frozen parameter settings, and temporarily forces replay `detach_saved_tensor`
flags to the graph-connected setting. `save_new_activations(train_mode=True)` must
restore both `ModelLog.detach_saved_tensors` and every per-layer `detach_saved_tensor`
flag in a `finally` path, including graph-mismatch failures.

## Circular References (GC concern)
```
ModelLog -> LayerPassLog -> source_model_log -> ModelLog  (CYCLE)
ModelLog -> ModuleLog -> _source_model_log -> ModelLog    (CYCLE)
ParamLog -> _param_ref -> nn.Parameter                   (PINS MODEL)
```
All rely on Python's cyclic GC. `cleanup()` in cleanup.py can be called explicitly.
