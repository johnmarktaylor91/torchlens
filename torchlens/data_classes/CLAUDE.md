# data_classes/ — Data Structures

## What This Is
All data structures for storing logged forward-pass information. Organized as a hierarchy:

```
ModelLog (top-level container)
  ├─ LayerLog (aggregate per-layer, groups passes)
  │   └─ LayerPassLog (per-pass tensor operation entry, ~80+ fields)
  │       └─ BufferLog (extends LayerPassLog for buffer tensors)
  ├─ ModuleLog (per module in model)
  │   └─ ModulePassLog (per invocation of a module)
  ├─ ParamLog (per model parameter)
  └─ FuncCallLocation (structured call stack frame)
```

Each main class has a companion Accessor (LayerAccessor, ModuleAccessor, ParamAccessor,
BufferAccessor) providing dict-like indexing by name, index, or substring.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `model_log.py` | 316 | ModelLog: top-level container, FLOPs properties, `_init_module_build_data()` |
| `layer_pass_log.py` | 477 | LayerPassLog: per-pass entry (~80+ fields), `save_tensor_data()`, `copy()` |
| `layer_log.py` | 553 | LayerLog: aggregate class, `__getattr__` delegation, LayerAccessor |
| `buffer_log.py` | 108 | BufferLog(LayerPassLog): `name`/`module_address` computed properties, BufferAccessor |
| `module_log.py` | 353 | ModuleLog, ModulePassLog, ModuleAccessor |
| `param_log.py` | 196 | ParamLog (lazy grad via `_param_ref`), ParamAccessor |
| `func_call_location.py` | 230 | FuncCallLocation: lazy source loading via linecache |
| `internal_types.py` | 41 | FuncExecutionContext + VisualizationOverrides dataclasses |
| `interface.py` | 430 | ModelLog query methods: `__getitem__`, `__str__`, `to_pandas()` |
| `cleanup.py` | 157 | Post-session teardown: destroy entries, free GPU memory |

## Key Access Patterns

```python
# ModelLog access
log["conv2d_1_5"]      # → LayerLog (aggregate)
log["conv2d_1_5:2"]    # → LayerPassLog (specific pass)
log[3]                 # → LayerPassLog (by ordinal)
log.layers             # → LayerAccessor (all LayerLogs)
log.modules            # → ModuleAccessor
log.params             # → ParamAccessor
log.buffers            # → BufferAccessor

# LayerLog delegation
layer = log.layers["conv2d_1_1"]
layer.tensor_contents  # → delegates to passes[1] for single-pass layers
layer.child_layers     # → union of no-pass labels across all passes
layer.passes           # → Dict[int, LayerPassLog]
```

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

## Circular References (GC concern)
```
ModelLog → LayerPassLog → source_model_log → ModelLog  (CYCLE)
ModelLog → ModuleLog → _source_model_log → ModelLog    (CYCLE)
ParamLog → _param_ref → nn.Parameter                   (PINS MODEL)
```
All rely on Python's cyclic GC rather than ref-counting. `cleanup()` in cleanup.py
can be called explicitly to break cycles.

## Gotchas
- Adding new fields: update class definition AND `constants.py` FIELD_ORDER
- `copy()` on LayerPassLog: shallow-copies 8 specific fields, deep-copies rest.
  Safe only because downstream code uses assignment, not mutation.
- `tensor_contents` for non-getitem output layers is a DIRECT REFERENCE to parent's
  saved data — mutation affects both
- `equivalent_operations` per-LayerPassLog holds direct reference to ModelLog-level
  sets; becomes stale after rename step 11 (cosmetic, not read downstream)
- `grad_contents` is a bare reference (no clone) — shared with parent tensor

## Related
- [capture/](../capture/CLAUDE.md) — Creates LayerPassLog entries during logging
- [postprocess/](../postprocess/CLAUDE.md) — Populates final fields, builds LayerLog/ModuleLog/ParamLog
- `constants.py` — FIELD_ORDER definitions must stay in sync with classes
