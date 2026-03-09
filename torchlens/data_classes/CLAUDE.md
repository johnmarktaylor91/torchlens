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

## Key Access Patterns

```python
# ModelLog access
log["conv2d_1_5"]      # -> LayerLog (aggregate)
log["conv2d_1_5:2"]    # -> LayerPassLog (specific pass)
log[3]                 # -> LayerPassLog (by ordinal)
log.layers             # -> LayerAccessor (all LayerLogs)
log.modules            # -> ModuleAccessor
log.params             # -> ParamAccessor
log.buffers            # -> BufferAccessor

# LayerLog delegation
layer = log.layers["conv2d_1_1"]
layer.activation  # -> delegates to passes[1] for single-pass layers
layer.child_layers     # -> union of no-pass labels across all passes
layer.passes           # -> Dict[int, LayerPassLog]
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

### _build_layer_logs Multi-Pass Merge
Only 3 fields merged across passes (has_input_ancestor OR, io_role char-merge,
is_leaf_module_output OR). All other 78 fields use first-pass values.
`cond_branch_start_children` and `cond_branch_then_children` use first pass only.

## Circular References (GC concern)
```
ModelLog -> LayerPassLog -> source_model_log -> ModelLog  (CYCLE)
ModelLog -> ModuleLog -> _source_model_log -> ModelLog    (CYCLE)
ParamLog -> _param_ref -> nn.Parameter                   (PINS MODEL)
```
All rely on Python's cyclic GC rather than ref-counting. `cleanup()` in cleanup.py
can be called explicitly to break cycles.

## Known Bugs
- **TO-PANDAS-NEW-FIELDS**: `to_pandas()` missing `func_config` and `cond_branch_then_children` columns
- **COND-THEN-MULTIPASS**: `cond_branch_then_children` not merged for multi-pass LayerLog (first pass only)

## Gotchas
- Adding new fields: update class definition AND `constants.py` FIELD_ORDER
- `copy()` on LayerPassLog: shallow-copies 8 specific fields, deep-copies rest.
  Safe only because downstream code uses assignment, not mutation.
- `activation` for non-getitem output layers is a DIRECT REFERENCE to parent's
  saved data — mutation affects both
- `equivalent_operations` per-LayerPassLog holds direct reference to ModelLog-level
  sets; becomes stale after rename step 11 (cosmetic, not read downstream)
- `gradient` is a bare reference (no clone) — shared with parent tensor
- `FuncCallLocation._frame_func_obj` set at construction but only released in
  `_load_source()` (lazy property trigger) — leaks if properties never accessed

## Related
- [capture/](../capture/CLAUDE.md) — Creates LayerPassLog entries during logging
- [postprocess/](../postprocess/CLAUDE.md) — Populates final fields, builds LayerLog/ModuleLog/ParamLog
- `constants.py` — FIELD_ORDER definitions must stay in sync with classes
