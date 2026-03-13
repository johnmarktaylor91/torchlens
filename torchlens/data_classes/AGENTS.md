# data_classes/ — Implementation Guide

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
