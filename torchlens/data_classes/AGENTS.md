# data_classes/ - Implementation Guide

## Key Access Patterns

```python
log["conv2d_1_5"]      # LayerLog aggregate
log["conv2d_1_5:2"]    # LayerPassLog for a specific pass
log[3]                 # LayerPassLog by ordinal
log.layers             # LayerAccessor
log.modules            # ModuleAccessor
log.params             # ParamAccessor
log.buffers            # BufferAccessor
```

Single-pass `LayerLog` values delegate per-pass attributes:

```python
layer = log.layers["linear_1_1"]
layer.activation
layer.child_layers       # union across passes
layer.passes             # dict[int, LayerPassLog]
```

## Field Management
- Add fields to the class definition and the matching FIELD_ORDER tuple in `constants.py`.
- Add tests for user-facing fields and update `to_pandas()` when the field should export.
- Avoid ad hoc state that is not scrubbed by save/load, cleanup, and postprocess trimming.

## ModelLog Gotchas
- `_pass_finished` changes `__len__`, `__getitem__`, iteration, and display behavior.
- Fast-pass postprocess relies on `_pass_finished` staying true between passes.
- Methods such as `save`, `load`, `find_sites`, `fork`, `replay`, `rerun`, and
  `preview_fastlog` bridge into other subpackages; avoid importing them at module top if it
  creates cycles.
- `graph_shape_hash` is computed before `_set_pass_finished`.

## LayerPassLog Gotchas
- `copy()` shallow-copies selected graph/conditional fields and deep-copies the rest.
- `activation` for some output/getitem cases may reference parent saved data directly.
- `gradient` is a bare reference; do not mutate it in-place.
- `save_tensor_data()` must route through `safe_copy()` and respect `train_mode`.
- `TensorLog` is a compatibility alias from `layer_pass_log.py`; new docs should prefer
  `LayerPassLog` unless referring to the alias itself.

## LayerLog Gotchas
- `__getattr__` delegation must raise `ValueError` for ambiguous multi-pass access.
- Aggregate graph properties are unions across passes.
- Conditional per-cond children need explicit merge handling; do not treat legacy THEN-only
  views as canonical.

## Module/Param/Buffer/Grad Logs
- `ModuleLog` and `ModulePassLog` are built in postprocess Step 17 from `_module_build_data`.
- `ParamLog` keeps `_param_ref` for lazy gradient access; call `release_param_ref()` when
  breaking model references.
- `BufferLog` extends `LayerPassLog` but owns buffer-specific `name` and `module_address`.
- `GradFnLog` and `GradFnPassLog` are populated by backward capture and rendered separately.

## Cleanup
`cleanup.py` removes backrefs, parameter refs, saved activations, conditional edges, and
intervention metadata for removed layers. Keep it in sync with any new cross-reference field.

## Known Risks
- `to_pandas()` can lag new metadata fields; check tests before assuming export coverage.
- `FuncCallLocation` source properties are lazy; avoid keeping live frame/function objects.
- Removing or renaming labels requires updating conditional, intervention, module, and lookup
  references together.
