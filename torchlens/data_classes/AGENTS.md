# data_classes/ - Implementation Guide

## Key Access Patterns

```python
log["conv2d_1_5"]      # Layer aggregate
log["conv2d_1_5:2"]    # Op for a specific pass
log[3]                 # Op by ordinal
log.layers             # LayerAccessor
log.modules            # ModuleAccessor
log.params             # ParamAccessor
log.buffers            # BufferAccessor
```

Single-pass `Layer` values delegate per-pass attributes:

```python
layer = log.layers["linear_1_1"]
layer.out
layer.children       # union across ops
layer.ops             # dict[int, Op]
```

## Field Management
- Add fields to the class definition and the matching FIELD_ORDER tuple in `constants.py`.
- Add tests for user-facing fields and update `to_pandas()` when the field should export.
- Avoid ad hoc state that is not scrubbed by save/load, cleanup, and postprocess trimming.

## Trace Gotchas
- `_tracing_finished` changes `__len__`, `__getitem__`, iteration, and display behavior.
- Fast-pass postprocess relies on `_tracing_finished` staying true between ops.
- Methods such as `save`, `load`, `find_sites`, `fork`, `replay`, `rerun`, and
  `preview_fastlog` bridge into other subpackages; avoid importing them at module top if it
  creates cycles.
- `graph_shape_hash` is computed before `_set_tracing_finished`.

## Op Gotchas
- `copy()` shallow-copies selected graph/conditional fields and deep-copies the rest.
- `out` for some output/getitem cases may reference parent saved data directly.
- `grad` is a bare reference; do not mutate it in-place.
- `save_activation()` must route through `safe_copy()` and respect `backward_ready`.
- `TensorLog` is a compatibility alias from `op.py`; new docs should prefer
  `Op` unless referring to the alias itself.

## Layer Gotchas
- `__getattr__` delegation must raise `ValueError` for ambiguous multi-pass access.
- Aggregate graph properties are unions across ops.
- Conditional per-cond children need explicit merge handling; do not treat legacy THEN-only
  views as canonical.

## Module/Param/Buffer/Grad Logs
- `Module` and `ModuleCall` are built in postprocess Step 17 from `_module_build_data`.
- `Param` keeps `_param_ref` for lazy grad access; call `release_param_ref()` when
  breaking model references.
- Buffer graph nodes are plain `Op` records with `is_buffer=True`; `Buffer` is the
  persistent address-level entity exposed by `Trace.buffers` and owns versions.
- `GradFn` and `GradFnCall` are populated by backward capture and rendered separately.

## Cleanup
`cleanup.py` removes backrefs, parameter refs, saved outs, conditional edges, and
intervention metadata for removed layers. Keep it in sync with any new cross-reference field.

## Known Risks
- `to_pandas()` can lag new metadata fields; check tests before assuming export coverage.
- `FuncCallLocation` source properties are lazy; avoid keeping live frame/function objects.
- Removing or renaming labels requires updating conditional, intervention, module, and lookup
  references together.
