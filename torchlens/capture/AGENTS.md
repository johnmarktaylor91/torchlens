# capture/ — Implementation Guide

## Label Formats
- Source tensors: `{type}_{num}_raw` (e.g., `"input_0_raw"`, `"buffer_1_raw"`)
- Function outputs: `{type}_{num}_{counter}_raw` (e.g., `"conv2d_1_5_raw"`)
- Labels are raw during capture; renamed to final labels in postprocess/labeling.py

## arg_positions.py
- 3-tier O(1) lookup: static `FUNC_ARG_SPECS` table → dynamic `_DYNAMIC_SPEC_CACHE` → BFS fallback
- `ArgSpec` frozen dataclass: `tensor_args`, `tensor_kwargs`, `param_args`, `param_kwargs`
- `extract_tensors_and_params()` — Main entry point, returns (tensors, params) from args/kwargs

## salient_args.py
- `@_register()` pattern for extractors per layer type
- `_build_arg_name_map()` maps positional args to named params
- Failure-safe: try-except returns `{}` on any error
- `_get()` helper returns `None` on missing keys (graceful degradation)

## flops.py
- 3-tier system: ZERO_FLOPS_OPS (view, reshape = 0), ELEMENTWISE_FLOPS (relu, sigmoid),
  SPECIALTY_HANDLERS (conv2d, matmul — shape-aware computation)
- MAC = 2 FLOPs convention

## Known Bugs
- **ARG-KWARGS-MISSING**: `extract_tensors_and_params()` doesn't extract tensors passed as
  keyword args for many common functions (linear, cat, where). `tensor_kwargs=()` in static
  entries means `linear(x, weight=w, bias=b)` only finds `x`.
- **salient_args silent drop**: `*args` silently dropped in `_build_arg_name_map` (lines 52-56)

## Gotchas
- **In-place ops**: `safe_copy` strips `tl_tensor_label_raw` from clone, ensuring
  in-place ops are always logged as new operations. Label propagated back after logging.
- **Barcode nesting detection**: Random barcodes detect bottom-level vs wrapper functions.
  If barcode unchanged after call → no nested torch calls → log it.
- **Buffer guard**: `torch_funcs.py:108` must check `not hasattr(t, "tl_tensor_label_raw")`
  to prevent duplicate buffer entries when multiple functions use the same buffer.
- **Fast-path validation**: 3 levels — tensor existence, execution order, parent sets.
  Graph divergence between passes raises an error.
- **pause_logging()**: Must wrap `activation_postfunc` calls and `get_tensor_memory_amount()`
  to prevent recursive logging of internal torch operations.
- **arg_positions dynamic cache**: Never cleared on torch version upgrades — could serve
  stale specs if torch updates function signatures.
