# capture/ — Real-Time Tensor Operation Logging

## What This Does
Captures every tensor operation during a model's forward pass. Handles both exhaustive
mode (full metadata, ~131 fields per operation) and fast mode (reuses prior graph structure,
saves only requested tensor data).

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `trace.py` | 500 | Forward-pass orchestration: input normalization, model execution, session setup/cleanup |
| `output_tensors.py` | 898 | Core logging: builds LayerPassLog entries, exhaustive/fast path split, identity detection |
| `source_tensors.py` | 357 | Logs input and buffer tensors as source nodes in the graph |
| `tensor_tracking.py` | 407 | Barcode system, parent-child links, backward hooks, arg hashing |
| `arg_positions.py` | 961 | O(1) tensor extraction via 3-tier lookup: static table (639 entries), dynamic cache, BFS fallback |
| `salient_args.py` | 444 | Extracts significant function args (hyperparameters) for metadata. 27 extractors for 50+ layer types |
| `flops.py` | 1393 | Per-operation FLOPs computation (3-tier: zero/elementwise/specialty, ~290 ops) |

## Key Functions

### trace.py
- `run_and_log_inputs_through_model()` — Core runner. Normalizes inputs, prepares model,
  runs forward pass inside `active_logging()` context, then triggers postprocessing.
- `save_new_activations()` — Re-logs activations for new inputs on already-traced model.
- Critical ordering: RNG capture → active_logging → model forward → cleanup → postprocess

### output_tensors.py
- `log_function_output_tensors_exhaustive()` — Creates 131-field dict per output tensor,
  builds LayerPassLog, tracks parent-child relationships
- `log_function_output_tensors_fast()` — Maps operation counter to existing raw label,
  validates function name matches, updates only tensor data + timing + RNG
- `_output_should_be_logged()` — Tensor logged if unlabeled OR bottom-level function
- `cond_branch_then_children` field added for conditional branch THEN detection

### source_tensors.py
- `log_source_tensor_exhaustive()` / `log_source_tensor_fast()` — Marks input/buffer
  tensors with metadata. Buffers get BufferLog instances.

### tensor_tracking.py
- `_get_operation_equivalence_type()` — Structural fingerprint for loop detection:
  `{func_name}_{arg_hash}[_outindex{i}][_module{origin}]`
- `_locate_parent_tensors_in_args()` — Maps parent positions in function arguments
  (only 2 nesting levels tracked — deeper parents may get wrong arg positions)
- `_add_backward_hook()` — Registers gradient capture (uses weakref to avoid GC leaks)

### arg_positions.py
- 3-tier O(1) lookup: static `FUNC_ARG_SPECS` table → dynamic `_DYNAMIC_SPEC_CACHE` → BFS fallback
- `ArgSpec` frozen dataclass: `tensor_args`, `tensor_kwargs`, `param_args`, `param_kwargs`
- `extract_tensors_and_params()` — Main entry point, returns (tensors, params) from args/kwargs

### salient_args.py
- `@_register()` pattern for extractors per layer type
- `_build_arg_name_map()` maps positional args to named params
- Failure-safe: try-except returns `{}` on any error
- `_get()` helper returns `None` on missing keys (graceful degradation)

### flops.py
- 3-tier system: ZERO_FLOPS_OPS (view, reshape = 0), ELEMENTWISE_FLOPS (relu, sigmoid),
  SPECIALTY_HANDLERS (conv2d, matmul — shape-aware computation)
- MAC = 2 FLOPs convention

## Label Formats
- Source tensors: `{type}_{num}_raw` (e.g., `"input_0_raw"`, `"buffer_1_raw"`)
- Function outputs: `{type}_{num}_{counter}_raw` (e.g., `"conv2d_1_5_raw"`)
- Labels are raw during capture; renamed to final labels in postprocess/labeling.py

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

## Related
- [decoration/](../decoration/CLAUDE.md) — Provides the decorated wrappers that call into this package
- [postprocess/](../postprocess/CLAUDE.md) — Processes the raw graph after capture completes
- [data_classes/](../data_classes/CLAUDE.md) — LayerPassLog and ModelLog definitions
