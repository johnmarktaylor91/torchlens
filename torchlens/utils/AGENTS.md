# utils/ - Shared Utilities

## What This Is
Small shared helpers used across capture, postprocess, validation, and data classes. Keep
this package mostly stateless and free of high-level TorchLens business logic.

## Files

| File | Purpose |
|------|---------|
| `arg_handling.py` | Safe arg copying and input normalization for model calls |
| `collections.py` | Iterable and nested-index helpers |
| `display.py` | Human-readable formatting, verbosity, identity helper |
| `hashing.py` | Barcodes, short hashes, graph-shape hash |
| `introspection.py` | Recursive object search and nested getattr/assign |
| `rng.py` | Python, NumPy, torch, CUDA, and autocast state capture/restore |
| `tensor_utils.py` | `safe_copy`, `safe_to`, `tensor_nanequal`, tensor memory helpers |
| `source_links.py` | Source link helpers for reports/rendering |
| `__init__.py` | Package marker |

## Tensor Operations
- `safe_copy()` clones through clean torch functions and strips `tl_*` attrs.
- `safe_to()` moves tensors under `pause_logging()`.
- `tensor_nanequal()` is NaN-aware and complex-aware.
- `get_tensor_memory_amount()` must use `pause_logging()` because tensor methods are wrapped.
- `MAX_FLOATING_POINT_TOLERANCE` is shared by validation.

## RNG and Autocast
- `log_current_rng_states()` and `set_rng_from_saved_states()` support deterministic replay.
- Autocast state helpers are used by wrapper execution context and validation replay.
- Capture RNG before entering `active_logging()`.

## Hashing
- `make_random_barcode()` supports barcode nesting detection.
- `make_short_barcode_from_input()` feeds operation equivalence.
- `compute_graph_shape_hash()` is used during postprocess finalization.

## Introspection
- `get_vars_of_type_from_obj()` is a bounded recursive finder for tensors/modules.
- `_ATTR_SKIP_SET` avoids expensive tensor pseudo-properties.
- `_is_cuda_available()` caches CUDA availability to avoid repeated driver probes.

## Gotchas
- Clean torch function imports must happen before decoration or use originals from `_state`.
- Avoid importing high-level modules here; utility imports should stay low in the dependency graph.
- Validation tolerance and quantized tensor behavior are known edge cases.
- Hash collisions are possible; do not use short hashes as security or persistence identifiers.
