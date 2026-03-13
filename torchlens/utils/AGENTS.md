# utils/ — Shared Utilities

## What This Is
Stateless helper functions used across the codebase. No torchlens-specific business logic —
these are general-purpose utilities for tensor operations, argument handling, RNG management,
hashing, introspection, display formatting, and collection manipulation.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `arg_handling.py` | ~150 | Safe argument copying, input normalization for model forward calls |
| `collections.py` | ~100 | Iterable helpers: `is_iterable`, `ensure_iterable`, `index_nested`, `remove_entry_from_list` |
| `display.py` | ~120 | Human-readable formatting: `int_list_to_compact_str`, `human_readable_size`, `warn_parallel` |
| `hashing.py` | ~80 | Barcode generation: `make_random_barcode` (8-char IDs), `make_short_barcode_from_input` |
| `introspection.py` | ~200 | Object introspection: `get_vars_of_type_from_obj` (recursive tensor finder), `nested_getattr`/`nested_assign` |
| `rng.py` | ~100 | RNG state capture/restore: Python, NumPy, torch, CUDA states for reproducible replay |
| `tensor_utils.py` | ~200 | Tensor ops: `safe_copy`, `safe_to`, `tensor_nanequal`, `get_tensor_memory_amount` |

## Key Functions by Use Case

### Argument Safety (arg_handling.py)
- `_safe_copy_arg(arg)` — Clones tensors, recurses into containers, leaves non-tensors as-is.
  Prevents deepcopy infinite loops on complex wrappers with circular references.
- `normalize_input_args(model, args)` — Uses `inspect.getfullargspec` to check if model
  expects 1 arg. If so and input is tuple/list, wraps as `[input]` instead of unpacking.

### Tensor Operations (tensor_utils.py)
- `safe_copy(t)` — Clones tensor using `pause_logging()` to avoid recursive logging.
  Strips all `tl_*` attributes from the clone (critical for in-place op detection).
- `safe_to(t, device)` — Device transfer inside `pause_logging()`.
- `tensor_nanequal(t1, t2)` — NaN-aware comparison using complex-safe `view_as_real`.
- `get_tensor_memory_amount(t)` — Must use `pause_logging()` because `nelement()` and
  `element_size()` are decorated tensor methods.
- `MAX_FLOATING_POINT_TOLERANCE = 3e-6` — Used by validation.

### Clean Torch Functions (tensor_utils.py)
Imports original torch functions at module-load time BEFORE decoration:
```python
clean_clone = torch.clone
clean_to = torch.Tensor.to
```
Used internally to avoid triggering decorated wrappers during logging.

### RNG State (rng.py)
- `log_current_rng_states()` — Captures Python random, NumPy, torch CPU, and CUDA states.
- `set_rng_from_saved_states(states)` — Restores all states for deterministic replay.
- Critical for two-pass architecture: exhaustive pass captures RNG, fast pass restores it.

### Introspection (introspection.py)
- `get_vars_of_type_from_obj(obj, var_type, depth)` — Recursive depth-limited search
  for tensors/modules in nested structures. `_ATTR_SKIP_SET` filters `.T`, `.mT`, `.H`.
- `_is_cuda_available()` — Caches `torch.cuda.is_available()` once per process.

### Hashing (hashing.py)
- `make_random_barcode()` — 8-char alphanumeric IDs for barcode nesting detection.
- `make_short_barcode_from_input(args)` — Hashes argument lists for tensor identity tracking
  and operation equivalence fingerprinting.

## Known Bugs
- **BFLOAT16-TOL**: `MAX_FLOATING_POINT_TOLERANCE = 3e-6` is 2,600x too tight for bfloat16
  (epsilon ~7.8e-3). No dtype-specific tolerance adjustment in `tensor_nanequal()`.
- **QUANTIZED-CRASH**: `tensor_nanequal()` calls `.isinf()` which raises AttributeError on
  quantized tensors.
- **RNG-MULTI-GPU**: `rng.py:70` only captures RNG state for CUDA device 0. Multi-GPU
  models get non-deterministic validation replays.
- **HASH-COLLISION**: `make_short_barcode_from_input` uses Python `hash()` + base64
  truncation. ~0.3-0.5% collision risk at 1K params. Could cause false same-layer grouping.

## Gotchas
- `get_tensor_memory_amount()` MUST use `pause_logging()` — `nelement()` and
  `element_size()` are decorated and would trigger infinite recursive logging.
- `tensor_nanequal` handles complex tensors via `view_as_real`/`view_as_complex` because
  `torch.nan_to_num` doesn't support complex types.
- Clean function imports must happen at module load time (before `decorate_all_once()`).
