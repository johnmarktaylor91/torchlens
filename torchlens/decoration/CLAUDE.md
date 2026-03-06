# decoration/ — Permanent Torch Function Wrapping

## What This Does
Wraps all PyTorch functions once at import time with toggle-gated logging wrappers,
and prepares models for logging by wrapping their forward methods and tagging modules.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `torch_funcs.py` | 490 | One-time decoration of ~2000 torch functions, sys.modules crawl, core interceptor |
| `model_prep.py` | 685 | Model preparation (one-time + per-session), module forward decorator, buffer discovery |

## torch_funcs.py — Key Functions

### `decorate_all_once()`
Called at `import torchlens` time. Wraps each function in ORIG_TORCH_FUNCS with
`torch_func_decorator`. Idempotent. Populates:
- `_state._orig_to_decorated` / `_state._decorated_to_orig` — bidirectional maps
- `_state._func_argnames` — pre-computed argument names
- JIT builtin table registration for `torch.jit.script` compatibility
- Shared originals (e.g., `torch.cos` and `torch._VF.cos`) reuse the same wrapper

### `torch_func_decorator(func)` — The Core Interceptor
For each intercepted call:
1. **Fast path**: `if not _logging_enabled: return func(...)` (one bool check)
2. Extract tensor args, generate random barcode, execute original function
3. **Barcode nesting detection**: If barcode unchanged → bottom-level function → log it
4. **In-place detection**: `id(out) == id(args[0])` → safe_copy strips label → always logged
5. Hand off to `capture/output_tensors.py` for logging

### `patch_detached_references()`
4-level sys.modules crawl using `_orig_to_decorated`:
1. Module `__dict__` values — `from torch import cos`
2. Class `__dict__` values — class attributes storing torch functions
3. Function `__defaults__` / `__kwdefaults__` — default arguments
4. Model instances — `self.act = torch.relu`

### DeviceContext Bypass
Python wrappers bypass C-level TorchFunctionMode dispatch. When `torch.device('meta')`
context is active (e.g., HuggingFace `from_pretrained`), factory functions need manual
device kwarg injection. Handled in wrapper fast path.

## model_prep.py — Key Functions

### `_prepare_model_once(model)`
Cached in `_state._prepared_models` (WeakSet). One-time per model instance:
- Assigns `tl_module_address`, `tl_module_type` to all submodules
- Wraps `module.forward` with `module_forward_decorator`
- Caches class metadata (source file, signature, docstring)

### `_prepare_model_session(model)`
Per-call setup:
- Forces `requires_grad=True` on all params (needed for grad_fn metadata)
- Initializes session-scoped `tl_*` attributes on modules
- Discovers and tags buffer tensors

### `module_forward_decorator`
Toggle-gated wrapper around `module.forward`:
- **Pre-forward** (`_handle_module_entry`): tracks inputs, increments pass counters,
  tags buffers, records module entry in thread
- **Post-forward** (`_handle_module_exit`): marks outputs, trims threads, detects
  bottom-level submodule exits
- **nn.Identity special handling**: Explicit `torch.identity(t)` call triggers logging
- **Fast path**: Skips `_handle_module_entry` but replicates identity injection for
  counter alignment

### `_cleanup_model_session(model)`
Restores `requires_grad`, removes session-scoped `tl_*` attributes.

## Gotchas
- `__wrapped__` removed from built-in function wrappers to prevent `inspect.unwrap`
  failures with `torch.jit.script`
- Two-phase model prep: permanent attrs survive sessions, session attrs are cleaned
- Fast-path module decorator skips `_handle_module_entry` entirely — any state that
  needs to align between passes must be replicated manually
- `_module_class_metadata_cache` cleared at session start to avoid stale data

## Related
- [capture/](../capture/CLAUDE.md) — Called by decorated wrappers to log operations
- `_state.py` — Global toggle and session state (no circular imports)
- `constants.py` — ORIG_TORCH_FUNCS defines which functions to decorate
