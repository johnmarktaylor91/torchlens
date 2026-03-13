# decoration/ — Permanent Torch Function Wrapping

## What This Does
Wraps all PyTorch functions once at import time with toggle-gated logging wrappers,
and prepares models for logging by wrapping their forward methods and tagging modules.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `torch_funcs.py` | 748 | One-time decoration of ~2000 torch functions, sys.modules crawl, core interceptor, DeviceContext bypass |
| `model_prep.py` | 962 | Model preparation (one-time + per-session), module forward decorator, buffer discovery |

## How It Connects

This is the foundation layer. `decorate_all_once()` runs at `import torchlens` time and
wraps every torch function with `torch_func_decorator`. These wrappers call into `capture/`
for logging. `model_prep.py` wraps `module.forward` methods to track module entry/exit.

The core interceptor in `torch_func_decorator` is the single hottest code path — every
torch function call passes through it, so the fast path (logging disabled) must be minimal.

## torch_funcs.py — Key Design

### `decorate_all_once()`
Wraps each function in ORIG_TORCH_FUNCS. Populates bidirectional maps
(`_orig_to_decorated` / `_decorated_to_orig`), pre-computed argument names, JIT builtin
table. Shared originals (e.g., `torch.cos` and `torch._VF.cos`) reuse the same wrapper.

### `torch_func_decorator(func)` — The Core Interceptor
1. **Fast path**: `if not _logging_enabled: return func(...)` (one bool check)
2. Extract tensor args via `extract_tensors_and_params()` (O(1) ArgSpec lookup)
3. Generate random barcode, execute original function
4. **Barcode nesting detection**: If barcode unchanged → bottom-level function → log it
5. **In-place detection**: `id(out) == id(args[0])` → safe_copy strips label → always logged
6. Hand off to `capture/output_tensors.py` for logging

### `patch_detached_references()`
4-level sys.modules crawl: module `__dict__` values, class `__dict__` values,
function `__defaults__`/`__kwdefaults__`, model instances.

### DeviceContext Bypass
Python wrappers bypass C-level TorchFunctionMode dispatch. Factory functions need manual
device kwarg injection when `torch.device('meta')` context is active.

## model_prep.py — Key Design

### Two-Phase Model Preparation
- `_prepare_model_once(model)` — Cached in WeakSet. Permanent attrs (`tl_module_address`,
  `tl_module_type`), forward wrappers, class metadata cache.
- `_prepare_model_session(model)` — Per-call: `requires_grad=True` on all params, session
  attrs, buffer discovery and tagging.

### Module Forward Decorator
- **Pre-forward**: tracks inputs, increments pass counters, tags buffers, records module entry
- **Post-forward**: marks outputs, trims threads, detects bottom-level submodule exits
- **nn.Identity special handling**: Explicit `torch.identity(t)` call triggers logging
- **Fast path**: Skips `_handle_module_entry` but replicates identity injection for counter alignment
