# decoration/ - Lazy Torch Function Wrapping

## What This Does
Installs toggle-gated wrappers around PyTorch functions and prepares model modules for
logging. In 2.x, wrapping is lazy: `import torchlens` has no side effects on `torch`.
`log_forward_pass()` calls `_ensure_model_prepared()`, which calls `wrap_torch()`,
`_prepare_model_once()`, `patch_detached_references()`, and `patch_model_instance()`.

## Files

| File | Purpose |
|------|---------|
| `torch_funcs.py` | Wrapper creation, explicit `wrap_torch()`/`unwrap_torch()`, sys.modules crawl, core interceptor |
| `model_prep.py` | One-time and per-session model preparation, module forward wrappers, buffer/param setup |
| `__init__.py` | Re-exports decoration lifecycle helpers |

## How It Connects

The wrapper hot path calls `capture/output_tensors.py` for forward records and fastlog
record contexts. Module forward wrappers in `model_prep.py` track module entry/exit and
source/buffer nodes. `_state.py` holds all mutable decoration maps and the logging toggle.

## torch_funcs.py - Key Design

### `wrap_torch()`
Installs wrappers on first use by calling `decorate_all_once()` and
`patch_detached_references()`. If `unwrap_torch()` previously removed wrappers, it reinstalls
cached wrapper objects instead of rebuilding them.

### `torch_func_decorator(func, func_name)`
1. Fast path: if `_logging_enabled` is false, optionally inject active device kwargs and call
   the original function.
2. Extract tensor and parameter args.
3. Apply barcode nesting detection.
4. Execute the original function.
5. Apply live hooks and log bottom-level output tensors.
6. Restore/propagate labels for in-place operations.

### `patch_detached_references()`
Incrementally scans loaded modules for stale references captured by `from torch import cos`
style imports. It also patches function defaults and model instance attributes.

### Explicit Lifecycle
`unwrap_torch()` restores original torch functions and clears logging state.
`wrapped()` is a context manager that wraps on entry and unwraps on exit. Public moved-name
shims expose these through `torchlens.decoration`, not top-level `__all__`.

## model_prep.py - Key Design

### Two-Phase Model Preparation
- `_prepare_model_once(model)` is cached per model and installs permanent attrs and forward
  wrappers.
- `_prepare_model_session(model, model_log, ...)` creates session attrs, discovers buffers,
  records params, and handles train-mode `requires_grad` semantics.

### Module Forward Decorator
- Pre-forward: input tracking, pass counters, buffer tagging, module entry.
- Post-forward: output tagging, module exit, bottom-level submodule detection.
- Fast path skips `_handle_module_entry` and must replicate only state needed for counter
  alignment and identity injection.

## DeviceContext and CPython Details
Python wrappers bypass some C-level PyTorch dispatch, so factory functions need manual
device kwarg injection in active `torch.device(...)` contexts. Replacing tensor indexing
also pollutes CPython sequence slots; `_fix_tensor_sequence_slot()` clears that after
wrap/unwrap cycles.
