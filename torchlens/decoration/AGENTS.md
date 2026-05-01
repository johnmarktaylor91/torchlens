# decoration/ - Implementation Guide

## Lifecycle
- `import torchlens` must not mutate the torch namespace.
- `_ensure_model_prepared(model)` is the normal entry point before capture.
- `wrap_torch()` is idempotent and lazily installs or reinstalls wrappers.
- `unwrap_torch()` restores original torch functions and disables logging.
- `patch_detached_references()` is incremental and should not rescan already-seen modules
  unless its cache is explicitly cleared.

## Gotchas
- `__wrapped__` is removed from built-in wrappers to prevent `inspect.unwrap` failures with
  `torch.jit.script`.
- `_state._orig_to_decorated` and `_state._decorated_to_orig` must remain 1:1 for shared
  builtins such as `torch.cos` and `torch._VF.cos`.
- The wrapper fast path is hot; keep logging-off overhead to a bool check plus required
  device-context handling.
- `DeviceContext` injection must work in both logging and non-logging paths.
- Tensor `__getitem__` replacement can corrupt CPython `sq_item`; keep
  `_fix_tensor_sequence_slot()` after wrap/unwrap.
- Fast-path module decorator skips `_handle_module_entry`; alignment state must be
  duplicated intentionally.
- `_module_class_metadata_cache` is per-session and should not leak stale source metadata.

## Model Prep Rules
- Permanent attrs survive sessions: module address/type and forward wrappers.
- Session attrs are cleaned by `_cleanup_model_session()`.
- In default capture, parameter setup may temporarily force `requires_grad`; cleanup restores.
- In `train_mode=True`, preserve user `requires_grad` choices.
- Buffer tagging must avoid duplicate buffer source nodes.
- Do not import from higher-level public API modules here; keep dependencies pointed toward
  `_state`, `constants`, `capture`, and `utils`.

## Public Lifecycle Helpers
`decoration/__init__.py` exports `decorate_all_once`, `wrap_torch`, `unwrap_torch`,
`wrapped`, `patch_detached_references`, `patch_model_instance`, and cache helpers. Top-level
access is through deprecation shims, not `torchlens.__all__`.

## Known Risks
- Stale detached references can survive in exotic containers that the sys.modules/model crawl
  does not inspect.
- Active logging during library initialization can expose device/context edge cases.
- Reintroducing import-time decoration would break the current 2.x compatibility contract.
