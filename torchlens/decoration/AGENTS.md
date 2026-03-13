# decoration/ — Implementation Guide

## Gotchas
- `__wrapped__` removed from built-in function wrappers to prevent `inspect.unwrap`
  failures with `torch.jit.script`
- Two-phase model prep: permanent attrs survive sessions, session attrs are cleaned
- Fast-path module decorator skips `_handle_module_entry` entirely — any state that
  needs to align between passes must be replicated manually
- `_module_class_metadata_cache` cleared at session start to avoid stale data
- Buffer auto-registration guard prevents re-registering already-labeled buffers

## Known Bugs
- **DEVICE-CONTEXT-LOGGING**: `_maybe_inject_device_kwarg` only called at line 241
  (non-logging path). During active logging, factory functions don't get device injection.
  Breaks HuggingFace `from_pretrained` under active logging.
