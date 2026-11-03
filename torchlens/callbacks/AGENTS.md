# callbacks/ - Implementation Guide

Trainer-framework callback integrations. The namespace is LAZY by design:
`torchlens.callbacks.lightning` imports only on attribute access
(`_CALLBACK_MODULES` gate in `__init__.py`), so `import torchlens` never
pays for (or fails on) an absent trainer framework.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | PEP-562 lazy loader over the callback module set (`{"lightning"}`) |
| `lightning.py` | `LayerProfilerCallback` and `SummaryCallback` (the `__all__` names) — built at ACCESS time by `_build_layer_profiler_callback()` / `_build_summary_callback()` on top of Lightning's callback base (`lightning.pytorch`, else standalone `pytorch_lightning`) |

## Gotchas

- Neither callback exists as a static class: `lightning.py`'s
  own module `__getattr__` resolves it lazily so the Lightning import (and
  its optional-extra refusal) happens on first use, never at module import.
  This is the same deferred-extras pattern as the appliance packages — do
  not "fix" it to import Lightning at the top.
- `_LayerProfilerCallbackMixin` / `_SummaryCallbackMixin` hold the
  framework-agnostic logic; the Lightning base is grafted on in the builders.
- `SummaryCallback` delegates the extra forward to `tl.summary`, which owns
  the eval/no_grad run and the exact mode and RNG restore; do not re-implement
  that restore in the callback.

## Tests

`tests/test_callbacks_lightning.py` (uses `pytest.importorskip` for the
optional dependency).
