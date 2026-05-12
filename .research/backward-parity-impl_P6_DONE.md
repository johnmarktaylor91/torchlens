# Backward-Parity Sprint P6 DONE

Date: 2026-05-12
Branch: `codex/backward-parity-P1-alpha3`

## Scope

P6 shipped observer backward parity and bundle backward visualization regression
coverage under the AD-32 reduced scope. The per-layer gradient oracle remains
deferred.

## Commits

- `2a024c3 feat(observers): direction kwarg for backward taps`
- `f05a356 test(observers): cover backward taps and bundle viz`
- `ec9e485 chore(changelog): record P6 observer bundle polish`
- `ef905b8 chore(todos): file deferred per-layer grad oracle`

## Changes

- `torchlens/observers.py`
  - Added `tap(..., direction="forward"|"backward"|"both")`.
  - Added backward gradient recording via `TapObserver.record_backward`.
  - Extended `TapRecord` with `direction`, `grad_kind`, and
    `backward_call_index`.
  - Added `record_span(..., direction="both")` metadata.
- `torchlens/intervention/hooks.py`
  - Expanded both-direction hooks into concrete forward/backward hook entries.
  - Routed backward observer normalization through `record_backward`.
  - Switched helper mount validation to resolver direction.
- `torchlens/backends/torch/backward.py`
  - Reused initial capture hook plans during `trace.log_backward(...)`.
- `torchlens/user_funcs.py`
  - Stored the initial live hook plan on traces for backward capture.
- `tests/test_observers_backward.py`
  - Added backward tap and span regression coverage.
- `tests/test_multi_trace_followons.py`
  - Added bundle backward regression coverage for the existing per-member path.
- `CHANGELOG.md`
  - Recorded P6 observer and bundle polish under Unreleased.
- `.project-context/todos.md`
  - Filed the deferred per-layer-grad oracle follow-up sprint.

## Verification

- `pytest tests/test_observers_backward.py -q`
  - `11 passed, 2 warnings`
- `pytest tests/test_multi_trace_followons.py -q -k "bundle_backward or show_bundle_graph"`
  - `3 passed, 2 deselected, 2 warnings`
- `ruff check torchlens/observers.py torchlens/intervention/hooks.py torchlens/backends/torch/backward.py torchlens/user_funcs.py tests/test_observers_backward.py tests/test_multi_trace_followons.py --fix`
  - `All checks passed!`
- `mypy torchlens/`
  - `Success: no issues found in 199 source files`

## Blocked Gate

- `ruff check . --fix` is currently blocked by an unrelated dirty notebook:
  `notebooks/torchlens_in_10_minutes.ipynb:cell 27` contains
  `from torchvision import ` with no imported symbol.

## Notes

- No TorchLens version bump.
- No per-layer-grad oracle helpers or diagnostic dataclasses were added.
- Unified backward bundle supergraph was intentionally skipped; P6 keeps the
  existing per-member backward bundle path.
