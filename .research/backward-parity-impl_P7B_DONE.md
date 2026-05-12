# Phase P7B Done

## Changes

- Restored capture-time `out_postfunc` handling in `torchlens/backends/torch/ops.py`
  so the LiveOpRecord field-dict save path uses the existing `OpLog` postfunc
  wrapper and train-mode differentiability validation.
- `train_mode=True` now rejects `out_postfunc=lambda t: t.detach()` with
  `TrainingModeConfigError` containing `disconnected from the autograd graph`.

## Test Results

- `pytest tests/test_activation_postfunc_refactor.py::test_train_mode_out_postfunc_detach_rejected tests/test_activation_postfunc_refactor.py::test_train_mode_out_postfunc_int_rejected tests/test_activation_postfunc_refactor.py::test_train_mode_out_postfunc_connected_ops -q`
  - Result: PASS
  - Counts: 3 passed, 5 warnings
- `pytest tests/test_activation_postfunc_refactor.py -q`
  - Result: PASS
  - Counts: 12 passed, 16 warnings
- `ruff check torchlens/backends/torch/ops.py --fix`
  - Result: PASS
  - Output: `All checks passed!`
- `ruff check . --fix`
  - Result: FAIL before reaching this change
  - Blocker: pre-existing syntax error in dirty notebook
    `notebooks/torchlens_in_10_minutes.ipynb:cell 27`, `from torchvision import`
- `pytest tests/ -m "not slow" -x --tb=short`
  - Result: FAIL
  - Counts: 1 failed, 74 passed, 1 skipped, 209 deselected, 66 warnings
  - Failing test: `tests/test_api_surface.py::test_all_size_exactly_40`
  - Failure: `len(torchlens.__all__) == 46`, expected `40`

## T2 Status

Full T2 is not green yet. The P7B regression file is green, but T2 is blocked by
API-surface drift unrelated to the train-mode `out_postfunc` validation path.

## Concerns

- P7 should not resume until the `torchlens.__all__` budget mismatch is resolved
  or the expected API budget is updated.
- Repository-wide ruff is blocked by an unrelated dirty notebook syntax error
  outside this task's scope.
