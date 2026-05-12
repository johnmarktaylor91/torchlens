# Backward Parity Sprint P4 DONE

- Phase: P4 -- backward intervention parity
- Branch: `codex/backward-parity-P1-alpha3`
- Implementation SHA: `57a9e3d`
- Completed: 2026-05-12T05:46:09-04:00
- Files changed in P4 scope:
  - `torchlens/intervention/selectors.py`
  - `torchlens/intervention/resolver.py`
  - `torchlens/intervention/errors.py`
  - `torchlens/backends/torch/backward.py`
  - `torchlens/intervention/runtime.py`
  - `torchlens/intervention/hooks.py`
  - `torchlens/data_classes/model_log.py`
  - `torchlens/intervention/bundle.py`
  - `torchlens/visualization/node_spec.py`
  - `torchlens/intervention/helpers.py`
  - `torchlens/intervention/__init__.py`
  - `torchlens/__init__.py`
  - `tests/test_backward_intervention.py`
  - `CHANGELOG.md`

## Summary

- Added `tl.grad_fn(...)`, `tl.intervening()`, and `tl.grad_fn_label(...)`.
- Added resolver direction inference and GradFnLog site resolution.
- Added live backward helper dispatch from grad_fn post-hooks and AccumulateGrad prehooks.
- Added `tl.grad_clip`, `tl.grad_noise`, and `tl.grad_clamp`.
- Added mount-shape validation with `HelperMountError`.
- Documented that backward replay remains intentionally unsupported.

## Verification

- `pytest tests/test_backward_intervention.py -v`: 24 passed, 1 warning.
- `pytest tests/ -m smoke -x --tb=short`: 194 passed, 1 skipped, 2184 deselected, 117 warnings.
- `mypy torchlens/`: success, no issues in 199 source files.
- `ruff check torchlens tests/test_backward_intervention.py --fix`: passed.
- `ruff check . --fix`: blocked by pre-existing notebook syntax error in
  `notebooks/torchlens_in_10_minutes.ipynb` cell 27: `from torchvision import`.

## Ready For P5

P4 acceptance gates are satisfied except the repository-wide ruff command, which
is blocked by unrelated dirty notebook state that predated this phase.
