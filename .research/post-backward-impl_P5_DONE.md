# Post-backward Megasprint P5 DONE

Implemented PATH E per-module-output gradient oracle.

## Scope

- Added stock per-module-output gradient capture via `nn.Module.register_forward_hook`.
- Added `LayerGradReport` and `_compare_module_output_grads`.
- Added opt-in `validate_backward_pass(validate_layer_grads=True)`.
- Added 20 PATH E tests in `tests/test_layer_grad_oracle.py`.
- Did not implement PATH B per-operation oracle.

## Verification

- `pytest tests/test_layer_grad_oracle.py -q -x --tb=short`: 20 passed.
- `mypy torchlens/`: passed.
- `pytest tests/ -m smoke -x --tb=short`: 199 passed.

## Blocked Gates

- `ruff check . --fix` is blocked by an unrelated pre-existing notebook syntax error in
  `notebooks/torchlens_in_10_minutes.ipynb` cell 27: `from torchvision import `.
  Scoped ruff over P5 files passes.
- `pytest tests/ -m "not slow" -x --tb=short` is blocked by an existing API surface
  assertion: `tests/test_api_surface.py::test_all_size_exactly_46` observes
  `len(torchlens.__all__) == 47`.
