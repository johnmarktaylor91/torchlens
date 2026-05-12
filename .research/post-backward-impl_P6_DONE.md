# Post-backward Megasprint P6 DONE

## Scope

Implemented MLX hardening on Linux CPU for the P6 MVP:

- MLX `intervention_ready=True`, live hooks, and `save_grads=True` now hard-reject.
- MLX wrappers are rebound per trace and unwrapped after capture.
- MLX source arrays now emit resolvable input op logs, so `Trace.draw()` handles input parents.
- MLX arrays are audit-nulled during portable save through `_ScrubOptions.unsupported_tensor_records`.
- `_backend_name` survives portable save/load.
- MLX optional dependency is pinned to `mlx>=0.26,<0.27`.
- Wrapper coverage includes Conv2d, normalization layers, Embedding, Dropout, MultiHeadAttention,
  additional reductions, shape ops, and activations.

## Commits

- `a74f0dd fix(mlx): hard-reject unsupported preview options`
- `429c11c fix(mlx): bind wrappers per trace and resolve inputs`
- `84622f1 fix(io): audit-null mlx tensors on save`
- `3f99767 feat(mlx): expand wrapper coverage`

## Tests

Passed:

```bash
pytest tests/test_mlx_backend_smoke.py tests/test_mlx_hardening.py -v
ruff check torchlens tests/test_mlx_backend_smoke.py tests/test_mlx_hardening.py tests/test_io_pickle.py pyproject.toml --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short
```

`ruff check . --fix` is blocked by a pre-existing unrelated dirty notebook syntax error:

```text
notebooks/torchlens_in_10_minutes.ipynb:cell 27:1:25
from torchvision import
```

That notebook was already modified before P6 and is outside this phase scope.

## Deferred

MLX module attribution remains deferred to P6.5 per AD-28. No MLX backward,
intervention API, validation, JIT, or `mx.compile` support was added.
