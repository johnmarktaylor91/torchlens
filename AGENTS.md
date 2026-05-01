# TorchLens - Codex Agent Briefing

## Project Overview
TorchLens extracts activations and computational graph metadata from PyTorch eager
models. It lazily wraps PyTorch functions with toggle-gated wrappers on first capture,
runs forward passes with the logging toggle enabled, and logs operations into
`ModelLog`, `LayerLog`, and `LayerPassLog` objects.

## Architecture
See `.project-context/architecture.md` for the older full map and
`.project-context/state_of_torchlens.md` for the current 2.x map. See subpackage
`CLAUDE.md` files for per-module details.

Key entry points:
- Main capture: `torchlens/user_funcs.py` - `log_forward_pass()`, `show_model_graph()`,
  `show_backward_graph()`, `validate_forward_pass()`
- Lazy decoration: `torchlens/decoration/model_prep.py:_ensure_model_prepared()` calls
  `wrap_torch()` and `patch_detached_references()`
- Forward-pass orchestration: `torchlens/capture/trace.py`
- Postprocess: `torchlens/postprocess/__init__.py` current 20-step pipeline
- Portable I/O: `torchlens/_io/bundle.py`, `torchlens/_io/tlspec.py`, `torchlens/io/__init__.py`
- Intervention: `torchlens/intervention/` plus top-level selector/helper aliases

## Conventions
- Conventional commits: prefer `docs(scope):`, `chore(scope):`, `test(scope):` for
  non-release changes; never use major-bump markers casually.
- `tl_` prefix on tensor/module attributes during logging
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state
- FIELD_ORDER constants in `constants.py` define canonical field sets; update both class
  fields and constants when adding fields
- NumPy-format docstrings on all functions
- Type hints on all functions
- Import order: stdlib -> third-party -> local (enforced by ruff)
- Line length: 100

## Quality Gates
Every task must pass before completion unless the task explicitly narrows verification:

```bash
ruff check . --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short
```

For changes touching module boundaries or public API, also run:

```bash
pytest tests/ -m "not slow" -x --tb=short
```

## Critical Invariants
1. `_state.py` must never import other torchlens modules.
2. `pause_logging()` must wrap internal torch ops during logging (`safe_copy`,
   `activation_postfunc`, `get_tensor_memory_amount`).
3. Wrappers are persistent after lazy installation; `_logging_enabled` gates behavior.
4. FIELD_ORDER constants and class definitions must stay in sync.
5. Step 6 module suffix mutation makes `_rebuild_pass_assignments` in Step 8 necessary.
6. RNG state capture/restore must happen before `active_logging()` context.
7. `_build_module_logs` must not run in `postprocess_fast`; `_module_build_data` is not
   populated in fast mode.
8. `_pass_finished` is not reset between passes; this is intentional for fast-path lookups.
9. Portable `.tlspec` public schema is manifest-only; executable callables are not portable.
10. `train_mode=True` rejects contradictory detaching/disk-save settings and preserves user
    `requires_grad` choices.

## Known Gotchas
- `__wrapped__` is removed from built-in function wrappers to avoid `inspect.unwrap`
  failures.
- Fast-path module decoration skips `_handle_module_entry`; alignment state must be
  replicated manually.
- `get_tensor_memory_amount()` must use `pause_logging()` because `nelement()` and
  `element_size()` are decorated.
- If a `@property` raises `AttributeError`, Python falls through to `__getattr__`; use
  `ValueError` for TorchLens multi-pass access errors.
- `copy()` on `LayerPassLog` shallow-copies selected graph fields and deep-copies the rest.
- `torchlens.__version__` and `pyproject.toml` are release-pipeline state; do not update them
  in feature/docs tasks unless release work explicitly asks for it.

## Build & Test

```bash
pip install -e ".[dev]"
pip install -e ".[test]"
pip install build && python -m build
pytest tests/ -m smoke
pytest tests/ -m "not slow"
pytest tests/
ruff format && ruff check --fix
```
