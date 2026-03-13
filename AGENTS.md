# TorchLens — Codex Agent Briefing

## Project Overview
TorchLens extracts activations and computational graph metadata from PyTorch models.
It permanently wraps all PyTorch functions at import time with toggle-gated wrappers,
runs forward passes with the toggle enabled, and logs every operation into
ModelLog/LayerLog/LayerPassLog objects.

## Architecture
See `.project-context/architecture.md` for the full map. See subpackage CLAUDE.md
files for per-module details.

Key entry points:
- Main API: `torchlens/user_funcs.py` — `log_forward_pass()`, `show_model_graph()`, `validate_forward_pass()`
- Import-time side effects: `torchlens/__init__.py` calls `decorate_all_once()` + `patch_detached_references()`
- Forward-pass orchestration: `torchlens/capture/trace.py`
- 18-step postprocess: `torchlens/postprocess/__init__.py`

## Conventions (see `.project-context/conventions.md`)
- Conventional commits: `fix(scope):`, `feat(scope):`, `chore(scope):`
- `tl_` prefix on tensor/module attributes during logging
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed
- FIELD_ORDER constants in `constants.py` define canonical field sets — update both class + constants
- NumPy-format docstrings on all functions
- Type hints on all functions
- Import order: stdlib → third-party → local (enforced by ruff)
- Line length: 100

## Quality Gates
Every task must pass before completion:
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
1. `_state.py` must NEVER import other torchlens modules (prevents circular deps)
2. `pause_logging()` must wrap any internal torch ops during logging (`safe_copy`, `activation_postfunc`, `get_tensor_memory_amount`)
3. Decorated wrappers are permanent — `_logging_enabled` bool gates behavior, not decoration state
4. FIELD_ORDER constants and class definitions must stay in sync
5. Step 6 module suffix mutation makes `_rebuild_pass_assignments` (Step 8) NECESSARY — not just defensive
6. RNG state capture/restore must happen BEFORE `active_logging()` context
7. `_build_module_logs` must NOT be called in `postprocess_fast` — `_module_build_data` isn't populated in fast mode
8. `_pass_finished` not reset between passes — intentional for fast-path lookups

## Known Gotchas
- Adding new fields requires updating BOTH the class definition AND `constants.py` FIELD_ORDER
- `__wrapped__` removed from built-in function wrappers to prevent `inspect.unwrap` failures
- Fast-path module decorator skips `_handle_module_entry` — alignment must be replicated manually
- `get_tensor_memory_amount()` MUST use `pause_logging()` — `nelement()`/`element_size()` are decorated
- Python property/__getattr__ trap: If a `@property` raises `AttributeError`, Python falls through to `__getattr__`. Use `ValueError` instead.
- `copy()` on LayerPassLog: shallow-copies 8 specific fields, deep-copies rest — safe only because downstream uses assignment not mutation

## Build & Test
```bash
pip install -e ".[dev]"         # dev install
pip install -e ".[test]"        # with test deps (timm, torchvision, etc.)
pip install build && python -m build  # build package
pytest tests/ -m smoke          # smoke tests (~6s)
pytest tests/ -m "not slow"     # skip slow real-world models
pytest tests/                   # all tests
ruff format && ruff check --fix # lint
```
