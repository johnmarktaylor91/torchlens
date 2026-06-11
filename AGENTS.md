# TorchLens - Codex Agent Briefing

## Project Overview
TorchLens extracts activations and computational graph metadata from PyTorch eager
models. It lazily wraps PyTorch functions with toggle-gated wrappers on first capture,
runs forward passes with the logging toggle enabled, and logs operations into
`Trace`, `Layer`, and `Op` objects.

## Architecture
See `.project-context/architecture.md` for the older full map and
`.project-context/state_of_torchlens.md` for the current 2.x map. See subpackage
`CLAUDE.md` files for per-module details.

Key entry points:
- Main capture: `torchlens/user_funcs.py` - `trace()`, `show_model_graph()`,
  `draw_backward()`, `validate_forward_pass()`
- Sparse capture: `tl.record(model, x, save=...)` returns `Recording`; `Recording.to_trace()`
  materializes full graph structure with explicit errors for unsaved payload reads.
- Lazy decoration: `torchlens/decoration/model_prep.py:_ensure_model_prepared()` calls
  `wrap_torch()` and `patch_detached_references()`
- Forward-pass orchestration: `torchlens/capture/trace.py`
- Postprocess: `torchlens/postprocess/__init__.py` current 20-step pipeline
- Portable I/O: `torchlens/_io/bundle.py`, `torchlens/_io/tlspec.py`, `torchlens/io/__init__.py`
- Intervention: `torchlens/intervention/` plus top-level selector/helper aliases
- Visualization: `Trace.draw(order_siblings=True)` applies a Graphviz-only verified
  sibling-ordering post-pass for forward unrolled graphs under the node cap.

Common unified capture patterns:

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("attn"),
    intervene=tl.when(tl.func("attn"), tl.zero_ablate()),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
```

## Conventions
- Conventional commits: prefer `docs(scope):`, `chore(scope):`, `test(scope):` for
  non-release changes; never use major-bump markers casually.
- TorchLens host-object metadata lives under `obj._tl`; sub-fields are snake_case and
  new metadata should extend a `TorchLensMeta` subclass rather than adding `tl_*` attrs.
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
5. Module suffixes are appended to `equivalence_class` at op creation before loop detection.
6. RNG state capture/restore must happen before `active_logging()` context.
7. `_build_module_logs` must not run in `postprocess_fast`; `_module_build_data` is not
   populated in fast mode.
8. `_pass_finished` is not reset between passes; this is intentional for fast-path lookups.
9. Portable `.tlspec` public schema is manifest-only; executable callables are not portable.
10. `backward_ready=True` rejects contradictory detaching/disk-save settings and preserves user
    `requires_grad` choices.
11. Sibling ordering is forward/unrolled/dot-only; collapsed, rolled, backward, focused,
    conditional, and large graphs must conservatively no-op.
12. Predicate `save=` is the primary selective-capture spelling; `record(keep_op=...)` and
    `record(keep_module=...)` are deprecated aliases.
13. `torch.func` / functorch transforms are captured as boundary ops; do not expect their
    per-element internal eager operations to appear unless a future expand-inside mode exists.

## Known Gotchas
- `__wrapped__` is removed from built-in function wrappers to avoid `inspect.unwrap`
  failures.
- Fast-path module decoration skips `_handle_module_entry`; alignment state must be
  replicated manually.
- `get_tensor_memory_amount()` must use `pause_logging()` because `nelement()` and
  `element_size()` are decorated.
- If a `@property` raises `AttributeError`, Python falls through to `__getattr__`; use
  `ValueError` for TorchLens multi-pass access errors.
- `copy()` on `Op` shallow-copies selected graph fields and deep-copies the rest.
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
