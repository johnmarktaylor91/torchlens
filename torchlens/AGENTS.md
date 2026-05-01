# torchlens/ - Implementation Guide

## Files in This Directory

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports, moved-name deprecation shims, `peek`, `extract`, `batched_extract`, validation aliases |
| `_state.py` | Global toggle, active log, decoration maps, prepared model registry; must not import torchlens modules |
| `_run_state.py` | Runtime state enum surfaced through `torchlens.io` |
| `_errors.py`, `_robustness.py`, `_training_validation.py` | Legacy/public error and compatibility helpers |
| `_literals.py` | Shared literal types for options and modes |
| `_source_links.py` | Source-link helpers used by reports/visualization |
| `constants.py` | FIELD_ORDER tuples and decorated torch function discovery |
| `options.py` | Immutable grouped options and flat-argument merge helpers |
| `observers.py` | `tap`, `record_span`, and active span state |
| `types.py` | Moved public type aliases not kept in top-level `__all__` |
| `user_funcs.py` | Main capture, summary, visualization, validation, and bundle graph entry points |

## Attribute Conventions
- `tl_` prefix on tensor/module attributes during logging.
- Permanent attrs: `tl_module_address`, `tl_module_type`, forward wrapper markers.
- Session attrs: `tl_source_model_log`, `tl_module_pass_num`, buffer labels, temporary counters.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state.

## Public Surface
`torchlens.__all__` is intentionally small and currently has 40 names. New user-facing
objects should usually live under submodules (`torchlens.io`, `torchlens.options`,
`torchlens.bridge`, `torchlens.errors`, etc.) with moved-name shims only when compatibility
requires them.

## Constants as Ordering Spec
FIELD_ORDER tuples define canonical serialized and display field sets. When adding a field,
update the class definition, the appropriate FIELD_ORDER constant, metadata tests, and any
`to_pandas()`/summary surface that should expose it.

## Critical Invariants
1. `_state.py` has no outgoing torchlens imports.
2. `_ensure_model_prepared()` is the lazy wrapping chokepoint; do not reintroduce import-time
   torch namespace mutation.
3. RNG state capture/restore must happen before `active_logging()`.
4. Internal torch ops during capture must be wrapped in `pause_logging()`.
5. Step 6 module suffix mutation makes loop pass rebuilding necessary.
6. `postprocess_fast()` must not call `_build_module_logs()`.
7. `train_mode=True` must preserve user `requires_grad` and reject detach/disk conflicts.
8. Portable I/O must reject unsafe paths/symlinks and unsupported tensor variants.

## Newer 2.x Subsystems
- `_io/` and `io/`: portable save/load, `.tlspec` manifests, lazy activation refs, rehydration.
- `intervention/`: Bundle, sites/selectors, hooks, helpers, replay/rerun/fork/save.
- `fastlog/`: sparse `Recording` path, predicate normalization, RAM/disk storage.
- `bridge/`: lazy optional adapters for external tools.
- `compat/`: migration helpers and `compat.report(model, x)`.
- `callbacks/`: Lightning callback integration.
- `partial/`: partial log wrapper for failed captures.
- `report/`, `stats/`, `viz/`, `experimental/`: explanation, aggregation, convenience visuals,
  and unstable APIs.

## Conditional Branch Attribution
- Step 5 builds AST file indexes, classifies terminal bools, materializes dense
  `conditional_events`, runs backward flood, attributes forward arm edges, then derives
  legacy THEN/ELIF/ELSE views.
- Primary structures are `ModelLog.conditional_events`, `conditional_arm_edges`,
  `conditional_edge_passes`, and `cond_branch_children_by_cond`.
- Graphviz renders IF/THEN/ELIF/ELSE labels; ELK and dagua conditional support remains more
  limited than Graphviz.

## Release Safety
Semantic-release uses `scripts/no_major_parser.py` plus commit hooks to block accidental
major bumps. For docs-only work use `docs(...)` or `chore(...)` and never add major-bump
markers to commit messages, PR text, or committed docs.
