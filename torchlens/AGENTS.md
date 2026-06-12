# torchlens/ - Implementation Guide

## Files in This Directory

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports, moved-name deprecation shims, `peek`, `extract`, `batched_extract`, validation aliases |
| `_state.py` | Global toggle, active log, decoration maps, prepared model registry; must not import torchlens modules |
| `_trace_state.py` | Runtime state enum surfaced through `torchlens.io` |
| `_errors.py`, `_robustness.py`, `_training_validation.py` | Legacy/public error and compatibility helpers |
| `_literals.py` | Shared literal types for options and modes |
| `_source_links.py` | Source-link helpers used by reports/visualization |
| `constants.py` | FIELD_ORDER tuples and decorated torch function discovery |
| `options.py` | Immutable grouped options and flat-argument merge helpers |
| `observers.py` | `tap`, `record_span`, and active span state |
| `types.py` | Moved public type aliases not kept in top-level `__all__` |
| `user_funcs.py` | Main capture, summary, visualization, validation, and bundle graph entry points |

## Attribute Conventions
- TorchLens metadata attached to user/model objects lives under `obj._tl`.
- Permanent module metadata uses `_tl.address` and `_tl.module_type`.
- Session tensor/parameter metadata is cleaned per capture; callable wrapper markers also live
  under `_tl`.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state.

## Public Surface
`torchlens.__all__` is intentionally small and currently has 60 names. New user-facing
objects should usually live under submodules (`torchlens.io`, `torchlens.options`,
`torchlens.bridge`, `torchlens.errors`, etc.) with moved-name shims only when compatibility
requires them.

Unified capture examples:

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
    save=tl.func("linear"),
    intervene=tl.when(tl.func("linear"), tl.scale(0.5)),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
```

`record(keep_op=...)` and `record(keep_module=...)` are deprecated compatibility aliases for
`record(save=...)`. `layers_to_save=[...]` still exists as the final-label two-pass path; an
unqualified recurrent layer label saves all passes, while `"label:2"` saves only pass 2.

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
5. Module suffixes are appended to `equivalence_class` at op creation before loop detection.
6. `postprocess_fast()` must not call `_build_module_logs()`.
7. `backward_ready=True` must preserve user `requires_grad` and reject detach/disk conflicts.
8. Portable I/O must reject unsafe paths/symlinks and unsupported tensor variants.

## Newer 2.x Subsystems
- `_io/` and `io/`: portable save/load, `.tlspec` manifests, lazy out refs, rehydration.
- `intervention/`: Bundle, sites/selectors, hooks, helpers, replay/rerun/fork/save.
- `fastlog/`: sparse `Recording` path, predicate normalization, RAM/disk storage.
- `bridge/`: lazy optional adapters for external tools.
- `compat/`: migration helpers and `compat.report(model, x)`.
- `callbacks/`: Lightning callback integration.
- `partial/`: partial log wrapper for failed captures.
- `debug/`, `report/`, `stats/`, `viz/`, `experimental/`: diagnostics, explanation,
  aggregation, convenience visuals, and unstable APIs.

## Conditional Branch Attribution
- Step 5 builds AST file indexes, classifies terminal bools, materializes dense
  `conditional_records`, runs backward flood, attributes forward arm edges, then derives
  legacy THEN/ELIF/ELSE views.
- Primary structures are `Trace.conditional_records`, `conditional_arm_entry_edges`,
  `conditional_edge_call_indices`, and `conditional_arm_children`.
- Graphviz renders IF/THEN/ELIF/ELSE labels; dagua conditional support remains more
  limited than Graphviz.

## Release Safety
Semantic-release uses `scripts/no_major_parser.py` plus commit hooks to block accidental
major bumps. For docs-only work use `docs(...)` or `chore(...)` and never add major-bump
markers to commit messages, PR text, or committed docs.
