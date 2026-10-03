## Files in This Directory

The directory is the `torchlens/` package (this text moved out of `torchlens/AGENTS.md`).

KEY files only, NOT exhaustive (the package holds ~48 top-level modules; notable
omissions include `runnable.py` — home of the 7 frozen runnable enums cited
below — `captured_run.py`, `hash.py`, `facets.py`, `_capture_fingerprint.py`,
and the 16-file `_runnable_*` execution seam). `ls torchlens/*.py` is the
authority.

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports, lazy facade, `pluck`, `extract`, `extract_dataset` routing |
| `_state.py` | Global toggle, active log, decoration maps, prepared model registry; no torchlens imports except the sanctioned `errors._base` leaf |
| `_trace_state.py` | Runtime state enum surfaced through `torchlens.io` |
| `_errors.py`, `_robustness.py`, `_training_validation.py` | Legacy/public error and compatibility helpers |
| `_literals.py` | Shared literal types for options and modes |
| `_source_links.py` | Source-link helpers used by reports/visualization |
| `constants.py` | FIELD_ORDER tuples and decorated torch function discovery |
| `options.py` | Immutable grouped options and flat-argument merge helpers |
| `observers.py` | `tap`, `span`, and active span state |
| `types.py` | Moved public type aliases not kept in top-level `__all__` |
| `user_funcs.py` | Main capture, summary, visualization, validation, and bundle graph entry points |
