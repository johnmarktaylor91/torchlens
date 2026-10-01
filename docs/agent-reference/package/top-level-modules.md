## Top-Level Modules

| Path | Purpose |
|------|---------|
| `__init__.py` | Top-level API, 116-name `__all__`, lazy facade, `pluck`/`extract` helpers |
| `_state.py` | Global logging toggle, active log, decoration maps, prepared-model registry; no torchlens imports except the sanctioned `errors._base` leaf (a RUNTIME base-class import, cycle-safe; only its TYPE_CHECKING block is typing-only) |
| `_trace_state.py` | Small runtime state enum exposed through `torchlens.io` |
| `_errors.py`, `errors/` | Public and legacy exception classes |
| `_io/`, `io/` | Portable `.tlspec` save/load, manifest, lazy tensor refs, public I/O helpers |
| `options.py` | Capture, save, visualization, replay, intervention, and streaming option groups |
| `observers.py` | `tap()` and `span()` observer helpers |
| `report/` | `report.explain(log)` and capture-time scalar logging |
| `stats/` | Streaming stats and `aggregate()` over dataloaders |
| `types.py`, `accessors/` | Moved type/accessor aliases for non-top-level public names |
