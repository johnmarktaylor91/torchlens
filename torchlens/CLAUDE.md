# torchlens/ - Core Package

## What This Is
TorchLens extracts activations and metadata from PyTorch eager models. `import
torchlens` exposes the public API and compatibility shims, but torch wrapping is lazy:
the first capture prepares the model and calls `wrap_torch()` from `decoration/`.

## Architecture Overview

```
import torchlens
  |- exposes 40 top-level public names in __all__
  |- imports submodule namespaces: fastlog, bridge, compat, export, options, report, stats, viz
  |
log_forward_pass(model, input)
  |- decoration/model_prep.py  - ensure torch is wrapped, prepare modules/buffers/params
  |- capture/trace.py          - run forward pass with active logging
  |- capture/output_tensors.py - build LayerPassLog records
  |- postprocess/              - current 20-step graph cleanup/finalization pipeline
  +- returns ModelLog

tl.fastlog.record(model, input, keep_op=...)
  |- uses the same wrapper hot path
  |- stores predicate-selected RecordContext/ActivationRecord values
  +- returns Recording, not ModelLog
```

Selective `layers_to_save` uses the two-pass strategy: Pass 1 exhaustive metadata, Pass
2 fast activation save. Fastlog uses explicit predicate passes instead and does not
try to build a faithful full graph.

`train_mode=True` is the public opt-in for losses built from saved activations. It keeps
floating tensors graph-connected, preserves user `requires_grad`, and rejects incompatible
detaching or disk-only activation storage.

## Top-Level Modules

| Path | Purpose |
|------|---------|
| `__init__.py` | Top-level API, 40-name `__all__`, deprecation shims, `peek`/`extract` helpers |
| `_state.py` | Global logging toggle, active log, decoration maps, prepared-model registry; no torchlens imports |
| `_run_state.py` | Small runtime state enum exposed through `torchlens.io` |
| `_errors.py`, `errors/` | Public and legacy exception classes |
| `_io/`, `io/` | Portable `.tlspec` save/load, manifest, lazy tensor refs, public I/O helpers |
| `options.py` | Capture, save, visualization, replay, intervention, and streaming option groups |
| `observers.py` | `tap()` and `record_span()` observer helpers |
| `report/` | `report.explain(log)` and capture-time scalar logging |
| `stats/` | Streaming stats and `aggregate()` over dataloaders |
| `types.py`, `accessors/` | Moved type/accessor aliases for non-top-level public names |

## Subpackages
- `capture/` - real-time forward and backward operation logging.
- `data_classes/` - `ModelLog`, `LayerLog`, `LayerPassLog`, module/param/buffer/grad logs.
- `decoration/` - lazy torch function wrapping, explicit wrap/unwrap, module prep.
- `fastlog/` - sparse predicate recording with RAM/disk storage and recovery.
- `postprocess/` - graph cleanup, conditionals, loop detection, labeling, finalization.
- `validation/` - forward replay, backward validation, metadata invariants, `.tlspec` schema checks.
- `visualization/` - Graphviz rendering, ELK layout, NodeSpec, themes, overlays, bundle diff.
- `intervention/` - selectors, sites, hooks, helpers, Bundle, fork/replay/rerun/save.
- `multi_trace/` - internal bundle supergraph and node diff support.
- `bridge/`, `compat/`, `callbacks/` - optional integrations and migration facades.
- `viewer/`, `paper/`, `notebook/`, `llm/`, `neuro/` - appliance package boundaries gated by extras.

## Key Concepts

### Toggle Architecture
- Lazy wrapping: `wrap_torch()` installs wrappers on first capture or explicit call.
- Persistent wrappers: after wrapping, calls only pay a `_state._logging_enabled` check when
  logging is off.
- `active_logging(model_log)` enables logging during the forward; `pause_logging()` protects
  internal TorchLens tensor ops from recursive capture.
- `patch_detached_references()` patches `from torch import cos` style references in loaded modules.

### Data Flow
1. Decoration intercepts torch function calls.
2. Barcode nesting detection identifies bottom-level operations.
3. `capture/` builds raw `LayerPassLog` entries.
4. `postprocess/` removes orphans, marks conditionals, detects loops, labels nodes, builds logs.
5. `ModelLog` exposes lookup, visualization, validation, save/load, intervention, and summary helpers.

### Portable Artifacts
`tl.save()` and `tl.load()` route through `_io/bundle.py`. Unified `.tlspec` directories have
`manifest.json` plus safetensors blobs; public schema validation lives in `validation/__init__.py`.
Intervention specs can be saved at audit, executable-with-callables, or portable levels.

### Appliances
The five appliance subfolders are part of the 2.x package layout. `viewer`, `paper`, and
`llm` are empty stubs with docstring intent. `notebook` and `neuro` currently enforce their
extras by importing required dependencies, but export no public objects yet.
