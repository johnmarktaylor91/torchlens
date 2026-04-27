# torchlens/ — Core Package

## What This Is
TorchLens extracts activations from PyTorch models by permanently wrapping all PyTorch
functions at import time with toggle-gated wrappers, then enabling the toggle during
each forward pass. Entry point: `log_forward_pass()` in `user_funcs.py`.

## Architecture Overview

```
import torchlens  (ONE TIME)
  |- decorate_all_once()      — wraps ~2000 torch functions
  |- patch_detached_references() — patches `from torch import cos` style imports
  |
log_forward_pass(model, input)
  |- decoration/model_prep.py  — prepare model (once + per-session)
  |- capture/trace.py          — run forward pass with logging enabled
  |- capture/output_tensors.py — log each tensor operation
  |- postprocess/              — 18-step pipeline (graph, loops, labels, modules)
  +- Returns ModelLog with all logged data

tl.fastlog.record(model, input, keep_op=...)
  |- decoration/model_prep.py  — prepare model without mutating trainability
  |- capture/output_tensors.py — build lightweight op RecordContext values
  |- decoration/model_prep.py  — build module enter/exit RecordContext values
  |- fastlog/_orchestrator.py  — evaluate predicates and append selected records
  +- Returns Recording with sparse RAM and/or disk-backed activation records
```

Two-pass strategy: when `layers_to_save` is a specific list, Pass 1 runs exhaustive
(metadata only), Pass 2 runs fast (saves only requested layers).
Predicate strategy: `tl.fastlog` runs one or more explicit `Recorder.log()` passes and
retains only events selected by predicates. It does not build a faithful `ModelLog`.

Training-target mode: `train_mode=True` is the unified opt-in for losses built from
saved activations. It is supported by `log_forward_pass()`,
`ModelLog.save_new_activations()`, and `tl.fastlog`. Slow/replay training capture is
RAM-only: disk activation saves are rejected before the forward. Fastlog may mirror to
disk only when an attached RAM payload is retained. Training mode keeps saved floating
tensors graph-connected, preserves user `requires_grad` settings, and rejects
inference-mode or compiled-model contexts that cannot provide the expected autograd
semantics.

## Key Concepts

### Toggle Architecture
- **Permanent decoration**: All torch functions wrapped once at import time. Wrappers
  check `_state._logging_enabled` (single bool) — when False, one branch check, negligible overhead.
- **Context managers**: `active_logging(model_log)` enables logging for a forward pass;
  `pause_logging()` temporarily disables (used internally to prevent recursive logging).
- **sys.modules crawl**: `patch_detached_references()` patches `from torch import cos`
  style imports across all loaded modules.

### Data Flow
1. Decoration layer intercepts every torch function call
2. Barcode nesting detection identifies bottom-level operations
3. `capture/` builds LayerPassLog entries with ~85+ fields per operation
4. `postprocess/` cleans graph, detects loops, assigns human-readable labels
5. `finalization` builds LayerLog aggregates, ModuleLog, ParamLog
6. ModelLog returned to user with full graph + metadata

## Subpackages
- **[capture/](capture/)** — Real-time tensor operation logging during forward pass (7 files)
- **[data_classes/](data_classes/)** — ModelLog, LayerLog, LayerPassLog, ModuleLog, ParamLog, etc. (10 files)
- **[decoration/](decoration/)** — One-time torch function wrapping + model preparation (2 files)
- **[fastlog/](fastlog/)** — Predicate-based sparse activation recording with RAM/disk storage (16 files)
- **[postprocess/](postprocess/)** — 18-step pipeline: graph cleanup, loop detection, labeling (6 files)
- **[utils/](utils/)** — Arg handling, tensor ops, RNG, hashing, display helpers (7 files)
- **[validation/](validation/)** — Forward replay, perturbation checks, metadata invariants (3 files)
- **[visualization/](visualization/)** — Graphviz + ELK + dagua rendering (3 files)
