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
```

Two-pass strategy: when `layers_to_save` is a specific list, Pass 1 runs exhaustive
(metadata only), Pass 2 runs fast (saves only requested layers).

## Files in This Directory

| File | ~Lines | Purpose |
|------|--------|---------|
| `__init__.py` | 26 | Public API exports + import-time decoration trigger |
| `_state.py` | 208 | Global toggle, session state, context managers, WeakSet, pre-computed mappings. **No imports from other torchlens modules** (prevents circular deps) |
| `constants.py` | 645 | 7 FIELD_ORDER tuples, function discovery (~90 IGNORED_FUNCS, ORIG_TORCH_FUNCS) |
| `user_funcs.py` | 664 | User-facing API: `log_forward_pass`, `validate_forward_pass`, `show_model_graph`, `get_model_metadata` |

## Key Concepts

### Toggle Architecture
- **Permanent decoration**: All torch functions wrapped once at import time. Wrappers
  check `_state._logging_enabled` (single bool) — when False, one branch check, negligible overhead.
- **Context managers**: `active_logging(model_log)` enables logging for a forward pass;
  `pause_logging()` temporarily disables (used internally to prevent recursive logging).
- **sys.modules crawl**: `patch_detached_references()` patches `from torch import cos`
  style imports across all loaded modules.

### Attribute Conventions
- `tl_` prefix on tensor/module attributes during logging
- **Permanent attrs** (survive sessions): `tl_module_address`, `tl_module_type`
- **Session attrs** (cleaned per-call): `tl_source_model_log`, `tl_module_pass_num`, etc.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed

### Constants as Ordering Spec
FIELD_ORDER tuples define canonical field sets. `_trim_and_reorder` reorders fields to
match but preserves all fields (no stripping). When adding new fields, update both the
class definition and the corresponding FIELD_ORDER in constants.py.

## Subpackages
- **[capture/](capture/CLAUDE.md)** — Real-time tensor operation logging during forward pass (7 files: trace, output_tensors, source_tensors, tensor_tracking, arg_positions, salient_args, flops)
- **[data_classes/](data_classes/CLAUDE.md)** — ModelLog, LayerLog, LayerPassLog, ModuleLog, ParamLog, etc. (10 files)
- **[decoration/](decoration/CLAUDE.md)** — One-time torch function wrapping + model preparation (2 files)
- **[postprocess/](postprocess/CLAUDE.md)** — 18-step pipeline: graph cleanup, loop detection, labeling (6 files)
- **[utils/](utils/CLAUDE.md)** — Arg handling, tensor ops, RNG, hashing, display helpers (7 files)
- **[validation/](validation/CLAUDE.md)** — Forward replay, perturbation checks, metadata invariants (3 files)
- **[visualization/](visualization/CLAUDE.md)** — Graphviz + ELK-based computational graph rendering (2 files)

## Critical Invariants
1. `_state.py` must never import other torchlens modules
2. RNG state capture/restore must happen BEFORE `active_logging()` context
3. `pause_logging()` must wrap any internal torch ops during logging (safe_copy, activation_postfunc, get_tensor_memory_amount)
4. Decorated wrappers are permanent — never undecorated
5. Field-order constants and class definitions must stay in sync
6. Step 6 module suffix mutation makes `_rebuild_pass_assignments` (Step 8) NECESSARY — not just defensive
