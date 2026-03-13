# torchlens/ — Implementation Guide

## Files in This Directory

| File | ~Lines | Purpose |
|------|--------|---------|
| `__init__.py` | 26 | Public API exports + import-time decoration trigger |
| `_state.py` | 208 | Global toggle, session state, context managers, WeakSet, pre-computed mappings. **No imports from other torchlens modules** (prevents circular deps) |
| `constants.py` | 645 | 7 FIELD_ORDER tuples, function discovery (~90 IGNORED_FUNCS, ORIG_TORCH_FUNCS) |
| `user_funcs.py` | 664 | User-facing API: `log_forward_pass`, `validate_forward_pass`, `show_model_graph`, `get_model_metadata` |

## Attribute Conventions
- `tl_` prefix on tensor/module attributes during logging
- **Permanent attrs** (survive sessions): `tl_module_address`, `tl_module_type`
- **Session attrs** (cleaned per-call): `tl_source_model_log`, `tl_module_pass_num`, etc.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed

## Constants as Ordering Spec
FIELD_ORDER tuples define canonical field sets. `_trim_and_reorder` reorders fields to
match but preserves all fields (no stripping). When adding new fields, update both the
class definition and the corresponding FIELD_ORDER in constants.py.

## Critical Invariants
1. `_state.py` must never import other torchlens modules
2. RNG state capture/restore must happen BEFORE `active_logging()` context
3. `pause_logging()` must wrap any internal torch ops during logging (safe_copy, activation_postfunc, get_tensor_memory_amount)
4. Decorated wrappers are permanent by default, but advanced users may call
   `undecorate_all_globally()` / `redecorate_all_globally()` as an explicit
   override when they need a clean PyTorch environment.
5. Field-order constants and class definitions must stay in sync
6. Step 6 module suffix mutation makes `_rebuild_pass_assignments` (Step 8) NECESSARY — not just defensive
