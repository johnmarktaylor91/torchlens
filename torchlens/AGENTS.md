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

## Conditional Branch Attribution
- Postprocess Step 5 is a six-phase pipeline: 5a build AST file indexes, 5b classify terminal bools to structural keys, 5c materialize `conditional_events` and assign dense `cond_id`s, 5d run the backward IF flood, 5e attribute ops and forward edges to branch arms, 5f derive legacy views and populate `conditional_edge_passes`.
- Primary structures are `ModelLog.conditional_events`, `ModelLog.conditional_arm_edges`, `ModelLog.conditional_edge_passes`, and `cond_branch_children_by_cond` on `LayerPassLog` / `LayerLog`. Legacy `conditional_then_edges` / `conditional_elif_edges` / `conditional_else_edges` and `cond_branch_then_children` / `cond_branch_elif_children` / `cond_branch_else_children` are derived views.
- A single forward edge may enter multiple arms simultaneously; rolled edges may map to different arms on different passes. Use `conditional_edge_passes` for accurate rolled-mode labels.
- User-visible effects: Graphviz renders `THEN` / `ELIF` / `ELSE` edge labels, and ternary (`IfExp`) attribution is first-class on Python 3.11+ via `col_offset`. Python 3.9/3.10 fail closed for same-line ternary ambiguity.
- Deferred in this sprint: dagua bridge wiring, ELK conditional rendering, and while-loop body attribution. Reference: `.project-context/plans/if-else-attribution/plan.md` (v7).
