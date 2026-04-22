# TorchLens Gotchas & Edge Cases

<!-- Agents: READ THIS before making changes. Append as you discover things. -->
<!-- Format: - [MODULE] GOTCHA: <description> -->

- [_state] `_state.py` must NEVER import other torchlens modules — everything imports from it
- [decoration] `__wrapped__` removed from built-in function wrappers to prevent `inspect.unwrap` failures with `torch.jit.script`
- [decoration] Fast-path module decorator skips `_handle_module_entry` entirely — counter alignment must be replicated manually at model_prep.py:506-511
- [decoration] `_module_class_metadata_cache` cleared at session start to avoid stale `inspect.getsourcelines` data
- [capture] `safe_copy` strips `tl_tensor_label_raw` from clone — critical for in-place op detection
- [capture] `pause_logging()` must wrap `activation_postfunc`, `get_tensor_memory_amount()`, and any internal torch ops to prevent infinite recursive logging
- [capture] `arg_positions` dynamic cache never cleared on torch version upgrades — could serve stale specs
- [capture] Buffer guard at `torch_funcs.py:108` must check `not hasattr(t, "tl_tensor_label_raw")` to prevent duplicate buffer entries
- [data_classes] Adding new fields requires updating BOTH the class definition AND `constants.py` FIELD_ORDER
- [data_classes] Python property/__getattr__ trap: If `@property` raises `AttributeError`, Python falls through to `__getattr__`. Use `ValueError` instead.
- [data_classes] `copy()` on LayerPassLog: shallow-copies 8 specific fields, deep-copies rest — safe only because downstream uses assignment not mutation
- [data_classes] `activation` for non-getitem output layers is a DIRECT REFERENCE to parent's saved data — mutation affects both
- [data_classes] `equivalent_operations` per-LayerPassLog holds direct reference to ModelLog-level sets; becomes stale after rename step 11
- [data_classes] `FuncCallLocation._frame_func_obj` leaks if lazy properties never accessed
- [postprocess] Step 6 appends module addresses to `operation_equivalence_type` — makes `_rebuild_pass_assignments` (Step 8) NECESSARY, not defensive
- [postprocess] `_build_layer_logs` multi-pass merge: only 3 fields merged (has_input_ancestor OR, io_role char-merge, is_leaf_module_output OR). All other 78 fields use first-pass values.
- [postprocess] `_build_module_logs` must NOT be called in `postprocess_fast` — `_module_build_data` isn't populated in fast mode
- [postprocess] `_pass_finished` not reset between passes — intentional for fast-path lookups
- [validation] Validation requires `save_function_args=True` — without it, replay can't reconstruct inputs
- [validation] `MAX_FLOATING_POINT_TOLERANCE = 3e-6` is too tight for bfloat16 (epsilon ~7.8e-3)
- [validation] `tensor_nanequal()` crashes on quantized tensors (`.isinf()` raises AttributeError)
- [visualization] Graphviz `render()` creates a DOT source file alongside the output (e.g., `.pdf` + source file)
- [visualization] `parent_node.layer_label in arg_label` substring check is fragile — relies on `type_num` uniqueness
- [visualization] ELK stress layout allocates O(n^2) memory — NEVER use for >100k nodes
- [utils] `get_tensor_memory_amount()` MUST use `pause_logging()` — `nelement()`/`element_size()` are decorated
- [utils] Clean function imports (`clean_clone`, `clean_to`) must happen at module load time, before `decorate_all_once()`
- [decoration] NEVER mutate a namespace while introspecting it -- `inspect.signature()` evaluates annotations lazily (PEP 649, Python 3.14+), so decorated names (e.g. `Tensor.bool`) can shadow builtins in annotation resolution. Collect all introspection data in a separate pass BEFORE decoration begins (#138)
- [decoration] `decorate_all_once()` idempotency guard must use `_is_decorated` (completion flag), NOT `_orig_to_decorated` (non-empty dict). Partial failure populates the dict without completing decoration; dict-based guard prevents retry and locks in incomplete state (#138)
- [decoration] `get_func_argnames` must catch both `ValueError` AND `TypeError` from `inspect.signature()` -- TypeError occurs on Python 3.14+ when deferred annotation evaluation resolves class-level names to wrapper functions instead of builtins (#138)
- [data_classes] Under `save_source_context=False`, `FuncCallLocation` must clear `_frame_func_obj` at construction time; retaining nested/local function objects until lazy load makes pickling fragile (D17)
- [postprocess] Ternary (`IfExp`) attribution is same-line for both arms, so `col_offset` is load-bearing; Python 3.9/3.10 lack that signal and must fail closed on ambiguous ternaries
- [postprocess] Python AST represents `elif` as nested `If` in `orelse`; always flatten the chain before materializing `ConditionalEvent`s or arm labels drift
- [postprocess] A single forward edge can legitimately enter multiple conditional arms at once (for example parameter/buffer deps inside nested `if` blocks); do not collapse arm-entry edges to one label
- [visualization] Rolled edges can belong to different arms on different passes; renderer labels must consult `conditional_edge_passes`, not only pass-stripped edge membership
