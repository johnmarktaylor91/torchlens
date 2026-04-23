# TorchLens I/O Audit

Date: 2026-04-23

Scope notes:
- Read-only audit of `torchlens/` plus relevant tests.
- Visualization/rendering code was intentionally skipped per task.
- All code-behavior claims below are backed by file:line citations. Runtime pickle experiments are called out explicitly as experiments, not code citations.

## TOC
- [1. Inventory](#1-inventory)
- [2. Gaps vs user goals](#2-gaps-vs-user-goals)
- [3. Risks and tricky spots](#3-risks-and-tricky-spots)
- [4. Recommendations](#4-recommendations)
- [5. Implementation outline](#5-implementation-outline)

## 1. Inventory

### 1.1 Existing serialization/export surfaces

| Location | Signature | Behavior summary |
| --- | --- | --- |
| `torchlens/data_classes/model_log.py:337-345` | `ModelLog.__getstate__(self) -> Dict[str, Any]` | Pickle hook that copies `__dict__`, strips `_module_logs`, `_buffer_accessor`, `_module_build_data`, and normalizes `_raw_layer_type_counter` to a plain `dict`. |
| `torchlens/data_classes/model_log.py:346-365` | `ModelLog.__setstate__(self, state: Dict[str, Any]) -> None` | Pickle restore hook that reinstalls an empty `ModuleAccessor`, reinitializes `_module_build_data`, and reconnects `LayerLog`/`LayerPassLog` backrefs to the owning `ModelLog`; it does not rebuild module/buffer accessors. |
| `torchlens/data_classes/layer_pass_log.py:390-398` | `LayerPassLog.__getstate__(self) -> Dict` / `__setstate__(self, state: Dict) -> None` | Pickle hooks that only strip the weakref-backed `_source_model_log_ref` and then restore raw state. |
| `torchlens/data_classes/layer_log.py:216-224` | `LayerLog.__getstate__(self) -> Dict` / `__setstate__(self, state: Dict) -> None` | Pickle hooks that only strip the weakref-backed `_source_model_log_ref` and then restore raw state. |
| `torchlens/data_classes/layer_pass_log.py:417-453` | `LayerPassLog.copy(self)` | Selective-depth clone helper used internally for synthetic output nodes; this is not Python’s `__copy__` hook. |
| `torchlens/data_classes/interface.py:421-541` | `to_pandas(self) -> pd.DataFrame` | Bound onto `ModelLog` as `model_log.to_pandas()`; exports one row per `LayerPassLog` with pass-level metadata columns. |
| `torchlens/data_classes/layer_log.py:665-687` | `LayerAccessor.to_pandas(self) -> pd.DataFrame` | Exports one row per aggregate `LayerLog` with a smaller layer-level summary. |
| `torchlens/data_classes/module_log.py:399-415` | `ModuleLog.to_pandas(self) -> pd.DataFrame` | Exports one row per layer inside one module, not one row per module pass. |
| `torchlens/data_classes/module_log.py:496-510` | `ModuleAccessor.to_pandas(self) -> pd.DataFrame` | Exports one row per `ModuleLog` summary across the model. |
| `torchlens/capture/trace.py:55-140` | `save_new_activations(self, model, input_args, input_kwargs=None, layers_to_save="all", random_seed=None)` | Re-runs the model in fast mode and refreshes saved activations in memory only; there is no disk output. |
| `torchlens/user_funcs.py:153-382` | `log_forward_pass(...) -> ModelLog` | Main public API for in-memory activation capture; supports `layers_to_save="all"`, `"none"`, or selective layer keys, and uses a two-pass strategy for selective saves. |
| `torchlens/user_funcs.py:385-411` | `get_model_metadata(...) -> ModelLog` | Public wrapper around `log_forward_pass(..., layers_to_save=None)` that returns metadata with no saved activations. |
| `torchlens/user_funcs.py:508-632` | `validate_forward_pass(...) -> bool` / `validate_saved_activations(...) -> bool` | Public in-memory replay/validation path that depends on saved activations and function metadata; not a file format. |
| `torchlens/user_funcs.py:635-697` | `validate_batch_of_models_and_inputs(...) -> pd.DataFrame` | Reads an existing CSV with `pd.read_csv()` and appends validation results back to CSV with `to_csv()`. |

What is not present:
- `ModelLog` only binds `to_pandas`, `save_new_activations`, validation, rendering, cleanup, and related helpers; it does not bind `save`, `load`, `to_csv`, `to_json`, `to_parquet`, or `to_dict` methods (`torchlens/data_classes/model_log.py:545-558`).
- The package public API exports `log_forward_pass`, `show_model_graph`, `get_model_metadata`, validation helpers, decoration lifecycle helpers, and data classes, but no `load()` or serialization helper (`torchlens/__init__.py:14-34`).
- `ParamAccessor` and `BufferAccessor` expose indexing/repr only; they do not define `to_pandas`, `to_csv`, `to_json`, or `to_parquet` (`torchlens/data_classes/param_log.py:191-248`, `torchlens/data_classes/buffer_log.py:72-132`).
- `ModulePassLog` is a container with `__repr__`/`__len__` only; there is no pass-level export surface for it (`torchlens/data_classes/module_log.py:36-103`).

### 1.2 Public API surface relevant to save/load/export

| Public API | Relevance to I/O | Notes |
| --- | --- | --- |
| `log_forward_pass` | In-memory activation/metadata capture | Saves activations into `LayerPassLog.activation` according to `layers_to_save`; no file output path exists in this function (`torchlens/user_funcs.py:201-215`, `torchlens/user_funcs.py:289-346`). |
| `get_model_metadata` | Metadata-only capture | Convenience wrapper for “no activations saved” (`torchlens/user_funcs.py:390-410`). |
| `validate_forward_pass` | In-memory replay using saved activations | Forces `save_function_args=True` and `save_rng_states=True`, then replays from saved activations (`torchlens/user_funcs.py:520-532`, `torchlens/user_funcs.py:587-612`). |
| `validate_batch_of_models_and_inputs` | Actual file I/O today | Only CSV read/write path in the audited scope (`torchlens/user_funcs.py:657-695`). |

### 1.3 Data-class contents

The canonical public field sets live in `torchlens/constants.py`:
- `MODEL_LOG_FIELD_ORDER` (`torchlens/constants.py:33-119`)
- `LAYER_PASS_LOG_FIELD_ORDER` (`torchlens/constants.py:121-261`)
- `LAYER_LOG_FIELD_ORDER` (`torchlens/constants.py:266-339`)
- `PARAM_LOG_FIELD_ORDER` (`torchlens/constants.py:361-382`)
- `MODULE_LOG_FIELD_ORDER` (`torchlens/constants.py:402-447`)

Important caveat: `ModelLog` also carries internal/transient attributes that are not in `MODEL_LOG_FIELD_ORDER`; `cleanup()` explicitly deletes those separately (`torchlens/data_classes/model_log.py:166-291`, `torchlens/data_classes/cleanup.py:58-82`).

#### ModelLog

Class definition: `torchlens/data_classes/model_log.py:113-292`
Canonical field list: `torchlens/constants.py:33-119`

| Attributes | Kind | Notes for pickle/export design |
| --- | --- | --- |
| `model_name`, `logging_mode`, `output_device` | primitive strings | Core identity/config. |
| `_pass_finished`, `_all_layers_logged`, `_all_layers_saved`, `keep_unsaved_layers`, `detach_saved_tensors`, `save_function_args`, `save_gradients`, `save_source_context`, `save_rng_states`, `detect_loops`, `verbose`, `has_gradients`, `mark_input_output_distances` | primitive bool flags | Serialization-safe, but they change behavior on reload (`torchlens/data_classes/model_log.py:172-191`). |
| `current_function_call_barcode`, `random_seed_used` | optional primitive | May be `None` before/during logging (`torchlens/data_classes/model_log.py:179-182`). |
| `activation_postfunc` | callable or `None` | Can make saved activations non-tensor objects because `save_tensor_data()` stores its return value directly (`torchlens/data_classes/model_log.py:179`, `torchlens/data_classes/layer_pass_log.py:488-490`). |
| `layer_list` | `list[LayerPassLog|BufferLog]` | Main ordered payload for pass-level export (`torchlens/data_classes/model_log.py:197-198`). |
| `layer_dict_main_keys` | `dict[str, LayerPassLog|BufferLog]` | Primary final-label lookup (`torchlens/data_classes/model_log.py:198-201`). |
| `layer_dict_all_keys` | `dict[mixed lookup key, LayerPassLog|BufferLog]` | Contains all lookup keys, not just canonical labels (`torchlens/data_classes/model_log.py:199-201`, `torchlens/data_classes/interface.py:77-123`). |
| `layer_logs` | `dict[str, LayerLog]` | Aggregate no-pass layer view (`torchlens/data_classes/model_log.py:202`). |
| `layer_labels`, `layer_labels_w_pass`, `layer_labels_no_pass` | `list[str]` | Final labels in various namespaces (`torchlens/data_classes/model_log.py:203-206`). |
| `layer_num_passes`, `buffer_num_passes` | `dict[str, int]`-like | Pass counts per layer/buffer (`torchlens/data_classes/model_log.py:206`, `torchlens/data_classes/model_log.py:229`). |
| `_raw_layer_dict`, `_raw_layer_labels_list`, `_raw_to_final_layer_labels`, `_final_to_raw_layer_labels` | raw-label dict/list structures | Raw-to-final mapping is critical to fast-path re-save and any future loader (`torchlens/data_classes/model_log.py:208-223`, `torchlens/capture/trace.py:124-140`). |
| `_layer_nums_to_save` | `list[int]` or sentinel `"all"` | Selective-save resolution output, not a stable user object (`torchlens/data_classes/model_log.py:210`, `torchlens/capture/trace.py:161-192`). |
| `_layer_counter`, `num_operations`, `total_activation_memory`, `num_tensors_saved`, `saved_activation_memory`, `total_param_*`, `pass_start_time`, `pass_end_time`, `time_*` | numeric primitives | Straightforward to tabularize. |
| `_raw_layer_type_counter`, `_lookup_keys_to_layer_num_dict`, `_layer_num_to_lookup_keys_dict`, `layers_with_params`, `equivalent_operations`, `conditional_arm_edges`, `conditional_edge_passes` | dicts of labels/counts/sets/lists | Mixed nested containers; not table-ready without normalization (`torchlens/data_classes/model_log.py:213-250`). |
| `_unsaved_layers_lookup_keys` | `set[str]` | Internal lookup helper (`torchlens/data_classes/model_log.py:214-217`). |
| `input_layers`, `output_layers`, `buffer_layers`, `internally_initialized_layers`, `_layers_where_internal_branches_merge_with_input`, `internally_terminated_layers`, `internally_terminated_bool_layers`, `layers_with_saved_activations`, `unlogged_layers`, `layers_with_saved_gradients`, `orphan_layers` | `list[str]` | Label collections for derived summaries and future exports (`torchlens/data_classes/model_log.py:225-246`). |
| `conditional_branch_edges`, `conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges` | edge lists of tuples | Nested, branch-specific graph metadata (`torchlens/data_classes/model_log.py:235-238`). |
| `conditional_events` | `list[ConditionalEvent]` | Structured branch metadata objects (`torchlens/data_classes/model_log.py:94-111`, `torchlens/data_classes/model_log.py:239`). |
| `param_logs` | `ParamAccessor` | Primary parameter accessor (`torchlens/data_classes/model_log.py:257-264`, `torchlens/data_classes/model_log.py:509-513`). |
| Internal extras: `_module_logs`, `_buffer_accessor` | `ModuleAccessor`, `BufferAccessor|None` | Not in `MODEL_LOG_FIELD_ORDER`; both are stripped or left empty on pickle restore (`torchlens/data_classes/model_log.py:279-284`, `torchlens/data_classes/model_log.py:337-365`). |
| Internal extras: `_module_build_data`, `_module_metadata`, `_module_forward_args`, `_param_logs_by_module` | transient dicts | Used to build `ModuleLog`/`BufferAccessor`, then cleared or rebuilt (`torchlens/data_classes/model_log.py:275-284`, `torchlens/postprocess/finalization.py:518-695`). |
| Internal extras: `_optimizer`, `_pre_forward_rng_states`, `_saved_gradients_set`, `_mod_pass_num`, `_mod_pass_labels`, `_mod_entered`, `_mod_exited` | transient refs/counters/sets | Session-scoped state that complicates raw pickling and is not a clean archival schema (`torchlens/data_classes/model_log.py:166`, `torchlens/data_classes/model_log.py:246`, `torchlens/data_classes/model_log.py:270-276`, `torchlens/capture/trace.py:425-428`). |

#### LayerPassLog

Class definition: `torchlens/data_classes/layer_pass_log.py:68-273`
Canonical field list: `torchlens/constants.py:121-261`

| Attributes | Kind | Notes for pickle/export design |
| --- | --- | --- |
| `layer_label`, `tensor_label_raw`, `layer_label_raw`, `layer_label_short`, `layer_label_w_pass`, `layer_label_w_pass_short`, `layer_label_no_pass`, `layer_label_no_pass_short`, `layer_type`, `func_name`, `grad_fn_name`, `buffer_address`, `buffer_parent`, `containing_module`, `leaf_module_pass` | primitive strings or `None` | Mostly tabularizable scalar metadata (`torchlens/data_classes/layer_pass_log.py:111-145`, `torchlens/data_classes/layer_pass_log.py:223-267`). |
| `operation_num`, `creation_order`, `layer_type_num`, `layer_total_num`, `pass_num`, `num_passes`, `tensor_memory`, `grad_memory`, `num_args`, `num_positional_args`, `num_keyword_args`, `num_params_total`, `num_params_trainable`, `num_params_frozen`, `params_memory`, `conditional_branch_depth` | primitive ints | Straightforward scalar columns (`torchlens/data_classes/layer_pass_log.py:111-145`, `torchlens/data_classes/layer_pass_log.py:153-192`, `torchlens/data_classes/layer_pass_log.py:235-267`). |
| `func_time` | primitive float | Per-op timing (`torchlens/data_classes/layer_pass_log.py:166`). |
| `_pass_finished`, `has_saved_activations`, `detach_saved_tensor`, `args_captured`, `has_child_tensor_variations`, `save_gradients`, `has_gradient`, `func_is_inplace`, `is_part_of_iterable_output`, `has_children`, `is_input_layer`, `has_input_ancestor`, `is_output_layer`, `feeds_output`, `is_final_output`, `is_output_ancestor`, `is_buffer_layer`, `is_internally_initialized`, `has_internally_initialized_ancestor`, `is_internally_terminated`, `is_terminal_bool_layer`, `bool_is_branch`, `is_scalar_bool`, `in_cond_branch`, `is_submodule_output`, `is_leaf_module_output` | primitive bools | Good DataFrame columns already exposed heavily by `ModelLog.to_pandas()` (`torchlens/data_classes/layer_pass_log.py:118`, `torchlens/data_classes/layer_pass_log.py:135-160`, `torchlens/data_classes/layer_pass_log.py:178-267`). |
| `source_model_log` | weakref-backed ref to `ModelLog` | Stored internally as `_source_model_log_ref`; pickle strips the weakref only (`torchlens/data_classes/layer_pass_log.py:115-118`, `torchlens/data_classes/layer_pass_log.py:376-398`). |
| `activation` | saved activation object, usually `torch.Tensor`, else `None` | Default path clones the tensor and optionally moves devices, but `activation_postfunc` can replace it with any object (`torchlens/data_classes/layer_pass_log.py:135`, `torchlens/data_classes/layer_pass_log.py:463-502`). |
| `gradient` | `torch.Tensor|None` | Saved as `detach().clone()` snapshot, not a live autograd edge (`torchlens/data_classes/layer_pass_log.py:155`, `torchlens/data_classes/layer_pass_log.py:503-517`). |
| `captured_args`, `captured_kwargs` | copied nested Python objects or `None` | May contain tensors and arbitrary Python containers when `save_function_args=True` (`torchlens/data_classes/layer_pass_log.py:141-142`, `torchlens/data_classes/layer_pass_log.py:495-501`). |
| `tensor_shape`, `grad_shape` | tuple-like or `None` | Shape snapshots (`torchlens/data_classes/layer_pass_log.py:143`, `torchlens/data_classes/layer_pass_log.py:158`). |
| `tensor_dtype`, `grad_dtype` | `torch.dtype|None` | PyTorch type objects, potentially version-sensitive (`torchlens/data_classes/layer_pass_log.py:144`, `torchlens/data_classes/layer_pass_log.py:159`). |
| `activation_postfunc`, `func_applied` | callable or `None` | `func_applied` is the raw producing callable and is a major pickle-fragility point (`torchlens/data_classes/layer_pass_log.py:138`, `torchlens/data_classes/layer_pass_log.py:163`, `torchlens/capture/output_tensors.py:318`). |
| `func_call_stack` | `list[FuncCallLocation]` | Nested lazy source objects; these carry `_frame_func_obj` until source is loaded when `save_source_context=True` (`torchlens/data_classes/layer_pass_log.py:165`, `torchlens/data_classes/func_call_location.py:109-123`, `torchlens/data_classes/func_call_location.py:140-198`). |
| `func_rng_states`, `func_autocast_state` | nested dicts | Opaque RNG/autocast snapshots used by validation replay (`torchlens/data_classes/layer_pass_log.py:169-170`, `torchlens/utils/rng.py:49-111`, `torchlens/validation/core.py:470-477`). |
| `func_argnames` | tuple of strings | Function signature names captured at decoration time (`torchlens/data_classes/layer_pass_log.py:171`). |
| `func_positional_args_non_tensor`, `func_kwargs_non_tensor`, `func_non_tensor_args` | list/dict/list | Non-tensor function config fragments; already semi-tabular but nested (`torchlens/data_classes/layer_pass_log.py:175-177`). |
| `parent_params` | `list[nn.Parameter]` during capture, later cleared to `[]` | Raw parameter refs are dropped during postprocessing to save memory (`torchlens/data_classes/layer_pass_log.py:184`, `torchlens/postprocess/finalization.py:115-118`). |
| `parent_param_barcodes`, `parent_param_shapes`, `recurrent_group`, `parent_layers`, `child_layers`, `input_ancestors`, `output_descendants`, `internally_initialized_parents`, `internally_initialized_ancestors`, `conditional_branch_stack`, `cond_branch_start_children`, `cond_branch_then_children`, `cond_branch_else_children`, `containing_modules`, `modules_entered`, `module_passes_entered`, `modules_exited`, `module_passes_exited`, `module_entry_exit_thread_output` | lists/sets of labels or tuples | Nested graph/module metadata; exportable, but not flat without normalization (`torchlens/data_classes/layer_pass_log.py:183-267`). |
| `parent_param_logs` | `list[ParamLog]` | Direct references to parameter metadata objects (`torchlens/data_classes/layer_pass_log.py:187`). |
| `parent_param_passes`, `parent_layer_arg_locs`, `cond_branch_elif_children`, `cond_branch_children_by_cond`, `modules_entered_argnames`, `module_entry_exit_threads_inputs` | nested dicts | Not directly Parquet/CSV-friendly without schema design (`torchlens/data_classes/layer_pass_log.py:186`, `torchlens/data_classes/layer_pass_log.py:206`, `torchlens/data_classes/layer_pass_log.py:248-250`, `torchlens/data_classes/layer_pass_log.py:256-264`). |
| `equivalent_operations`, `root_ancestors` | `set[str]` | Shared graph groupings; `equivalent_operations` is a direct reference to a `ModelLog`-owned set (`torchlens/data_classes/layer_pass_log.py:194-202`, `torchlens/capture/source_tensors.py:199-200`). |
| `operation_equivalence_type`, `io_role`, `bool_context_kind`, `bool_wrapper_kind`, `scalar_bool_value`, `iterable_output_index`, `buffer_pass`, `min_distance_from_input`, `max_distance_from_input`, `min_distance_from_output`, `max_distance_from_output`, `flops_forward`, `flops_backward` | scalar primitives or `None` | Mostly flat metadata fields already used for summaries. |
| `func_config` | `dict[str, Any]` | Lightweight op config extracted from args/kwargs (`torchlens/data_classes/layer_pass_log.py:266-267`, `torchlens/capture/output_tensors.py:340-349`). |
| Extras: `parent_layer_log` | strong ref to aggregate `LayerLog` | Not in `LAYER_PASS_LOG_FIELD_ORDER`, restored by `ModelLog.__setstate__` when layer logs exist (`torchlens/data_classes/layer_pass_log.py:269-273`, `torchlens/data_classes/model_log.py:356-365`). |

#### LayerLog

Class definition: `torchlens/data_classes/layer_log.py:46-162`
Canonical field list: `torchlens/constants.py:266-339`

| Attributes | Kind | Notes for pickle/export design |
| --- | --- | --- |
| `layer_label`, `layer_label_short`, `layer_type`, `func_name`, `grad_fn_name`, `containing_module` | scalar strings or `None` | Aggregate layer identity (`torchlens/data_classes/layer_log.py:67-85`, `torchlens/data_classes/layer_log.py:140-142`). |
| `layer_type_num`, `layer_total_num`, `num_passes`, `num_args`, `num_positional_args`, `num_keyword_args`, `tensor_memory`, `flops_forward`, `flops_backward`, `num_params_total`, `num_params_trainable`, `num_params_frozen`, `params_memory` | scalar ints | First-pass or merged numeric metadata (`torchlens/data_classes/layer_log.py:68-114`). |
| `source_model_log` | weakref-backed ref to `ModelLog` | Pickle strips `_source_model_log_ref`, then `ModelLog.__setstate__` reconnects it (`torchlens/data_classes/layer_log.py:74-78`, `torchlens/data_classes/layer_log.py:202-224`, `torchlens/data_classes/model_log.py:356-365`). |
| `func_applied`, `activation_postfunc` | callable or `None` | Same pickle concerns as `LayerPassLog` for `func_applied` (`torchlens/data_classes/layer_log.py:81`, `torchlens/data_classes/layer_log.py:98-100`). |
| `tensor_shape`, `tensor_dtype`, `output_device`, `detach_saved_tensor`, `save_gradients` | scalar metadata | Representative values from first pass (`torchlens/data_classes/layer_log.py:92-101`). |
| `parent_param_barcodes`, `parent_param_shapes`, `containing_modules`, `modules_exited`, `module_passes_exited`, `cond_branch_start_children`, `cond_branch_then_children`, `cond_branch_else_children`, `pass_labels` | lists | Mostly pass-stripped aggregate metadata (`torchlens/data_classes/layer_log.py:107-117`, `torchlens/data_classes/layer_log.py:148-162`). |
| `parent_param_logs` | `list[ParamLog]` | Direct refs to parameter logs (`torchlens/data_classes/layer_log.py:108-114`). |
| `conditional_branch_stacks` | `list[list[tuple[int, str]]]` | Aggregate branch-stack signatures across passes (`torchlens/data_classes/layer_log.py:136-139`). |
| `conditional_branch_stack_passes`, `cond_branch_children_by_cond`, `cond_branch_elif_children` | nested dicts | Aggregated cross-pass conditional metadata (`torchlens/data_classes/layer_log.py:136-139`, `torchlens/postprocess/finalization.py:251-331`). |
| `equivalent_operations` | `set[str]` | Shared layer-equivalence set (`torchlens/data_classes/layer_log.py:119-122`). |
| `is_input_layer`, `is_output_layer`, `is_final_output`, `is_buffer_layer`, `is_internally_initialized`, `is_internally_terminated`, `is_terminal_bool_layer`, `is_scalar_bool`, `in_cond_branch`, `has_input_ancestor`, `is_leaf_module_output` | bool flags | Mixed first-pass and merged aggregate booleans (`torchlens/data_classes/layer_log.py:123-157`, `torchlens/postprocess/finalization.py:742-770`). |
| `buffer_address`, `buffer_parent`, `buffer_pass`, `io_role`, `scalar_bool_value` | scalar or `None` | Special-case aggregate fields used for compatibility/visualization (`torchlens/data_classes/layer_log.py:127-157`). |
| `passes` | `dict[int, LayerPassLog]` | Main route back to pass-level payloads (`torchlens/data_classes/layer_log.py:159-162`). |
| Extras not in `LAYER_LOG_FIELD_ORDER`: `modules_exited`, `module_passes_exited`, `buffer_pass`, `has_input_ancestor`, `io_role`, `is_leaf_module_output`, `in_cond_branch` | additional aggregate compatibility fields | They are assigned in the class body and merged in `_build_layer_logs`, but not listed in the canonical field order (`torchlens/data_classes/layer_log.py:144-158`, `torchlens/postprocess/finalization.py:742-770`). |

#### BufferLog

Class definition: `torchlens/data_classes/buffer_log.py:22-69`

`BufferLog` subclasses `LayerPassLog` and does not add new stored fields; it relies on the inherited `is_buffer_layer`, `buffer_address`, `buffer_pass`, and `buffer_parent` fields plus two computed properties:

| Attributes | Kind | Notes |
| --- | --- | --- |
| all `LayerPassLog` fields | inherited | Buffer entries participate in the same graph/log schema as ordinary layer-pass entries (`torchlens/data_classes/buffer_log.py:22-33`). |
| `name` | computed `str` | Last path component of `buffer_address` (`torchlens/data_classes/buffer_log.py:35-42`). |
| `module_address` | computed `str` | Parent module address derived from `buffer_address` (`torchlens/data_classes/buffer_log.py:43-50`). |

#### ModuleLog

Class definition: `torchlens/data_classes/module_log.py:104-217`
Canonical field list: `torchlens/constants.py:402-447`

| Attributes | Kind | Notes for pickle/export design |
| --- | --- | --- |
| `address`, `name`, `module_class_name`, `source_file`, `class_docstring`, `init_signature`, `init_docstring`, `forward_signature`, `forward_docstring`, `address_parent`, `call_parent` | scalar strings or `None` | Module identity/source metadata (`torchlens/data_classes/module_log.py:167-187`). |
| `source_line`, `address_depth`, `nesting_depth`, `num_passes`, `num_params`, `num_params_trainable`, `num_params_frozen`, `params_memory` | scalar ints | Mostly flat summary fields (`torchlens/data_classes/module_log.py:172-203`). |
| `all_addresses`, `address_children`, `call_children`, `pass_labels`, `all_layers`, `buffer_layers`, `methods` | `list[str]` | Good candidates for normalized child tables rather than CSV columns (`torchlens/data_classes/module_log.py:168`, `torchlens/data_classes/module_log.py:180-205`, `torchlens/data_classes/module_log.py:212`). |
| `passes` | `dict[int, ModulePassLog]` | Pass-level module execution objects; there is no export helper for them (`torchlens/data_classes/module_log.py:188-190`). |
| `params` | `ParamAccessor` | Parameter metadata scoped to this module (`torchlens/data_classes/module_log.py:196-203`). |
| `requires_grad`, `is_training`, `has_forward_hooks`, `has_backward_hooks` | bool flags | Flat summary fields (`torchlens/data_classes/module_log.py:203`, `torchlens/data_classes/module_log.py:208-210`). |
| `extra_attributes` | `dict[str, Any]` | Potentially arbitrary Python objects, not inherently tabular (`torchlens/data_classes/module_log.py:211`). |
| Extra `_source_model_log_ref` | weakref to `ModelLog` | There is no `ModuleLog.__getstate__`; instead the entire `_module_logs` accessor is stripped at `ModelLog.__getstate__` time (`torchlens/data_classes/module_log.py:214-243`, `torchlens/data_classes/model_log.py:337-345`). |
| Extra `_buffer_accessor` | cached `BufferAccessor|None` | Lazily scoped buffer accessor for this module (`torchlens/data_classes/module_log.py:205-207`, `torchlens/data_classes/module_log.py:289-307`). |

Related nested type: `ModulePassLog`
- Definition: `torchlens/data_classes/module_log.py:36-103`
- Contents: `module_address`, `all_module_addresses`, `pass_num`, `pass_label`, `layers`, `input_layers`, `output_layers`, `forward_args`, `forward_kwargs`, `call_parent`, `call_children`.
- Important detail: `_build_module_logs()` nulls out `forward_args` and `forward_kwargs` after construction to release tensor references (`torchlens/postprocess/finalization.py:691-695`).

#### ParamLog

Class definition: `torchlens/data_classes/param_log.py:29-188`
Canonical field list: `torchlens/constants.py:361-382`

| Attributes | Kind | Notes for pickle/export design |
| --- | --- | --- |
| `address`, `name`, `module_address`, `module_type`, `barcode` | scalar strings | Stable parameter identity (`torchlens/data_classes/param_log.py:51-60`). |
| `shape`, `grad_shape` | shape tuples or `None` | Flat metadata (`torchlens/data_classes/param_log.py:53`, `torchlens/data_classes/param_log.py:73`, `torchlens/data_classes/param_log.py:119-127`). |
| `dtype`, `grad_dtype` | `torch.dtype|None` | PyTorch type objects (`torchlens/data_classes/param_log.py:54`, `torchlens/data_classes/param_log.py:74`, `torchlens/data_classes/param_log.py:129-137`). |
| `num_params`, `memory`, `grad_memory`, `num_passes` | scalar ints | Flat numeric metadata (`torchlens/data_classes/param_log.py:55-56`, `torchlens/data_classes/param_log.py:69-76`, `torchlens/data_classes/param_log.py:139-147`). |
| `trainable`, `is_quantized`, `has_grad`, `has_optimizer` | bool-like flags / optional bool | `has_grad` is lazy, computed from `_param_ref.grad` (`torchlens/data_classes/param_log.py:57`, `torchlens/data_classes/param_log.py:83-92`, `torchlens/data_classes/param_log.py:94-117`). |
| `memory_str`, `grad_memory_str` | derived strings | Convenience display fields (`torchlens/data_classes/param_log.py:78-80`, `torchlens/data_classes/param_log.py:149-157`). |
| `used_by_layers`, `linked_params` | `list[str]` | Reverse mappings populated during postprocessing (`torchlens/data_classes/param_log.py:70-71`, `torchlens/postprocess/finalization.py:78-118`). |
| Extra `_param_ref` | strong ref to live `nn.Parameter` or `None` | This is intentionally retained until cleanup/release, and there is no custom pickle hook to strip it (`torchlens/data_classes/param_log.py:63-67`, `torchlens/data_classes/param_log.py:181-184`, `torchlens/data_classes/cleanup.py:31-40`). |
| Extra `_has_grad`, `_grad_shape`, `_grad_dtype`, `_grad_memory`, `_grad_memory_str` | cached lazy-gradient state | Internal cache fields not listed in `PARAM_LOG_FIELD_ORDER` (`torchlens/data_classes/param_log.py:72-76`). |

## 2. Gaps vs user goals

### a. Pickling a ModelLog and round-tripping it cleanly

Current state:
- There are explicit pickle hooks on `ModelLog`, `LayerLog`, and `LayerPassLog` (`torchlens/data_classes/model_log.py:337-365`, `torchlens/data_classes/layer_log.py:216-224`, `torchlens/data_classes/layer_pass_log.py:390-398`).
- `LayerPassLog.func_applied` stores the producing callable directly, and output/source logging populate that field from the original function object (`torchlens/data_classes/layer_pass_log.py:163`, `torchlens/capture/output_tensors.py:318`, `torchlens/capture/source_tensors.py:167-168`).
- `ModelLog.__getstate__` strips `_module_logs` and `_buffer_accessor`, but `ModelLog.__setstate__` only recreates an empty `ModuleAccessor` and leaves `_buffer_accessor` unreconstructed; it only restores layer-related backrefs (`torchlens/data_classes/model_log.py:337-365`). The real builder for modules/buffers is `_build_module_logs()`, which is not called on unpickle (`torchlens/postprocess/finalization.py:518-695`).

Runtime experiments (throwaway scripts, not checked in):
- `nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))` failed at `pickle.dumps(log)` with `_pickle.PicklingError: Can't pickle <built-in function linear>...`.
- A parameter-free `torch.sin(x) + 1` model did round-trip, but the restored object had `len(rt.modules) == 0`, `rt.buffers is None`, and `rt.root_module` failed because module/buffer accessors were not rebuilt.
- A parameterized `x * self.scale` model also round-tripped, kept `ParamLog._param_ref`, and still came back with `len(rt.modules) == 0`.

Existing test coverage is much narrower than “clean round-trip”:
- `tests/test_metadata.py` and `tests/test_conditional_branches.py` only assert pickle round-trip behavior for logs captured with `save_source_context=False`, focusing on `FuncCallLocation` placeholder state, not full `ModelLog` usability after reload (`tests/test_metadata.py:980-1030`, `tests/test_conditional_branches.py:1900-1930`).
- Those tests pass because `FuncCallLocation` explicitly zeros `_frame_func_obj` and fills no-source placeholders when source loading is disabled (`torchlens/data_classes/func_call_location.py:87-92`, `torchlens/data_classes/func_call_location.py:125-133`).

Verdict:
- Exists: partially.
- Works: only for some logs; common `nn.Linear` logs fail outright, and successful round-trips still lose module/buffer access.
- Missing: a stable callable-serialization strategy, module/buffer accessor rebuild, and explicit compatibility guarantees. The logical implementation points are `LayerPassLog.func_applied` capture and `ModelLog.__setstate__` rehydration (`torchlens/capture/output_tensors.py:318`, `torchlens/data_classes/model_log.py:346-365`).

### b. Exporting layer-level metadata to a pandas DataFrame

Current state:
- `model_log.layers.to_pandas()` exists and returns one row per aggregate `LayerLog` (`torchlens/data_classes/model_log.py:515-519`, `torchlens/data_classes/layer_log.py:665-687`).
- `model_log.to_pandas()` also exists, but it is pass-level, not aggregate layer-level (`torchlens/data_classes/interface.py:421-541`).

Verdict:
- Exists: yes.
- Works: yes for pandas users.
- Missing: explicit naming that distinguishes layer-level vs pass-level exports, and wrapper methods for CSV/Parquet/JSON.

### c. Exporting pass-level / module-level / buffer-level / param-level metadata to DataFrames

Current state:
- Pass-level: `model_log.to_pandas()` exports one row per `LayerPassLog` (`torchlens/data_classes/interface.py:421-541`).
- Layer-level: `model_log.layers.to_pandas()` exports one row per `LayerLog` (`torchlens/data_classes/layer_log.py:665-687`).
- Module-level summary: `model_log.modules.to_pandas()` exports one row per `ModuleLog` (`torchlens/data_classes/module_log.py:496-510`).
- Module-contained layers: `model_log.modules["addr"].to_pandas()` exports layers inside one module, not module passes (`torchlens/data_classes/module_log.py:399-415`).
- Buffer-level: no DataFrame export surface on `BufferAccessor` (`torchlens/data_classes/buffer_log.py:72-132`).
- Param-level: no DataFrame export surface on `ParamAccessor` (`torchlens/data_classes/param_log.py:191-248`).
- Module-pass-level: `ModulePassLog` has no `to_pandas()` and its `forward_args`/`forward_kwargs` are nulled after module-log construction (`torchlens/data_classes/module_log.py:36-103`, `torchlens/postprocess/finalization.py:691-695`).

Verdict:
- Exists: pass-level yes, layer-level yes, module summary yes, module-contained-layers yes.
- Works: limited but usable for those exact shapes.
- Missing: first-class DataFrames for `ParamLog`, `BufferLog`, and `ModulePassLog`, plus a normalized schema for nested dict/list/set fields.

### d. Exporting to CSV, Parquet, JSON

Current state:
- The only first-class CSV writer in scope is the batch-validation harness, which persists validation results to CSV (`torchlens/user_funcs.py:635-697`).
- The layer/module DataFrame surfaces let a caller manually call pandas writers themselves, but TorchLens does not wrap those operations (`torchlens/data_classes/interface.py:421-541`, `torchlens/data_classes/layer_log.py:665-687`, `torchlens/data_classes/module_log.py:399-415`, `torchlens/data_classes/module_log.py:496-510`).
- There are no native `to_csv`, `to_parquet`, `to_json`, `to_dict`, or `asdict` methods on the audited data classes or accessors (`torchlens/data_classes/model_log.py:545-558`, `torchlens/data_classes/param_log.py:191-248`, `torchlens/data_classes/buffer_log.py:72-132`).

Verdict:
- Exists: CSV only for validation results, not for `ModelLog`/`LayerLog`/`ParamLog` persistence.
- Works: only if the user manually converts existing DataFrames.
- Missing: native CSV/Parquet/JSON exporters, schema/version metadata, and importers.

### e. Saving activations to disk (format, eager vs streaming, selective-layer vs all)

Current state:
- Activation capture is in memory. `save_tensor_data()` clones the tensor, optionally moves it to `output_device`, optionally applies `activation_postfunc`, and stores the result on `LayerPassLog.activation` (`torchlens/data_classes/layer_pass_log.py:463-502`).
- `log_forward_pass()` supports saving activations for `"all"`, `"none"`, or selective layer keys; selective mode runs an exhaustive pass then a fast pass (`torchlens/user_funcs.py:201-215`, `torchlens/user_funcs.py:286-346`).
- `save_new_activations()` replays the same graph and refreshes activations in memory only (`torchlens/capture/trace.py:55-140`, `torchlens/capture/output_tensors.py:608-644`, `torchlens/capture/source_tensors.py:286-338`).

Verdict:
- Exists: in-memory activation capture, including selective-layer refresh.
- Works: yes in memory, subject to fast-path graph-stability checks.
- Missing: any disk format, any streaming/spill-to-disk path, any tensor sharding, any lazy reload.

### f. Loading any of the above back into usable ModelLog/LayerLog objects

Current state:
- There is no public loader API in `torchlens.__init__` or `torchlens/user_funcs.py` (`torchlens/__init__.py:14-34`, `torchlens/user_funcs.py:153-697`).
- Python pickle is the only implicit “load” path, and it is incomplete for module/buffer accessors even when it succeeds (`torchlens/data_classes/model_log.py:337-365`, `torchlens/postprocess/finalization.py:518-695`).
- No CSV/Parquet/JSON importers exist for any of the data classes or accessors (`torchlens/data_classes/model_log.py:545-558`, `torchlens/data_classes/param_log.py:191-248`, `torchlens/data_classes/buffer_log.py:72-132`).

Verdict:
- Exists: implicit `pickle.load()` only, and only partially.
- Works: not cleanly enough to call this a supported load path.
- Missing: a package-supported loader plus a format that can reconstruct usable `ModelLog`/`LayerLog`/`ModuleLog` objects deterministically.

## 3. Risks and tricky spots

### Circular references and object graphs

- `LayerPassLog` stores a weakref-backed `source_model_log`, but also a strong `parent_layer_log` backref that is intentionally outside `FIELD_ORDER` (`torchlens/data_classes/layer_pass_log.py:80-85`, `torchlens/data_classes/layer_pass_log.py:115-118`, `torchlens/data_classes/layer_pass_log.py:269-273`).
- `LayerLog` stores `passes: Dict[int, LayerPassLog]`, and each pass stores `parent_layer_log`, so the aggregate/pass layer view is cyclic by design (`torchlens/data_classes/layer_log.py:159-162`, `torchlens/postprocess/finalization.py:764-767`).
- `ModuleLog` stores `passes: Dict[int, ModulePassLog]` plus a weakref-backed `_source_model_log_ref` (`torchlens/data_classes/module_log.py:188-217`).
- `ParamLog` keeps a strong `_param_ref` to the actual `nn.Parameter` until cleanup or explicit release (`torchlens/data_classes/param_log.py:63-67`, `torchlens/data_classes/cleanup.py:31-40`).
- Cleanup is aggressive because these graphs are cyclic and heavy; it deletes layer-entry attributes first, then `ModelLog` fields and internal containers (`torchlens/data_classes/cleanup.py:42-83`).

### Weakrefs

- Weakref-backed fields exist on `LayerPassLog.source_model_log`, `LayerLog.source_model_log`, `ModuleLog._source_model_log`, and accessor-side source refs (`torchlens/data_classes/layer_pass_log.py:376-389`, `torchlens/data_classes/layer_log.py:202-214`, `torchlens/data_classes/module_log.py:233-243`, `torchlens/data_classes/layer_log.py:611-619`, `torchlens/data_classes/buffer_log.py:83-92`).
- The pickle hooks only strip/rebuild weakrefs for `LayerPassLog` and `LayerLog`; `ModuleLog` never gets a custom state hook because `_module_logs` is stripped wholesale at the `ModelLog` level (`torchlens/data_classes/model_log.py:337-365`).

### Tensor device, grad, and post-processing state

- Saved activations are cloned, optionally detached, and optionally moved to `output_device`; default behavior can preserve GPU residency (`torchlens/data_classes/layer_pass_log.py:481-490`, `torchlens/user_funcs.py:224-227`).
- Saved gradients are detached/cloned snapshots (`torchlens/data_classes/layer_pass_log.py:503-517`).
- `ParamLog` gradient metadata is lazy and tied to `_param_ref.grad`, so a successful pickle can carry a serialized parameter snapshot rather than a live link to the current model instance (`torchlens/data_classes/param_log.py:94-117`).
- `activation_postfunc` can replace `activation` with any Python object, not necessarily a tensor (`torchlens/data_classes/layer_pass_log.py:488-490`).

### Memory footprint

- Every saved layer can hold an activation payload, optional copied args/kwargs, optional gradient, and optional child-version tensors used for validation replay (`torchlens/data_classes/layer_pass_log.py:135-160`, `torchlens/data_classes/layer_pass_log.py:495-501`, `torchlens/capture/output_tensors.py:392-405`).
- `ModelLog` explicitly tracks aggregate activation memory and saved-activation memory, which reflects that activations are kept resident in memory today (`torchlens/data_classes/model_log.py:252-255`).
- `save_new_activations()` clears and repopulates those in-memory payloads; it is not a spill mechanism (`torchlens/capture/trace.py:88-109`).

### Thread safety and the global toggle

- Logging is controlled by global mutable state in `torchlens/_state.py`; `active_logging()` is explicitly “NOT nestable” and assumes only one logged forward pass at a time (`torchlens/_state.py:183-209`).
- There is no dedicated load path that fences itself from this global state. Unpickling itself does not flip the toggle, but the package has no designed concurrency story for “load while another thread is logging” because serialization/load APIs do not exist (`torchlens/_state.py:36-54`, `torchlens/__init__.py:14-34`).

### Pickle security and PyTorch-version brittleness

- The current implicit format is raw Python pickle; loading untrusted pickles is therefore arbitrary-code-execution unsafe in the general Python sense.
- `func_applied` stores raw callables, and validation replay later calls them directly (`torchlens/capture/output_tensors.py:318`, `torchlens/validation/core.py:468-477`). That makes pickle compatibility sensitive to callable identity and refactors, and runtime experiments already show built-in-op failures.
- RNG snapshots use opaque Python/NumPy/Torch state objects (`torchlens/utils/rng.py:49-89`), and dtypes/devices/autocast metadata are also framework-version-specific (`torchlens/data_classes/layer_pass_log.py:144`, `torchlens/data_classes/layer_pass_log.py:169-170`, `torchlens/utils/rng.py:92-141`).
- `FuncCallLocation` only clears `_frame_func_obj` after lazy source loading when source loading is enabled; pickling before that can capture function objects unexpectedly (`torchlens/data_classes/func_call_location.py:109-123`, `torchlens/data_classes/func_call_location.py:140-198`).

## 4. Recommendations

### Format strategy

Recommendation: make a directory-bundle format the primary path, with:
- `manifest.json` for versioning and bundle metadata,
- Parquet tables for flat metadata surfaces,
- `safetensors` shards for activations and gradients,
- explicit pickle wrappers only as a debug/escape-hatch path.

Why this is the right tradeoff here:
- Pickle-only is already broken for common built-in ops and incomplete even when it succeeds because module/buffer accessors are not rebuilt (`torchlens/capture/output_tensors.py:318`, `torchlens/data_classes/model_log.py:337-365`).
- Metadata is already naturally splitting into flat summaries (`to_pandas()` surfaces) plus large tensor payloads, so Parquet + `safetensors` matches the real object shape better than one monolithic pickle (`torchlens/data_classes/interface.py:421-541`, `torchlens/data_classes/layer_log.py:665-687`, `torchlens/data_classes/module_log.py:399-415`, `torchlens/data_classes/module_log.py:496-510`).
- `safetensors` avoids arbitrary code execution and gives deterministic tensor payloads; Parquet gives queryable tables.

Tradeoff I would accept:
- I would accept an extra dependency and a more explicit schema-normalization step in exchange for stability, safer loads, and large-model practicality.

I would not ship “pickle-only” as the product story.

### Should streaming-to-disk during forward pass be in scope?

Recommendation: not for the first implementation.

Why:
- The current capture pipeline assumes `LayerPassLog.activation` is immediately available in memory for validation, output-node postprocessing, fast-path refresh, and some graph-replay logic (`torchlens/data_classes/layer_pass_log.py:463-502`, `torchlens/postprocess/__init__.py:200-247`, `torchlens/validation/core.py:450-500`).
- Streaming would cut across `save_tensor_data()`, source/output fast paths, validation replay, and output-node copying. That is real surface area, not a small add-on (`torchlens/data_classes/layer_pass_log.py:463-502`, `torchlens/capture/output_tensors.py:608-644`, `torchlens/capture/source_tensors.py:286-338`).

Effort estimate:
- Basic bundle save/load without streaming: about 1 to 2 engineer-weeks.
- Robust streaming capture with selective layers and replay compatibility: about 3 to 5 engineer-weeks.

### Lazy vs eager tensor loading

Recommendation: lazy by default, eager optional.

Why:
- Activations dominate size and current TorchLens already exposes useful metadata without needing every tensor resident (`torchlens/data_classes/model_log.py:252-255`, `torchlens/data_classes/interface.py:421-541`).
- Large-model use cases are exactly where eager reload will hurt the most.

Tradeoff I would accept:
- Slightly more complicated access semantics and first-access latency on `layer.activation` are worth it to keep bundle loads cheap and predictable.

Practical shape:
- `tl.load(path, lazy_tensors=True)` should build the graph/metadata eagerly and materialize tensors on first access.
- `tl.load(path, lazy_tensors=False)` should eagerly load tensor shards into memory.

### Concrete API proposal

I would build this API:

```python
model_log.save(
    path,
    format="auto",
    include_activations=True,
    include_gradients=False,
    tensor_format="safetensors",
    table_format="parquet",
    overwrite=False,
)

tl.load(path, lazy_tensors=True, map_location="cpu") -> ModelLog

model_log.passes_df() -> pd.DataFrame
model_log.layers_df() -> pd.DataFrame
model_log.modules_df() -> pd.DataFrame
model_log.params_df() -> pd.DataFrame
model_log.buffers_df() -> pd.DataFrame

model_log.to_csv(path, table="passes")
model_log.to_parquet(path, table="passes")
model_log.to_json(path, table="passes")

model_log.to_pickle(path)
tl.load_pickle(path) -> ModelLog
```

Reasoning:
- The current `to_pandas()` name is ambiguous because `ModelLog.to_pandas()` is pass-level while `LayerAccessor.to_pandas()` is layer-level (`torchlens/data_classes/interface.py:421-541`, `torchlens/data_classes/layer_log.py:665-687`). The new API should name the grain explicitly.
- `tl.load(...)` should exist at package level because there is currently no load surface in `__init__.py` (`torchlens/__init__.py:14-34`).
- Pickle should be explicit and visibly “less safe / less stable,” not the default silent protocol.

### Backward compatibility and format versioning

Recommendation: yes, add a format version tag immediately.

I would store:
- `io_format_version`
- `torchlens_version`
- `torch_version`
- `schema_versions` per table
- optional feature flags like `has_activations`, `has_gradients`, `lazy_tensor_encoding`

Migration approach:
- Directory bundle manifest drives migrations.
- Keep loader-side migrators for at least N-1 minor schema versions.
- Treat raw pickle as best-effort debug compatibility, not archival compatibility.

This is necessary because the current object model already evolves via `FIELD_ORDER` changes and custom state logic, and older pickles are not otherwise self-describing (`torchlens/constants.py:3-11`, `torchlens/data_classes/model_log.py:337-365`).

## 5. Implementation outline

Ordered by dependency.

### Phase 1: Normalize export surfaces first

Goal:
- Make existing in-memory metadata exportable at clear grains before introducing a bundle format.

Work items:
1. Add explicit DataFrame builders with grain-specific names:
   - `ModelLog.passes_df()`
   - `ModelLog.layers_df()`
   - `ModelLog.modules_df()`
   - `ModelLog.params_df()`
   - `ModelLog.buffers_df()`
2. Keep `to_pandas()` as a compatibility alias for `passes_df()` or deprecate it carefully.
3. Add tests for param/buffer/module-pass exports.

Files touched:
- `torchlens/data_classes/interface.py`
- `torchlens/data_classes/layer_log.py`
- `torchlens/data_classes/module_log.py`
- `torchlens/data_classes/param_log.py`
- `torchlens/data_classes/buffer_log.py`
- `torchlens/data_classes/model_log.py`
- `torchlens/__init__.py`
- `tests/`

### Phase 2: Introduce a stable bundle writer

Goal:
- Persist metadata and tensors without using raw pickle as the primary format.

Work items:
1. Add an `io/` module or subpackage for manifest writing, Parquet writers, and `safetensors` tensor sharding.
2. Add `ModelLog.save(...)`.
3. Define manifest schema and versioning.
4. Encode nested list/dict/set fields either as normalized child tables or JSON-encoded columns where appropriate.

Files touched:
- new `torchlens/io/` package
- `torchlens/data_classes/model_log.py`
- `torchlens/__init__.py`
- possibly `pyproject.toml` / package metadata for dependencies
- `tests/`

### Phase 3: Build a loader that reconstructs usable logs

Goal:
- `tl.load(...)` returns a usable `ModelLog`, not just raw tables.

Work items:
1. Add bundle manifest reader and table loader.
2. Reconstruct `ModelLog`, `LayerLog`, `LayerPassLog`, `ModuleLog`, `ParamLog`, and `BufferLog` from normalized tables.
3. Rebuild `ModuleAccessor`, `BufferAccessor`, and weakref-backed graph links explicitly.
4. Add lazy tensor references/proxies for activations and gradients.

Files touched:
- new `torchlens/io/load.py` or equivalent
- `torchlens/data_classes/model_log.py`
- `torchlens/data_classes/layer_pass_log.py`
- `torchlens/data_classes/layer_log.py`
- `torchlens/data_classes/module_log.py`
- `torchlens/data_classes/param_log.py`
- `torchlens/data_classes/buffer_log.py`
- `torchlens/__init__.py`
- `tests/`

### Phase 4: Make pickle explicit and demoted

Goal:
- Preserve a debug path without pretending it is the main archival format.

Work items:
1. Add `ModelLog.to_pickle()` and `tl.load_pickle()`.
2. Warn on load that pickle is trusted-input-only and version-brittle.
3. Either strip or canonicalize `func_applied` for pickle mode specifically, or document that pickle is unsupported for some ops.

Files touched:
- `torchlens/data_classes/model_log.py`
- `torchlens/data_classes/layer_pass_log.py`
- `torchlens/__init__.py`
- `tests/`

### Phase 5: Optional streaming / spill-to-disk capture

Goal:
- Reduce RAM pressure for very large captures.

Work items:
1. Introduce a tensor-storage backend abstraction under `save_tensor_data()`.
2. Support eager-in-memory and streamed-to-bundle backends.
3. Update fast-path refresh and validation replay to understand externalized tensors.

Files touched:
- `torchlens/data_classes/layer_pass_log.py`
- `torchlens/capture/output_tensors.py`
- `torchlens/capture/source_tensors.py`
- `torchlens/capture/trace.py`
- `torchlens/validation/core.py`
- new `torchlens/io/storage.py`
- `tests/`

Bottom line:
- TorchLens already has useful in-memory capture and a few DataFrame summaries.
- It does not yet have a coherent persistence system.
- I would build a versioned bundle format around Parquet + `safetensors`, keep pickle only as an explicit escape hatch, and postpone true streaming capture until after stable save/load works.
