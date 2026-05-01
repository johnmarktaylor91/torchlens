# Method ledger

Generated for Phase 0.7 from `inspect.getmembers(cls)` plus public FIELD_ORDER entries. Counts below refer to public inspectable members; FIELD_ORDER data attributes are inventoried but not counted against method budgets.

## ModelLog

Current public inspectable members: 67. Target kept/new members: 35 (budget <= 40).

| Method/property/attribute | Current location | Status | Migration notes |
|---|---|---|---|
| `DEFAULT_FILL_STATE` (attribute) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; class-level implementation table. | No namespace alias proposed. |
| `FORK_POLICY` (attribute) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; fork machinery table. | No namespace alias proposed. |
| `PORTABLE_STATE_SPEC` (attribute) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; serialization machinery table. | No namespace alias proposed. |
| `activation_postfunc` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `activation_postfunc_repr` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `append_run_state_from` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; rerun append implementation helper. | No namespace alias proposed. |
| `attach_hooks` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `backward_memory_backend` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `backward_num_passes` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `backward_peak_memory_bytes` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `backward_root_grad_fn_id` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_num_passes` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffers` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `capture_full_args` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `check_metadata_invariants` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.validation.check_metadata_invariants(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `cleanup` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; lifecycle cleanup helper. | No namespace alias proposed. |
| `clear_hooks` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `conditional_arm_edges` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_branch_edges` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_edge_passes` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_elif_edges` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_else_edges` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_events` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_then_edges` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `current_function_call_barcode` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `detach_hooks` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `detach_saved_tensors` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `detect_loops` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `do` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `equivalent_operations` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `find_sites` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `flops_by_type` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.flops_by_type(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `fork` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `grad_fn_logs` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fn_order` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fns` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `gradient_postfunc` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `gradient_postfunc_repr` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `gradients_to_save` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `graph_shape_hash` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_backward_log` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_conditional_branching` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `has_gradients` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `input_id_at_capture` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `input_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `input_metadata` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `input_shape_hash` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `internally_initialized_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `internally_terminated_bool_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `internally_terminated_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `intervention_ready` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `io_format_version` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_appended` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_branching` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `is_recurrent` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `keep_unsaved_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `last_run_ctx` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `last_run_records` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `layer_dict_all_keys` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_dict_main_keys` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_labels` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_labels_no_pass` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_labels_w_pass` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_list` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_logs` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_num_passes` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layers` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `layers_with_params` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layers_with_saved_activations` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layers_with_saved_gradients` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `load` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `log_backward` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `logging_mode` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `macs_by_type` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.macs_by_type(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `mark_input_output_distances` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `max_recurrent_loops` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `model_name` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `modules` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `name` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_context_lines` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_grad_fns` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `num_intervening_grad_fns` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `num_operations` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_tensors_saved` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_tensors_total` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `operation_history` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `orphan_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `output_device` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `output_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `param_logs` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `parent_run` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `pass_end_time` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `pass_start_time` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `preview_fastlog` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.fastlog.preview(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `print_all_fields` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; debug helper accidentally exposed. | No namespace alias proposed. |
| `random_seed_used` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `recording_backward` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; internal backward-recording context. | No namespace alias proposed. |
| `relationship_evidence` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `release_param_refs` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; memory cleanup helper. | No namespace alias proposed. |
| `render_dagua_graph` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.viz.render_dagua_graph(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `render_graph` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.viz.render_graph(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `replace_run_state_from` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; rerun state replacement helper. | No namespace alias proposed. |
| `replay` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `replay_from` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `rerun` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `resolve_sites` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `root_module` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `run_state` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `save_function_args` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_gradients` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_intervention` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `save_new_activations` (method) | `torchlens.data_classes.model_log.ModelLog` | DROP from public surface; legacy refresh helper superseded by save/replay paths. | No namespace alias proposed. |
| `save_raw_activation` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_raw_gradient` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_rng_states` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_source_context` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `saved_activation_memory` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `saved_activation_memory_str` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `set` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `show` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `show_backward_graph` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.viz.show_backward_graph(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `source_model_class` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `source_model_id` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `summary` (method) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `time_cleanup` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `time_forward_pass` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `time_function_calls` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `time_logging` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `time_setup` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `time_total` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `to_csv` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.export.csv(log, path)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_dagua_graph` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.viz.to_dagua_graph(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_json` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.export.json(log, path)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_pandas` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.export.pandas(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_parquet` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.export.parquet(log, path)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `total_activation_memory` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_activation_memory_str` (property) | `torchlens.data_classes.model_log.ModelLog` | KEEP |  |
| `total_autograd_saved_bytes` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_flops` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_flops(log)` | Aggregate report metric; method/property budget pressure. |
| `total_flops_backward` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_flops_backward(log)` | Aggregate report metric; method/property budget pressure. |
| `total_flops_forward` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_flops_forward(log)` | Aggregate report metric; method/property budget pressure. |
| `total_macs` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_macs(log)` | Aggregate report metric; method/property budget pressure. |
| `total_macs_backward` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_macs_backward(log)` | Aggregate report metric; method/property budget pressure. |
| `total_macs_forward` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_macs_forward(log)` | Aggregate report metric; method/property budget pressure. |
| `total_param_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_param_tensors` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_params` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_params_frozen` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_params_memory` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `total_params_memory_str` (property) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.report.total_params_memory_str(log)` | Aggregate report metric; method/property budget pressure. |
| `total_params_trainable` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `train_mode` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `unlogged_layers` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `validate_forward_pass` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.validation.validate_forward_pass(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `validate_saved_activations` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.validation.validate_saved_activations(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `verbose` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `visualization_field_audit` (method) | `torchlens.data_classes.model_log.ModelLog` | MOVE to namespace `tl.viz.visualization_field_audit(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `weight_fingerprint_at_capture` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `weight_fingerprint_full` (attribute) | `torchlens.constants.MODELLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |

## LayerLog

Current public inspectable members: 49. Target kept/new members: 30 (budget <= 30).

| Method/property/attribute | Current location | Status | Migration notes |
|---|---|---|---|
| `PORTABLE_STATE_SPEC` (attribute) | `torchlens.data_classes.layer_log.LayerLog` | DROP from public surface; serialization machinery table. | No namespace alias proposed. |
| `activation` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `activation_postfunc` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `autograd_saved_bytes` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `autograd_saved_tensor_count` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_address` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_parent` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `captured_args` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `captured_kwargs` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `child_layers` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `child_layers_per_pass` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `child_passes_per_layer` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `co_parent_layers` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `cond_branch_children_by_cond` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_elif_children` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_else_children` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_start_children` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_then_children` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_branch_stack_passes` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_branch_stacks` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `containing_module` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `containing_modules` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `corresponding_grad_fn` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `creation_order` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `detach_saved_tensor` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `edges_vary_across_passes` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `equivalent_operations` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `extra_data` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `flops_backward` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `flops_forward` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_applied` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_argnames` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_call_stack` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `func_config` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_is_inplace` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_name` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_rng_states` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `func_time` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `get_child_layers` (method) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `get_parent_layers` (method) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `grad_fn_id` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fn_name` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fn_object` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `gradient` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `has_children` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `has_co_parents` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `has_gradient` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `has_parents` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `has_saved_activations` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `has_siblings` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `is_buffer_layer` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_computed_inside_submodule` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `is_final_output` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_input_layer` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_internally_initialized` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_internally_terminated` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_output_layer` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_part_of_iterable_output` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_scalar_bool` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_terminal_bool_layer` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `iterable_output_index` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_no_pass` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `layer_label_no_pass_short` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `layer_label_short` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_w_pass` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `layer_label_w_pass_short` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `layer_total_num` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_type` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_type_num` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `leaf_module_passes` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `lookup_keys` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `macs_backward` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |
| `macs_forward` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |
| `module_nesting_depth` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |
| `num_args` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_keyword_args` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_param_tensors` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |
| `num_params_frozen` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_params_total` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_params_trainable` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_passes` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_positional_args` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `operation_equivalence_type` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `operation_num` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `output_device` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `params_memory` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params_memory_str` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |
| `parent_layer_arg_locs` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `parent_layers` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `parent_layers_per_pass` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `parent_param_barcodes` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_param_logs` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_param_shapes` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_passes_per_layer` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `pass_labels` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `pass_num` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `passes` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `print_all_fields` (method) | `torchlens.data_classes.layer_log.LayerLog` | DROP from public surface; debug helper accidentally exposed. | No namespace alias proposed. |
| `save_gradients` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `scalar_bool_value` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `sibling_layers` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `source_model_log` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `tensor` (property) | `torchlens.data_classes.layer_log.LayerLog` | KEEP |  |
| `tensor_dtype` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `tensor_memory` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `tensor_memory_str` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |
| `tensor_shape` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `transformed_activation_dtype` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation_memory` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation_shape` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_details(layer)` | Detailed relationship/derived view; reduce object method budget. |
| `transformed_gradient_dtype` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient_memory` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient_shape` (attribute) | `torchlens.constants.LAYERLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `uses_params` (property) | `torchlens.data_classes.layer_log.LayerLog` | MOVE to namespace `tl.report.layer_metric(layer, name)` | Derived report metric; keep raw fields on object. |

## LayerPassLog

Current public inspectable members: 30. Target kept/new members: 23 (budget <= 25).

| Method/property/attribute | Current location | Status | Migration notes |
|---|---|---|---|
| `DEFAULT_FILL_STATE` (attribute) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; class-level implementation table. | No namespace alias proposed. |
| `FORK_POLICY` (attribute) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; fork machinery table. | No namespace alias proposed. |
| `PORTABLE_STATE_SPEC` (attribute) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; serialization machinery table. | No namespace alias proposed. |
| `activation` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `activation_postfunc` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `args_captured` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `autograd_saved_bytes` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `autograd_saved_tensor_count` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `bool_conditional_id` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `bool_context_kind` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `bool_is_branch` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `bool_wrapper_kind` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_address` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_parent` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_pass` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `captured_arg_template` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `captured_args` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `captured_kwarg_template` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `captured_kwargs` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `child_layers` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `children_tensor_versions` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `co_parent_layers` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `cond_branch_children_by_cond` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_elif_children` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_else_children` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_start_children` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `cond_branch_then_children` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_branch_depth` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `conditional_branch_stack` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `container_spec` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `containing_module` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `containing_modules` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `copy` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; fork/postprocess implementation helper. | No namespace alias proposed. |
| `corresponding_grad_fn` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `creation_order` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `detach_saved_tensor` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `edge_uses` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `equivalent_operations` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `extra_data` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `feeds_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `flops_backward` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `flops_forward` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_applied` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_argnames` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_autocast_state` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_call_id` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_call_stack` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_config` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_is_inplace` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_kwargs_non_tensor` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_name` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_non_tensor_args` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_positional_args_non_tensor` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_rng_states` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `func_time` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `get_child_layers` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `get_parent_layers` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `grad_dtype` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fn_id` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fn_name` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_fn_object` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_memory` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `grad_memory_str` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP | Conservative keep; ambiguous public member. |
| `grad_shape` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `gradient` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_child_tensor_variations` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_children` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_co_parents` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `has_gradient` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_input_ancestor` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_internally_initialized_ancestor` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_parents` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `has_saved_activations` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_siblings` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `in_cond_branch` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `input_ancestors` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `internally_initialized_ancestors` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `internally_initialized_parents` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `intervention_log` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `io_role` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_buffer_layer` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_computed_inside_submodule` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `is_final_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_input_layer` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_internally_initialized` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_internally_terminated` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_leaf_module_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_output_ancestor` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_output_layer` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_part_of_iterable_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_scalar_bool` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_submodule_input` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `is_submodule_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `is_terminal_bool_layer` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `iterable_output_index` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_no_pass` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_no_pass_short` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_raw` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_short` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_w_pass` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_label_w_pass_short` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_total_num` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_type` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layer_type_num` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `leaf_module_pass` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `log_tensor_grad` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; autograd callback implementation helper. | No namespace alias proposed. |
| `lookup_keys` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `macs_backward` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `macs_forward` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `materialize_activation` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `materialize_gradient` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `max_distance_from_input` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `max_distance_from_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `min_distance_from_input` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `min_distance_from_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `module_address_normalized` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `module_entry_exit_thread_output` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `module_entry_exit_threads_inputs` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `module_nesting_depth` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `module_passes_entered` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `module_passes_exited` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `modules_entered` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `modules_entered_argnames` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `modules_exited` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_args` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_keyword_args` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_param_tensors` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `num_params_frozen` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_params_total` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_params_trainable` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_passes` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_positional_args` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `operation_equivalence_type` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `operation_num` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `output_descendants` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `output_device` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `output_path` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `params_memory` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params_memory_str` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `parent_layer_arg_locs` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_layers` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_param_barcodes` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_param_logs` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_param_passes` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_param_shapes` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `parent_params` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `pass_num` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `passes` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `print_all_fields` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; debug helper accidentally exposed. | No namespace alias proposed. |
| `recurrent_group` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `root_ancestors` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_gradients` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `save_tensor_data` (method) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | DROP from public surface; tensor persistence implementation helper. | No namespace alias proposed. |
| `scalar_bool_value` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `sibling_layers` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `source_model_log` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `tensor` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `tensor_dtype` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `tensor_label_raw` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `tensor_memory` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `tensor_memory_str` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |
| `tensor_shape` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation_dtype` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation_memory` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_activation_shape` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient_dtype` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient_memory` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `transformed_gradient_shape` (attribute) | `torchlens.constants.LAYERPASSLOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `uses_params` (property) | `torchlens.data_classes.layer_pass_log.LayerPassLog` | KEEP |  |

## ModuleLog

Current public inspectable members: 22. Target kept/new members: 16 (budget <= 20).

| Method/property/attribute | Current location | Status | Migration notes |
|---|---|---|---|
| `PORTABLE_STATE_SPEC` (attribute) | `torchlens.data_classes.module_log.ModuleLog` | DROP from public surface; serialization machinery table. | No namespace alias proposed. |
| `address` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `address_children` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `address_depth` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `address_parent` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `all_addresses` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `all_layers` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffer_layers` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `buffers` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `call_children` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `call_parent` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `class_docstring` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `extra_attributes` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `flops` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `flops_backward` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `flops_forward` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `forward_args` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `forward_docstring` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `forward_kwargs` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `forward_signature` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `gradient` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `has_backward_hooks` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `has_forward_hooks` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `init_docstring` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `init_signature` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `input_layers` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `is_shared` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `is_training` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `layers` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `macs` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `macs_backward` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `macs_forward` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `methods` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `module_class_name` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `name` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `nesting_depth` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_layers` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `num_params` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_params_frozen` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_params_trainable` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `num_passes` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `output_layers` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `params` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params_memory` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `params_memory_str` (property) | `torchlens.data_classes.module_log.ModuleLog` | KEEP |  |
| `pass_labels` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `passes` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `requires_grad` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `show_graph` (method) | `torchlens.data_classes.module_log.ModuleLog` | MOVE to namespace `tl.viz.show_module_graph(module_log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `source_file` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `source_line` (attribute) | `torchlens.constants.MODULELOG FIELD_ORDER` | KEEP | Public data field from FIELD_ORDER; keep as data-object attribute unless Phase 1a narrows fields separately. |
| `to_csv` (method) | `torchlens.data_classes.module_log.ModuleLog` | MOVE to namespace `tl.export.csv(log, path)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_json` (method) | `torchlens.data_classes.module_log.ModuleLog` | MOVE to namespace `tl.export.json(log, path)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_pandas` (method) | `torchlens.data_classes.module_log.ModuleLog` | MOVE to namespace `tl.export.pandas(log)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `to_parquet` (method) | `torchlens.data_classes.module_log.ModuleLog` | MOVE to namespace `tl.export.parquet(log, path)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |

## Bundle

Current public inspectable members: 24. Target kept/new members: 24 (budget <= 25).

| Method/property/attribute | Current location | Status | Migration notes |
|---|---|---|---|
| `show_diff` (method) | `NEW (Phase 9)` | NEW | Bundle diff renderer hero demo; brings Bundle to 25/25. |
| `add` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `attach_hooks` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `baseline_name` (property) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `clear` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `cluster` (method) | `torchlens.intervention.bundle.Bundle` | MOVE to namespace `tl.viz.cluster(bundle)` | Preserve method wrapper for one minor cycle if currently documented or used in tests. |
| `compare_at` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `diff` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `do` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `evict_all_but` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `fork` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `help` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `joint_metric` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `members` (property) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `metric` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `most_changed` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `names` (property) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `node` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `pop` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `relationship` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `relationship_matrix` (property) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `replay` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `rerun` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `set_capacity` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |
| `show` (method) | `torchlens.intervention.bundle.Bundle` | KEEP |  |

## Budget summary

| Class | Current public inspectable members | Target kept/new members | Budget |
|---|---:|---:|---:|
| ModelLog | 67 | 35 | 40 |
| LayerLog | 49 | 30 | 30 |
| LayerPassLog | 30 | 23 | 25 |
| ModuleLog | 22 | 16 | 20 |
| Bundle | 24 | 24 | 25 |
