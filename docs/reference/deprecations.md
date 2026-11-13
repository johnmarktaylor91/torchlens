# TorchLens Removed-Spellings Ledger
<!-- # record-groups: keep-flat (this page lists the former flat spellings) -->

Every warning-producing compatibility shim listed below was REMOVED outright in
the 2026-08 shim-removal pass (before launch the policy is to remove and
rename, not to keep compatibility shims). The old
spelling now raises (``AttributeError``/``TypeError``); the "New name" column
is the spelling to use. Torch-version compatibility and artifact-format
compatibility (tlspec floors, legacy-save loading) are NOT shims and are
unaffected.

## Top-Level Moved Names

| Old name | New name | Status |
| --- | --- | --- |
| `ActivationPostfunc` | none (no longer public) | removed |
| `Buffer` | `torchlens.types.Buffer` | removed |
| `FuncCallLocation` | `torchlens.types.FuncCallLocation` | removed |
| `GradientPostfunc` | none (no longer public) | removed |
| `GradFnAccessor` | none (no longer public) | removed |
| `GradFn` | `torchlens.types.GradFn` | removed |
| `GradFnCall` | `torchlens.types.GradFnCall` | removed |
| `LayerAccessor` | none (no longer public) | removed |
| `MetadataInvariantError` | `torchlens.errors.MetadataInvariantError` | removed |
| `MutatedReferenceError` | `torchlens.errors.MutatedReferenceError` | removed |
| `ModuleAccessor` | none (no longer public) | removed |
| `Module` | `torchlens.types.Module` | removed |
| `ModuleCall` | `torchlens.types.ModuleCall` | removed |
| `NodeSpec` | `torchlens.viz.NodeSpec` | removed |
| `Param` | `torchlens.types.Param` | removed |
| `PostTraceParamUnavailable` | `torchlens.errors.ParamReleasedError` | removed |
| `TraceState` | `torchlens.io.TraceState` | removed |
| `SaveLevel` | none (no longer public) | removed |
| `SiteTable` | none (no longer public) | removed |
| `SpecCompat` | `torchlens.types.SpecCompat` | removed |
| `StreamingOptions` | `torchlens.options.StorageOptions` | removed |
| `TargetManifestDiff` | `torchlens.types.TargetManifestDiff` | removed |
| `TensorLog` | none (no longer public) | removed |
| `TensorSliceSpec` | none (no longer public) | removed |
| `TorchLensPostfuncError` | `torchlens.errors.UserTransformError` | removed |
| `TrainingModeConfigError` | `torchlens.errors.TrainingModeConfigError` | removed |
| `VisualizationOptions` | `torchlens.options.VisualizationOptions` | removed |
| `build_render_audit` | none (no longer public) | removed |
| `check_metadata_invariants` | `torchlens.validation.check_metadata_invariants` | removed |
| `check_spec_compat` | `torchlens.intervention.check_spec_compat` | removed |
| `cleanup_tmp` | `torchlens.io.cleanup_tmp` | removed |
| `get_model_metadata` | `torchlens.trace(model, x, save="none")` | removed |
| `list_logs` | `torchlens.io.list_traces` | removed |
| `log_model_metadata` | `torchlens.trace(model, x, save="none")` | removed |
| `preview_fastlog` | none (no longer public) | removed |
| `rehydrate_nested` | `torchlens.io.materialize_nested` | removed |
| `render_lines_to_html` | none (no longer public) | removed |
| `reset_naming_counter` | `torchlens.io.reset_naming_counter` | removed |
| `resolve_sites` | `torchlens.Trace` indexing (`trace[selector]` returns every matching site as a `torchlens.Selection`) | removed |
| `save_intervention` | `torchlens.io.save_intervention` | removed |
| `suppress_mutate_warnings` | `torchlens.errors.quiet()` | removed |
| `unwrap_torch` | `torchlens.backends.torch.wrappers.unwrap_torch` | removed |
| `validate_batch_of_models_and_inputs` | none: call `torchlens.validate(model, x)` per model | removed |
| `wrap_torch` | `torchlens.backends.torch.wrappers.wrap_torch` | removed |
| `wrapped` | `torchlens.backends.torch.wrappers.wrapped` | removed |

## Top-Level Paper-Era Names

| Old name | New name | Status |
| --- | --- | --- |
| `log_forward_pass` | `trace` | removed |
| `validate_model_activations` | `validate(scope="forward")` | removed |
| `validate_saved_activations` | `validate(scope="saved")` | removed |
| `render_graph` | `Trace.draw()` (or `torchlens.draw`, the top-level spelling below) | removed |
| `render_model_graph` | `Trace.draw()` (or `torchlens.draw`, the top-level spelling below) | removed |
| `draw_model_graph` | `Trace.draw()` (or `torchlens.draw`, the top-level spelling below) | removed |
| `ModelHistory` | `Trace` | removed |
| `get_model_structure` | structure trace accessors | removed |
| `show_model_structure` | structure trace accessors | removed |

## Top-Level Convenience Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `peek` | `pluck` | removed |
| `batched_extract` | `extract_dataset` | removed |
| `validate_forward_pass` | `torchlens.validation.validate_forward_pass` | removed |
| `validate_backward_pass` | `torchlens.validation.validate_backward_pass` | removed |
| `validate_saved_activations` | `torchlens.validate(scope="saved")` | removed |
| `summary` | `torchlens.summary` | removed |
| `show_model_graph` | `torchlens.draw` | removed |
| `draw_backward` | `torchlens.draw` | removed; pass `graph="backward"` |
| `draw_combined` | `torchlens.draw` | removed; pass `graph="combined"` |
| `show_bundle_graph` | `torchlens.Bundle.draw` | removed; call `bundle.draw(path)` (`graph=` replaces `signal=`, `node_overrides=` replaces `vis_node_overrides=`, `open_viewer=True` opens a viewer) |
| `load_intervention_spec` | `torchlens.io.load_intervention_spec` | removed |
| `ModuleInputSnapshot` | `torchlens.types.ModuleInputSnapshot` | removed |
| `PreHookEffect` | `torchlens.types.PreHookEffect` | removed |
| `TensorInputObservation` | `torchlens.types.TensorInputObservation` | removed |

## Capture And Option Keyword Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `capture_output_structure` | `capture_container_structure` | removed |
| `show_containers="auto"` | `show_containers="collapsed"` | removed |
| `show_containers="cluster"` | `show_containers="nodes"` | removed |
| `layers_to_save` | `save=...` | removed |
| `random_seed` | grouped capture options | removed |
| `save_grads` | backward capture options | removed |
| `vis_node_mode` | `node_style` | removed |
| `vis_opt` | `view` (full alias chain `vis_opt` -> `vis_mode` -> `view`; the warning names `view`) | removed |
| flat `CaptureOptions` fields | grouped option fields | removed |
| `mark_layer_depths` | `capture.compute_input_output_distances` | removed |
| `num_context_lines` | `capture.source_context_lines` | removed |
| `mode` | `visualization.view` | removed |
| `max_module_depth` | `visualization.depth` | removed |
| `layout_engine` | `visualization.layout` | removed |
| `node_mode` | `visualization.node_style` | removed |
| `save_outs_to` | `streaming.path` | removed |
| `keep_outs_in_memory` | `streaming.retain_in_memory` | removed |
| `out_sink` | `streaming.on_save` | removed |
| `log_forward_pass(layers=...)` | `trace(save=...)` | removed |
| `log_forward_pass(save_function_args=...)` | `trace(save_args=...)` | removed |
| `log_forward_pass(save_gradients=...)` | `trace(save_grads=...)` | removed |
| `log_forward_pass(keep_unsaved_layers=...)` | no direct equivalent; see migration guide | removed |
| flat visualization option fields | grouped visualization option fields | removed |

## Recording And Fastlog Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `record(keep_op=...)` | `record(save=...)` | REMOVED (predicate consolidation) | removed; raises TypeError |
| `record(keep_module=...)` | no equivalent; see note below | REMOVED (predicate consolidation) | removed; raises TypeError |
| `Recorder(keep_op=...)` | `Recorder(save=...)` | REMOVED (predicate consolidation) | removed; raises TypeError |
| `Recorder(keep_module=...)` | no equivalent; see note below | REMOVED (predicate consolidation) | removed; raises TypeError |
| `dry_run(keep_op=...)` | `dry_run(save=...)` | RENAMED (predicate consolidation; was never deprecation-warned) | removed; raises TypeError |
| `dry_run(keep_module=...)` | no equivalent; see note below | REMOVED (predicate consolidation) | removed; raises TypeError |

**`keep_module` note.** Predicate-gated
module-event selection was REMOVED, not migrated: `save=` routes to the op
predicate only, and `default_module=` records ALL module enter/exit boundary
events uniformly — it is not a per-module predicate. A module predicate passed
via `save=` selects zero module events. The internal `keep_module` options
slot and its two public simulators were deleted with it
(`Trace.preview_fastlog(keep_module=...)` and
`RecordingTrace.repredicate(other_keep_module=...)` no longer accept module
predicates; both raise `TypeError`), so no surface simulates a configuration
that capture cannot produce.

`record(save=...)` and `dry_run(save=...)` now default to `None` (previously an
internal `MISSING` sentinel that existed only to arbitrate against the removed
`keep_op=` alias); omitted and `save=None` are equivalent.

## Trace And Conditional Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `Trace.conditional_then_entry_edges` | `Trace.graph.conditional_arm_entry_edges` | removed |
| `Trace.conditional_elif_entry_edges` | `Trace.graph.conditional_arm_entry_edges` | removed |
| `Trace.conditional_else_entry_edges` | `Trace.graph.conditional_arm_entry_edges` | removed |
| `Trace.validate_saved_outs()` | `Trace.validate_forward_pass()` | removed |
| `Trace.replay()` | `Trace.do(site, edit)` | removed |
| `Trace.replay_from()` | `Trace.do(site, value)` | removed |
| `Trace.push()` | `Trace.do(site, edit)`: one verb stages and propagates | removed |
| `Trace.push_from()` | `Trace.do(site, value)` | removed |
| `Trace.attach_hooks()`, `Op.attach_hooks()`, `Layer.attach_hooks()` | `Trace.do(site, hook)`; `InterventionOptions(engine="set_only")` stages without propagating | removed |
| `Trace.set()`, `Layer.set()`, `Op.set()` | `Trace.do(site, value, intervention=InterventionOptions(engine="set_only"))` | removed |
| `Trace.remove()` and `with trace.attach_hooks(...)` | `Trace.clear_edits(site)`; the scoped form stays as `with trace.fork() as scoped: scoped.do(...)`, which detaches the last `do` hooks on exit | removed |
| `ReplayOptions(hooks=)` | a mapping of sites to edits on `Trace.do` | removed |
| `ReplayOptions(differentiable=)` | `InterventionOptions(differentiable=True)` on `Trace.do` | removed |
| `ReplayOptions(append=)` | `Trace.append(inputs, model=model)` | removed |
| `Trace.rerun()` | `Trace.run()` | removed |

## Record Members

| Old name | New name | Status |
| --- | --- | --- |
| `Op.gpu_kernels` | none in this release (artifacts keep the kernel annotation) | removed |
| `AtenOp.gpu_kernels` | none in this release (artifacts keep the kernel annotation) | removed |
| `Op.activation_memory` | `Op.cost.activation_memory` | removed |
| `Op.arg_expressions` | `Op.source.arg_expressions` | removed |
| `Op.arg_names` | `Op.call.arg_names` | removed |
| `Op.args_template` | `Op.call.args_template` | removed |
| `Op.arithmetic_intensity` | `Op.cost.arithmetic_intensity` | removed |
| `Op.atomic_module` | `Op.module_context.atomic_module` | removed |
| `Op.atomic_module_address` | `Op.module_context.atomic_module_address` | removed |
| `Op.atomic_module_call` | `Op.module_context.atomic_module_call` | removed |
| `Op.autograd_memory` | `Op.cost.autograd_memory` | removed |
| `Op.bool_value` | `Op.control_flow.bool_value` | removed |
| `Op.buffer_sink_ops` | `Op.graph.buffer_sink_ops` | removed |
| `Op.buffer_source` | `Op.graph.buffer_source` | removed |
| `Op.buffer_source_func_name` | `Op.graph.buffer_source_func_name` | removed |
| `Op.buffer_source_ops` | `Op.graph.buffer_source_ops` | removed |
| `Op.bytes_delta_at_call` | `Op.cost.bytes_delta_at_call` | removed |
| `Op.bytes_peak_at_call` | `Op.cost.bytes_peak_at_call` | removed |
| `Op.bytes_read` | `Op.cost.bytes_read` | removed |
| `Op.bytes_written` | `Op.cost.bytes_written` | removed |
| `Op.co_parents` | `Op.graph.co_parents` | removed |
| `Op.code_context` | `Op.source.frames` | removed |
| `Op.compute_record` | `Op.cost.compute_record` | removed |
| `Op.conditional_arm_children` | `Op.control_flow.arm_children` | removed |
| `Op.conditional_branch_depth` | `Op.control_flow.branch_depth` | removed |
| `Op.conditional_branch_stack` | `Op.control_flow.branch_stack` | removed |
| `Op.conditional_context_kind` | `Op.control_flow.context_kind` | removed |
| `Op.conditional_depth` | `Op.control_flow.depth` | removed |
| `Op.conditional_elif_children` | `Op.control_flow.elif_children` | removed |
| `Op.conditional_else_children` | `Op.control_flow.else_children` | removed |
| `Op.conditional_entry_children` | `Op.control_flow.entry_children` | removed |
| `Op.conditional_then_children` | `Op.control_flow.then_children` | removed |
| `Op.conditional_wrapper_kind` | `Op.control_flow.wrapper_kind` | removed |
| `Op.derived_grad` | `Op.autograd.derived_grad` | removed |
| `Op.dropped_edge_tensor_args` | `Op.call.dropped_edge_tensor_args` | removed |
| `Op.edge_substitutions` | `Op.graph.edge_substitutions` | removed |
| `Op.equivalence_class` | `Op.graph.equivalence_class` | removed |
| `Op.equivalent_ops` | `Op.graph.equivalent_ops` | removed |
| `Op.flops_backward` | `Op.cost.flops_backward` | removed |
| `Op.flops_forward` | `Op.cost.flops_forward` | removed |
| `Op.flops_total` | `Op.cost.flops_total` | removed |
| `Op.func_autocast_state` | `Op.call.autocast_state` | removed |
| `Op.func_call_id` | `Op.call.call_id` | removed |
| `Op.func_config` | `Op.call.config` | removed |
| `Op.func_duration` | `Op.cost.duration` | removed |
| `Op.func_id` | `Op.call.func_id` | removed |
| `Op.func_non_tensor_args` | `Op.call.non_tensor_args` | removed |
| `Op.func_qualname` | `Op.source.func_qualname` | removed |
| `Op.func_rng_states` | `Op.call.rng_states` | removed |
| `Op.fx_call_index` | `Op.source.fx_call_index` | removed |
| `Op.fx_qualpath` | `Op.source.fx_path` | removed |
| `Op.grad_dtype` | `Op.autograd.grad_dtype` | removed |
| `Op.grad_fn` | `Op.autograd.grad_fn` | removed |
| `Op.grad_fn_class_name` | `Op.autograd.grad_fn_class_name` | removed |
| `Op.grad_fn_class_qualname` | `Op.autograd.grad_fn_class_qualname` | removed |
| `Op.grad_fn_handle` | `Op.autograd.grad_fn_handle` | removed |
| `Op.grad_for` | `Op.autograd.grad_for` | removed |
| `Op.grad_shape` | `Op.autograd.grad_shape` | removed |
| `Op.grads` | `Op.autograd.grads` | removed |
| `Op.has_children` | `Op.graph.has_children` | removed |
| `Op.has_co_parents` | `Op.graph.has_co_parents` | removed |
| `Op.has_input_ancestor` | `Op.graph.has_input_ancestor` | removed |
| `Op.has_internal_source_ancestor` | `Op.graph.has_internal_source_ancestor` | removed |
| `Op.has_output_descendant` | `Op.graph.has_output_descendant` | removed |
| `Op.has_parents` | `Op.graph.has_parents` | removed |
| `Op.has_saved_args` | `Op.call.has_saved_args` | removed |
| `Op.has_siblings` | `Op.graph.has_siblings` | removed |
| `Op.in_conditionals` | `Op.control_flow.in_conditionals` | removed |
| `Op.injection_provenance` | `Op.call.injection_provenance` | removed |
| `Op.input_memory` | `Op.cost.input_memory` | removed |
| `Op.input_to_module_calls` | `Op.module_context.input_to_module_calls` | removed |
| `Op.input_to_modules` | `Op.module_context.input_to_modules` | removed |
| `Op.internal_source_ancestors` | `Op.graph.internal_source_ancestors` | removed |
| `Op.internal_source_parents` | `Op.graph.internal_source_parents` | removed |
| `Op.io_role` | `Op.graph.io_role` | removed |
| `Op.is_atomic_module` | `Op.module_context.is_atomic_module` | removed |
| `Op.is_buffer` | `Op.graph.is_buffer` | removed |
| `Op.is_compute_op` | `Op.graph.is_compute` | removed |
| `Op.is_final_output` | `Op.graph.is_final_output` | removed |
| `Op.is_in_conditional` | `Op.control_flow.is_in_conditional` | removed |
| `Op.is_in_conditional_body` | `Op.control_flow.is_in_conditional_body` | removed |
| `Op.is_in_conditional_evaluation` | `Op.control_flow.is_in_conditional_evaluation` | removed |
| `Op.is_internal_source` | `Op.graph.is_internal_source` | removed |
| `Op.is_internally_initialized` | `Op.call.is_internally_initialized` | removed |
| `Op.is_module_input` | `Op.module_context.is_module_input` | removed |
| `Op.is_module_output` | `Op.module_context.is_module_output` | removed |
| `Op.is_orphan` | `Op.graph.is_orphan` | removed |
| `Op.is_output_parent` | `Op.graph.is_output_parent` | removed |
| `Op.is_scalar_bool` | `Op.control_flow.is_scalar_bool` | removed |
| `Op.is_terminal_bool` | `Op.control_flow.is_terminal_bool` | removed |
| `Op.is_terminal_conditional_bool` | `Op.control_flow.is_terminal_conditional_bool` | removed |
| `Op.is_transform` | `Op.call.is_func_transform` | removed |
| `Op.kwargs_template` | `Op.call.kwargs_template` | removed |
| `Op.macs_forward` | `Op.cost.macs_forward` | removed |
| `Op.materialize_grad` | `Op.payload.materialize_grad` | removed |
| `Op.materialize_out` | `Op.payload.materialize_out` | removed |
| `Op.max_distance_from_input` | `Op.graph.max_distance_from_input` | removed |
| `Op.max_distance_to_output` | `Op.graph.max_distance_to_output` | removed |
| `Op.min_distance_from_input` | `Op.graph.min_distance_from_input` | removed |
| `Op.min_distance_to_output` | `Op.graph.min_distance_to_output` | removed |
| `Op.module_call_depth` | `Op.module_context.call_depth` | removed |
| `Op.module_call_stack` | `Op.module_context.call_stack` | removed |
| `Op.module_entry_arg_keys` | `Op.module_context.entry_arg_keys` | removed |
| `Op.non_tensor_kwargs` | `Op.call.non_tensor_kwargs` | removed |
| `Op.non_tensor_pos_args` | `Op.call.non_tensor_pos_args` | removed |
| `Op.num_args_total` | `Op.call.num_args_total` | removed |
| `Op.num_autograd_tensors` | `Op.autograd.num_autograd_tensors` | removed |
| `Op.num_inputs` | `Op.graph.num_inputs` | removed |
| `Op.num_kwargs` | `Op.call.num_kwargs` | removed |
| `Op.num_pos_args` | `Op.call.num_pos_args` | removed |
| `Op.output_of_module_calls` | `Op.module_context.output_of_module_calls` | removed |
| `Op.output_of_modules` | `Op.module_context.output_of_modules` | removed |
| `Op.param_memory` | `Op.cost.param_memory` | removed |
| `Op.parent_arg_positions` | `Op.call.parent_arg_positions` | removed |
| `Op.parent_param_ops` | `Op.graph.parent_param_ops` | removed |
| `Op.parent_params` | `Op.graph.parent_params` | removed |
| `Op.recurrent_ops` | `Op.graph.recurrent_ops` | removed |
| `Op.root_ancestors` | `Op.graph.root_ancestors` | removed |
| `Op.saved_args` | `Op.call.saved_args` | removed |
| `Op.saved_kwargs` | `Op.call.saved_kwargs` | removed |
| `Op.siblings` | `Op.graph.siblings` | removed |
| `Op.terminal_bool_for` | `Op.control_flow.terminal_bool_for` | removed |
| `Op.tl_authored_root` | `Op.call.tl_authored_root` | removed |
| `Op.transform_chain` | `Op.call.func_transform_chain` | removed |
| `Op.transform_config` | `Op.call.func_transform_config` | removed |
| `Op.transform_fn_name` | `Op.call.func_transform_fn_name` | removed |
| `Op.transform_fn_qualname` | `Op.call.func_transform_fn_qualname` | removed |
| `Op.transform_fn_source` | `Op.call.func_transform_fn_source` | removed |
| `Op.transform_kind` | `Op.call.func_transform_kind` | removed |
| `Op.transformed_activation_memory` | `Op.cost.transformed_activation_memory` | removed |
| `Op.transformed_grad` | `Op.payload.transformed_grad` | removed |
| `Op.transformed_grad_dtype` | `Op.payload.transformed_grad_dtype` | removed |
| `Op.transformed_grad_shape` | `Op.payload.transformed_grad_shape` | removed |
| `Op.transformed_out` | `Op.payload.transformed_out` | removed |
| `Op.transformed_out_dtype` | `Op.payload.transformed_out_dtype` | removed |
| `Op.transformed_out_shape` | `Op.payload.transformed_out_shape` | removed |
| `Op.unattributed_tensor_args` | `Op.call.unattributed_tensor_args` | removed |
| `Op.var_names` | `Op.source.var_names` | removed |
| `Layer.activation_memory` | `Layer.cost.activation_memory` | removed |
| `Layer.arg_names` | `Layer.call.arg_names` | removed |
| `Layer.autograd_memory` | `Layer.cost.autograd_memory` | removed |
| `Layer.bool_value` | `Layer.control_flow.bool_value` | removed |
| `Layer.buffer_pass` | `Layer.graph.buffer_pass` | removed |
| `Layer.buffer_source` | `Layer.graph.buffer_source` | removed |
| `Layer.buffer_source_func_name` | `Layer.graph.buffer_source_func_name` | removed |
| `Layer.child_ops_per_layer` | `Layer.graph.child_ops_per_layer` | removed |
| `Layer.children_per_pass` | `Layer.graph.children_per_pass` | removed |
| `Layer.code_context` | `Layer.source.frames` | removed |
| `Layer.conditional_arm_children` | `Layer.control_flow.arm_children` | removed |
| `Layer.conditional_branch_stack_ops` | `Layer.control_flow.branch_stack_ops` | removed |
| `Layer.conditional_depth` | `Layer.control_flow.depth` | removed |
| `Layer.conditional_elif_children` | `Layer.control_flow.elif_children` | removed |
| `Layer.conditional_else_children` | `Layer.control_flow.else_children` | removed |
| `Layer.conditional_entry_children` | `Layer.control_flow.entry_children` | removed |
| `Layer.conditional_role_stacks` | `Layer.control_flow.role_stacks` | removed |
| `Layer.conditional_then_children` | `Layer.control_flow.then_children` | removed |
| `Layer.equivalence_class` | `Layer.graph.equivalence_class` | removed |
| `Layer.equivalent_ops` | `Layer.graph.equivalent_ops` | removed |
| `Layer.flops_backward` | `Layer.cost.flops_backward` | removed |
| `Layer.flops_forward` | `Layer.cost.flops_forward` | removed |
| `Layer.flops_total` | `Layer.cost.flops_total` | removed |
| `Layer.func_config` | `Layer.call.config` | removed |
| `Layer.func_duration` | `Layer.cost.duration` | removed |
| `Layer.func_rng_states` | `Layer.call.rng_states` | removed |
| `Layer.fx_call_index` | `Layer.source.fx_call_index` | removed |
| `Layer.fx_qualpath` | `Layer.source.fx_path` | removed |
| `Layer.grad_fn` | `Layer.autograd.grad_fn` | removed |
| `Layer.grad_fn_class_name` | `Layer.autograd.grad_fn_class_name` | removed |
| `Layer.grad_fn_class_qualname` | `Layer.autograd.grad_fn_class_qualname` | removed |
| `Layer.grad_fn_handle` | `Layer.autograd.grad_fn_handle` | removed |
| `Layer.has_children` | `Layer.graph.has_children` | removed |
| `Layer.has_co_parents` | `Layer.graph.has_co_parents` | removed |
| `Layer.has_input_ancestor` | `Layer.graph.has_input_ancestor` | removed |
| `Layer.has_parents` | `Layer.graph.has_parents` | removed |
| `Layer.has_siblings` | `Layer.graph.has_siblings` | removed |
| `Layer.in_conditionals` | `Layer.control_flow.in_conditionals` | removed |
| `Layer.io_role` | `Layer.graph.io_role` | removed |
| `Layer.is_atomic_module` | `Layer.module_context.is_atomic_module` | removed |
| `Layer.is_buffer` | `Layer.graph.is_buffer` | removed |
| `Layer.is_compute_layer` | `Layer.graph.is_compute` | removed |
| `Layer.is_final_output` | `Layer.graph.is_final_output` | removed |
| `Layer.is_in_conditional` | `Layer.control_flow.is_in_conditional` | removed |
| `Layer.is_in_conditional_body` | `Layer.control_flow.is_in_conditional_body` | removed |
| `Layer.is_in_conditional_evaluation` | `Layer.control_flow.is_in_conditional_evaluation` | removed |
| `Layer.is_internal_sink` | `Layer.graph.is_internal_sink` | removed |
| `Layer.is_internal_source` | `Layer.graph.is_internal_source` | removed |
| `Layer.is_module_input` | `Layer.module_context.is_module_input` | removed |
| `Layer.is_orphan` | `Layer.graph.is_orphan` | removed |
| `Layer.is_scalar_bool` | `Layer.control_flow.is_scalar_bool` | removed |
| `Layer.is_terminal_bool` | `Layer.control_flow.is_terminal_bool` | removed |
| `Layer.macs_forward` | `Layer.cost.macs_forward` | removed |
| `Layer.module_call_depth` | `Layer.module_context.call_depth` | removed |
| `Layer.num_args_total` | `Layer.call.num_args_total` | removed |
| `Layer.num_autograd_tensors` | `Layer.autograd.num_autograd_tensors` | removed |
| `Layer.num_kwargs` | `Layer.call.num_kwargs` | removed |
| `Layer.num_pos_args` | `Layer.call.num_pos_args` | removed |
| `Layer.output_of_module_calls` | `Layer.module_context.output_of_module_calls` | removed |
| `Layer.output_of_modules` | `Layer.module_context.output_of_modules` | removed |
| `Layer.parent_arg_positions` | `Layer.call.parent_arg_positions` | removed |
| `Layer.parent_ops_per_layer` | `Layer.graph.parent_ops_per_layer` | removed |
| `Layer.parents_per_pass` | `Layer.graph.parents_per_pass` | removed |
| `Layer.saved_args` | `Layer.call.saved_args` | removed |
| `Layer.saved_kwargs` | `Layer.call.saved_kwargs` | removed |
| `Layer.site_peers` | `Layer.graph.site_peers` | removed |
| `Layer.terminal_bool_for` | `Layer.control_flow.terminal_bool_for` | removed |
| `Layer.total_activation_memory` | `Layer.cost.total_activation_memory` | removed |
| `Layer.total_autograd_memory` | `Layer.cost.total_autograd_memory` | removed |
| `Layer.total_flops_backward` | `Layer.cost.total_flops_backward` | removed |
| `Layer.total_flops_forward` | `Layer.cost.total_flops_forward` | removed |
| `Layer.total_flops_total` | `Layer.cost.total_flops` | removed |
| `Layer.total_func_duration` | `Layer.cost.total_func_duration` | removed |
| `Layer.total_macs_forward` | `Layer.cost.total_macs_forward` | removed |
| `Layer.total_macs_total` | `Layer.cost.total_macs` | removed |
| `Layer.total_param_memory` | `Layer.cost.total_param_memory` | removed |
| `Layer.transformed_activation_memory` | `Layer.cost.transformed_activation_memory` | removed |
| `Layer.transformed_grad` | `Layer.payload.transformed_grad` | removed |
| `Layer.transformed_grad_dtype` | `Layer.payload.transformed_grad_dtype` | removed |
| `Layer.transformed_grad_shape` | `Layer.payload.transformed_grad_shape` | removed |
| `Layer.transformed_out` | `Layer.payload.transformed_out` | removed |
| `Layer.transformed_out_dtype` | `Layer.payload.transformed_out_dtype` | removed |
| `Layer.transformed_out_shape` | `Layer.payload.transformed_out_shape` | removed |
| `Op.address` | `Op.buffer_address` | removed |
| `Op.activation_transform` | `trace.options.activation_transform` (a capture setting) | removed |
| `Op.detach_saved_activations` | `trace.options.detach_saved_activations` (a capture setting) | removed |
| `Op.output_device` | `trace.options.output_device` (a capture setting) | removed |
| `Op.grad_fn_object_id` | `Op.autograd.grad_fn` | removed |
| `Layer.address` | `Layer.buffer_address` | removed |
| `Layer.activation_transform` | `trace.options.activation_transform` (a capture setting) | removed |
| `Layer.detach_saved_activations` | `trace.options.detach_saved_activations` (a capture setting) | removed |
| `Layer.output_device` | `trace.options.output_device` (a capture setting) | removed |
| `Layer.grad_fn_object_id` | `Layer.autograd.grad_fn` | removed |
| `Op.out_ref` | `Op.out` / `Op.payload.materialize_out()` | removed |
| `Op.grad_ref` | `Op.grad` / `Op.payload.materialize_grad()` | removed |
| `Op.grad_fn_cls` | `Op.autograd.grad_fn_class_name` | removed |
| `BackendSpec.maturity` | none (no backend carries a maturity label; the backend table in `docs/backends.md` lists features) | removed |
| `BackendProtocol.maturity` | none (as above) | removed |

## Module Paths

A submodule is a public import path only when its parent lists it in `__all__` or it is taught (`torchlens.backends.torch`, `torchlens.backends.registry`, `torchlens.intervention.errors`, `torchlens.intervention.hooks`, `torchlens.intervention.stochastic` and the per-framework backend namespaces `torchlens.backends.jax`, `mlx`, `paddle`, `tf` and `tinygrad`). Reading an old spelling as an attribute of its parent (`tl.intervention.types`) raises the typed `facade_redirect` error; an `import` statement of an old path fails with `ModuleNotFoundError`.

| Old name | New name | Status |
| --- | --- | --- |
| `torchlens.agent.cli` | none public; import its names from `torchlens.agent` (private home `torchlens.agent._cli`) | private |
| `torchlens.attribution.graph` | none public; import its names from `torchlens.attribution` (private home `torchlens.attribution._graph`) | private |
| `torchlens.attribution.onebackward` | none public; import its names from `torchlens.attribution` (private home `torchlens.attribution._onebackward`) | private |
| `torchlens.backends.default_specs` | none public; import its names from `torchlens.backends` (private home `torchlens.backends._default_specs`) | private |
| `torchlens.bridge.hf_attention` | none public; import its names from `torchlens.bridge` (private home `torchlens.bridge._hf_attention`) | private |
| `torchlens.errors.episode` | none public; import its names from `torchlens.errors` (private home `torchlens.errors._episode`) | private |
| `torchlens.errors.runnable` | none public; import its names from `torchlens.errors` (private home `torchlens.errors._runnable`) | private |
| `torchlens.intervention.audit` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._audit`) | private |
| `torchlens.intervention.binding` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._binding`) | private |
| `torchlens.intervention.bundle` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._bundle`) | private |
| `torchlens.intervention.edge_semantics` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._edge_semantics`) | private |
| `torchlens.intervention.edge_substitution` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._edge_substitution`) | private |
| `torchlens.intervention.handles` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._handles`) | private |
| `torchlens.intervention.helpers` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._helpers`) | private |
| `torchlens.intervention.injection` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._injection`) | private |
| `torchlens.intervention.masked_edit` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._masked_edit`) | private |
| `torchlens.intervention.model_door` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._model_door`) | private |
| `torchlens.intervention.param_substitution` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._param_substitution`) | private |
| `torchlens.intervention.predicates` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._predicates`) | private |
| `torchlens.intervention.regions` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._regions`) | private |
| `torchlens.intervention.replay` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._replay`) | private |
| `torchlens.intervention.rerun` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._rerun`) | private |
| `torchlens.intervention.runtime` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._runtime`) | private |
| `torchlens.intervention.save` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._save`) | private |
| `torchlens.intervention.site_keys` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._site_keys`) | private |
| `torchlens.intervention.spec` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._spec`) | private |
| `torchlens.intervention.spec_compat` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._spec_compat`) | private |
| `torchlens.intervention.sweep` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._sweep`) | private |
| `torchlens.intervention.types` | none public; import its names from `torchlens.intervention` (private home `torchlens.intervention._types`) | private |
| `torchlens.recording.cleanup` | none public; import its names from `torchlens.recording` (private home `torchlens.recording._cleanup`) | private |
| `torchlens.recording.exceptions` | none public; import its names from `torchlens.recording` (private home `torchlens.recording._exceptions`) | private |
| `torchlens.recording.options` | none public; import its names from `torchlens.recording` (private home `torchlens.recording._options`) | private |
| `torchlens.recording.storage_disk` | none public; import its names from `torchlens.recording` (private home `torchlens.recording._storage_disk`) | private |
| `torchlens.recording.storage_ram` | none public; import its names from `torchlens.recording` (private home `torchlens.recording._storage_ram`) | private |
| `torchlens.recording.types` | none public; import its names from `torchlens.recording` (private home `torchlens.recording._types`) | private |
| `torchlens.validation.backward` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._backward`) | private |
| `torchlens.validation.consolidated` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._consolidated`) | private |
| `torchlens.validation.core` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._core`) | private |
| `torchlens.validation.diagnostics` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._diagnostics`) | private |
| `torchlens.validation.exemptions` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._exemptions`) | private |
| `torchlens.validation.invariants` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._invariants`) | private |
| `torchlens.validation.status` | none public; import its names from `torchlens.validation` (private home `torchlens.validation._status`) | private |
| `torchlens.viz.batch_summary` | none public; import its names from `torchlens.viz` (private home `torchlens.viz._batch_summary`) | private |
| `torchlens.viz.node_plots` | none public; import its names from `torchlens.viz` (private home `torchlens.viz._node_plots`) | private |
| `torchlens.intervention.compose` (the module) | `torchlens.intervention.compose` is the public function or object; the module is `torchlens.intervention._compose` | private |
| `torchlens.intervention.population` (the module) | `torchlens.intervention.population` is the public function or object; the module is `torchlens.intervention._population` | private |
| `torchlens.receptive_field.rules` (the module) | `torchlens.receptive_field.rules` is the public function or object; the module is `torchlens.receptive_field._builtin_rules` | private |
| `torchlens.recording.dry_run` (the module) | `torchlens.recording.dry_run` is the public function or object; the module is `torchlens.recording._dry_run_door` | private |
| `torchlens.recording.recover` (the module) | `torchlens.recording.recover` is the public function or object; the module is `torchlens.recording._recover` | private |
| `torchlens.viz.feature_maps` (the module) | `torchlens.viz.feature_maps` is the public function or object; the module is `torchlens.viz._feature_maps` | private |

## Intervention, Sweep, And Observer Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `replay()` | `do(site, edit)` | removed |
| `replay_from()` | `do(site, value)` | removed |
| `rerun()` | `run()` | removed |
| `Bundle.replay()` | `Bundle.do(site, edit)` | removed |
| `Bundle.rerun()` | `Bundle.run()` | removed |
| `torchlens.bundle.AmbiguousLabelError` | `torchlens.errors.AmbiguousLabelError` | removed |
| `Bundle.measure_members(metric=, order=)` | `Bundle.map(fn)`: live members by name, released candidates read `"unmeasured"` | removed |
| `intervening()` | `without_op()` | removed |
| `sweep(param=...)` | `sweep(at=...)` | removed |
| `record_span()` | `span()` | removed |
| `get_model_metadata()` | `torchlens.trace(model, x, save="none")` | removed |

## Gradient Edits (gradient sites)

The site, not a keyword or a gradient-only helper, says which tensor an edit applies to:
`tl.grad(site)` addresses the gradient of the site's output. Removed with no alias:

| Old name | New name | Status |
| --- | --- | --- |
| `tl.zero_grad()` | `tl.zero_ablate()` on `tl.grad(site)` | removed |
| `tl.scale_grad(f)` | `tl.scale(f)` on `tl.grad(site)` | removed |
| `tl.clamp_grad(lo, hi)` | `tl.clamp(low=lo, high=hi)` on `tl.grad(site)` | removed |
| `tl.add_grad_noise(std, seed=)` | `tl.add_noise(std, seed=)` on `tl.grad(site)` | removed |
| `tl.clip_grad(max_norm)` | `tl.clip_norm(max_norm)` on `tl.grad(site)` | removed |
| `tl.hook_grad(fn)` | `fn` on `tl.grad(site)` | removed |
| `direction=` on `Trace.do`, `tl.when`, `tl.observers.tap`, `tl.stats.narrate`, `tl.sweep` | the site: `tl.grad(site)`, or `tl.grad(site) \| site` for both | removed |
| `tl.stats.aggregate(target=)` | metric keys are sites (`tl.grad(site)` for a gradient) | removed |
| `tl.observers.span(direction=)` | `span(signal=)` | removed |
| `show_bundle_graph(direction=)` | `show_bundle_graph(signal=)` | removed |
| `FireRecord.direction`, `TapRecord.direction`, `HookContext.direction` | `.signal` | removed |
| `Edit.kind`, `Edit.direction` | `Edit.signal`; `Rule.signal` (derived) | removed |
| `Rule.direction`, `Rule(direction=)` | the target decides the pass: `Rule(tl.grad(site), edit)` for the gradient, `Rule(site, edit)` for the activation; `Rule.signal` (derived) | removed |
| `HookContext.ctx`, `HookContext.run_ctx` | `HookContext.state`, `HookContext.run_state` | removed |
| `torchlens.intervention.sites` (`sites()`, `SiteCollection`, `SiteSpec`) | pass selectors to `torchlens.experiment.sweep_sites(...)` directly | removed |

## Experiment Ledger

The opt-in semantic record (hypothesis, observations, verdicts in a
`.tlledger` file) was removed on 1 Nov 2026. The Bundle is the record that
stays: `torchlens.experiment.sweep_sites` returns one whose per-candidate
effect table (`bundle.effects()`) and operation chronology (`bundle.history`)
persist with the artifact. Each old spelling raises the typed
`facade_redirect` refusal naming that record; there is no alias.

| Old name | New name | Status |
| --- | --- | --- |
| `tl.experiment.ledger(path, hypothesis=, metric=)` | your own record beside the Bundle (`bundle.effects()`, `bundle.history`) | removed |
| `tl.experiment.ExperimentLedger` | none (the Bundle is the record) | removed |
| `tl.experiment.LedgerEntry` | none | removed |
| `tl.experiment.active_ledger()` | none | removed |
| `tl.experiment.read_ledger(path)` | `tl.load(path).effects()` on the saved Bundle | removed |
| MCP `torchlens_ledger_overview` / `torchlens_ledger_entry` / `torchlens_ledger_evidence` | `torchlens_overview` on the saved Bundle artifact | removed |
| `ledger(on_record_error=)` and the `ledger_*` refusal and warning codes | none | removed |
