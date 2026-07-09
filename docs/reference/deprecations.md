# TorchLens Deprecation Ledger

This ledger tracks warning-producing compatibility shims. Where the exact first
release is not encoded in source, the since-version is recorded conservatively as
`2.x compatibility shim`. Planned removal is `2.0 API freeze - TBD by maintainer`
unless a narrower policy is later set.

## Top-Level Moved Names

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `ActivationPostfunc` | `torchlens.types.ActivationPostfunc` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Buffer` | `torchlens.types.Buffer` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `FuncCallLocation` | `torchlens.types.FuncCallLocation` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `GradientPostfunc` | `torchlens.types.GradientPostfunc` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `GradFnAccessor` | `torchlens.accessors.GradFnAccessor` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `GradFn` | `torchlens.types.GradFn` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `GradFnCall` | `torchlens.types.GradFnCall` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `LayerAccessor` | `torchlens.accessors.LayerAccessor` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `MetadataInvariantError` | `torchlens.errors.MetadataInvariantError` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `MutatedReferenceError` | `torchlens.errors.MutatedReferenceError` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `ModuleAccessor` | `torchlens.accessors.ModuleAccessor` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Module` | `torchlens.types.Module` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `ModuleCall` | `torchlens.types.ModuleCall` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `NodeSpec` | `torchlens.experimental.dagua.NodeSpec` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Param` | `torchlens.types.Param` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `PostTraceParamUnavailable` | `torchlens.errors.PostTraceParamUnavailable` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `TraceState` | `torchlens.io.TraceState` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `SaveLevel` | `torchlens.types.SaveLevel` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `SiteTable` | `torchlens.types.SiteTable` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `SpecCompat` | `torchlens.types.SpecCompat` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `StreamingOptions` | `torchlens.options.StreamingOptions` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `TargetManifestDiff` | `torchlens.types.TargetManifestDiff` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `TensorLog` | `torchlens.types.TensorLog` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `TensorSliceSpec` | `torchlens.types.TensorSliceSpec` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `TorchLensPostfuncError` | `torchlens.errors.TorchLensPostfuncError` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `TrainingModeConfigError` | `torchlens.errors.TrainingModeConfigError` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `VisualizationOptions` | `torchlens.options.VisualizationOptions` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `build_render_audit` | `torchlens.experimental.dagua.build_render_audit` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `check_metadata_invariants` | `torchlens.validation.check_metadata_invariants` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `check_spec_compat` | `torchlens.validation.check_spec_compat` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `cleanup_tmp` | `torchlens.io.cleanup_tmp` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `get_model_metadata` | `torchlens.io.get_model_metadata` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `list_logs` | `torchlens.io.list_logs` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `log_model_metadata` | `torchlens.io.log_model_metadata` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `trace_to_dagua_graph` | `torchlens.experimental.dagua.trace_to_dagua_graph` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `preview_fastlog` | `torchlens.fastlog.preview` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `rehydrate_nested` | `torchlens.io.rehydrate_nested` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `render_lines_to_html` | `torchlens.experimental.dagua.render_lines_to_html` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `render_trace_with_dagua` | `torchlens.experimental.dagua.render_trace_with_dagua` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `reset_naming_counter` | `torchlens.io.reset_naming_counter` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `resolve_sites` | `torchlens.validation.resolve_sites` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `save_intervention` | `torchlens.io.save_intervention` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `suppress_mutate_warnings` | `torchlens.io.suppress_mutate_warnings` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `unwrap_torch` | `torchlens.backends.torch.wrappers.unwrap_torch` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `validate_batch_of_models_and_inputs` | `torchlens.validation.validate_batch_of_models_and_inputs` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `wrap_torch` | `torchlens.backends.torch.wrappers.wrap_torch` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `wrapped` | `torchlens.backends.torch.wrappers.wrapped` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |

## Top-Level Paper-Era Names

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `log_forward_pass` | `trace` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `get_model_activations` | `extract` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `validate_model_activations` | `validate(scope="forward")` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `validate_saved_activations` | `validate(scope="saved")` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `render_graph` | `Trace.draw()` or `show_model_graph()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `render_model_graph` | `Trace.draw()` or `show_model_graph()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `draw_model_graph` | `Trace.draw()` or `show_model_graph()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `ModelHistory` | `Trace` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `get_model_structure` | structure trace accessors | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `show_model_structure` | structure trace accessors | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |

## Top-Level Convenience Aliases

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `peek` | `pluck` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `batched_extract` | `extract_dataset` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `validate_forward_pass` | `torchlens.validation.validate_forward_pass` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `validate_backward_pass` | `torchlens.validation.validate_backward_pass` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `validate_saved_outs` | `torchlens.validation.validate_saved_outs` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `summary` | `torchlens.visualization.summary` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `show_model_graph` | `torchlens.visualization.show_model_graph` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `draw_backward` | `torchlens.visualization.draw_backward` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `draw_combined` | `torchlens.visualization.draw_combined` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `load_intervention_spec` | `torchlens.io.load_intervention_spec` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |

## Capture And Option Keyword Aliases

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `capture_output_structure` | `capture_container_structure` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `layers_to_save` | `save` or grouped capture options | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `random_seed` | grouped capture options | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `save_grads` | backward capture options | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `vis_node_mode` | `node_style` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `vis_opt` | `vis_mode` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| flat `CaptureOptions` fields | grouped option fields | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `mark_layer_depths` | `capture.compute_input_output_distances` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `num_context_lines` | `capture.source_context_lines` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `mode` | `visualization.view` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `max_module_depth` | `visualization.depth` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `layout_engine` | `visualization.layout` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `node_mode` | `visualization.node_style` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `save_outs_to` | `streaming.bundle_path` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `keep_outs_in_memory` | `streaming.retain_in_memory` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `out_sink` | `streaming.out_callback` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| flat visualization option fields | grouped visualization option fields | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |

## Recording And Fastlog Aliases

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `record(keep_op=...)` | `record(save=...)` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `record(keep_module=...)` | `record(save=...)` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Recorder(keep_op=...)` | `Recorder(save=...)` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Recorder(keep_module=...)` | `Recorder(save=...)` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |

## Trace And Conditional Aliases

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `Trace.conditional_then_entry_edges` | `Trace.conditional_arm_entry_edges` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Trace.conditional_elif_entry_edges` | `Trace.conditional_arm_entry_edges` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Trace.conditional_else_entry_edges` | `Trace.conditional_arm_entry_edges` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Trace.validate_saved_outs()` | `Trace.validate_forward_pass()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Trace.replay()` | `Trace.push()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Trace.replay_from()` | `Trace.push_from()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Trace.rerun()` | `Trace.run()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |

## Intervention, Sweep, And Observer Aliases

| Old name | New name | Since-version | Planned removal |
| --- | --- | --- | --- |
| `replay()` | `push()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `replay_from()` | `push_from()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `rerun()` | `run()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Bundle.replay()` | `Bundle.push()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `Bundle.rerun()` | `Bundle.run()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `intervening()` | `without_op()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `sweep(param=...)` | `sweep(at=...)` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `record_span()` | `span()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
| `get_model_metadata()` | `log_model_metadata()` | 2.x compatibility shim | 2.0 API freeze - TBD by maintainer |
