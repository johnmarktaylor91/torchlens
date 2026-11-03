# TorchLens Removed-Spellings Ledger

Every warning-producing compatibility shim listed below was REMOVED outright in
the 2026-08 shim-removal pass (interim-phase ruling: deprecation shims are not
justified before launch; policy is remove-and-rename, not shim). The old
spelling now raises (``AttributeError``/``TypeError``); the "New name" column
is the spelling to use. Torch-version compatibility and artifact-format
compatibility (tlspec floors, legacy-save loading) are NOT shims and are
unaffected.

## Top-Level Moved Names

| Old name | New name | Status |
| --- | --- | --- |
| `ActivationPostfunc` | `torchlens.types.ActivationPostfunc` | removed |
| `Buffer` | `torchlens.types.Buffer` | removed |
| `FuncCallLocation` | `torchlens.types.FuncCallLocation` | removed |
| `GradientPostfunc` | `torchlens.types.GradientPostfunc` | removed |
| `GradFnAccessor` | `torchlens.accessors.GradFnAccessor` | removed |
| `GradFn` | `torchlens.types.GradFn` | removed |
| `GradFnCall` | `torchlens.types.GradFnCall` | removed |
| `LayerAccessor` | `torchlens.accessors.LayerAccessor` | removed |
| `MetadataInvariantError` | `torchlens.errors.MetadataInvariantError` | removed |
| `MutatedReferenceError` | `torchlens.errors.MutatedReferenceError` | removed |
| `ModuleAccessor` | `torchlens.accessors.ModuleAccessor` | removed |
| `Module` | `torchlens.types.Module` | removed |
| `ModuleCall` | `torchlens.types.ModuleCall` | removed |
| `NodeSpec` | `torchlens.experimental.dagua.NodeSpec` | removed |
| `Param` | `torchlens.types.Param` | removed |
| `PostTraceParamUnavailable` | `torchlens.errors.PostTraceParamUnavailable` | removed |
| `TraceState` | `torchlens.io.TraceState` | removed |
| `SaveLevel` | `torchlens.types.SaveLevel` | removed |
| `SiteTable` | `torchlens.types.SiteTable` | removed |
| `SpecCompat` | `torchlens.types.SpecCompat` | removed |
| `StreamingOptions` | `torchlens.options.StreamingOptions` | removed |
| `TargetManifestDiff` | `torchlens.types.TargetManifestDiff` | removed |
| `TensorLog` | `torchlens.types.TensorLog` | removed |
| `TensorSliceSpec` | `torchlens.types.TensorSliceSpec` | removed |
| `TorchLensPostfuncError` | `torchlens.errors.TorchLensPostfuncError` | removed |
| `TrainingModeConfigError` | `torchlens.errors.TrainingModeConfigError` | removed |
| `VisualizationOptions` | `torchlens.options.VisualizationOptions` | removed |
| `build_render_audit` | `torchlens.experimental.dagua.build_render_audit` | removed |
| `check_metadata_invariants` | `torchlens.validation.check_metadata_invariants` | removed |
| `check_spec_compat` | `torchlens.validation.check_spec_compat` | removed |
| `cleanup_tmp` | `torchlens.io.cleanup_tmp` | removed |
| `get_model_metadata` | `torchlens.io.log_model_metadata` | removed |
| `list_logs` | `torchlens.io.list_logs` | removed |
| `log_model_metadata` | `torchlens.io.log_model_metadata` | removed |
| `trace_to_dagua_graph` | `torchlens.experimental.dagua.trace_to_dagua_graph` | removed |
| `preview_fastlog` | `torchlens.fastlog.preview` | removed |
| `rehydrate_nested` | `torchlens.io.rehydrate_nested` | removed |
| `render_lines_to_html` | `torchlens.experimental.dagua.render_lines_to_html` | removed |
| `render_trace_with_dagua` | `torchlens.experimental.dagua.render_trace_with_dagua` | removed |
| `reset_naming_counter` | `torchlens.io.reset_naming_counter` | removed |
| `resolve_sites` | `torchlens.validation.resolve_sites` | removed |
| `save_intervention` | `torchlens.io.save_intervention` | removed |
| `suppress_mutate_warnings` | `torchlens.io.suppress_mutate_warnings` | removed |
| `unwrap_torch` | `torchlens.backends.torch.wrappers.unwrap_torch` | removed |
| `validate_batch_of_models_and_inputs` | `torchlens.validation.validate_batch_of_models_and_inputs` | removed |
| `wrap_torch` | `torchlens.backends.torch.wrappers.wrap_torch` | removed |
| `wrapped` | `torchlens.backends.torch.wrappers.wrapped` | removed |

## Top-Level Paper-Era Names

| Old name | New name | Status |
| --- | --- | --- |
| `log_forward_pass` | `trace` | removed |
| `validate_model_activations` | `validate(scope="forward")` | removed |
| `validate_saved_activations` | `validate(scope="saved")` | removed |
| `render_graph` | `Trace.draw()` (or `torchlens.visualization.show_model_graph`, itself a moved top-level alias below) | removed |
| `render_model_graph` | `Trace.draw()` (or `torchlens.visualization.show_model_graph`, itself a moved top-level alias below) | removed |
| `draw_model_graph` | `Trace.draw()` (or `torchlens.visualization.show_model_graph`, itself a moved top-level alias below) | removed |
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
| `validate_saved_outs` | `torchlens.validation.validate_saved_outs` | removed |
| `summary` | `torchlens.visualization.summary` | removed |
| `show_model_graph` | `torchlens.visualization.show_model_graph` | removed |
| `draw_backward` | `torchlens.visualization.draw_backward` | removed |
| `draw_combined` | `torchlens.visualization.draw_combined` | removed |
| `load_intervention_spec` | `torchlens.io.load_intervention_spec` | removed |
| `ModuleInputSnapshot` | `torchlens.types.ModuleInputSnapshot` | removed |
| `PreHookEffect` | `torchlens.types.PreHookEffect` | removed |
| `TensorInputObservation` | `torchlens.types.TensorInputObservation` | removed |

## Capture And Option Keyword Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `capture_output_structure` | `capture_container_structure` | removed |
| `layers_to_save` | `save` or grouped capture options | removed |
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
| `log_forward_pass(layers=...)` | `trace(layers_to_save=...)` | removed |
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

**`keep_module` note (honest capability statement).** Predicate-gated
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
| `Trace.remove()` and `with trace.attach_hooks(...)` | `Trace.clear_edits(site)`; the scoped form stays as `with trace.fork() as scoped: scoped.do(...)`, which detaches the last `do` hooks on exit | removed |
| `ReplayOptions(hooks=)` | a mapping of sites to edits on `Trace.do` | removed |
| `ReplayOptions(differentiable=)` | `InterventionOptions(differentiable=True)` on `Trace.do` | removed |
| `Trace.rerun()` | `Trace.run()` | removed |

## Intervention, Sweep, And Observer Aliases

| Old name | New name | Status |
| --- | --- | --- |
| `replay()` | `do(site, edit)` | removed |
| `replay_from()` | `do(site, value)` | removed |
| `rerun()` | `run()` | removed |
| `Bundle.replay()` | `Bundle.do(site, edit)` | removed |
| `Bundle.rerun()` | `Bundle.run()` | removed |
| `Bundle.measure_members(metric=, order=)` | `Bundle.map(fn)`: live members by name, released candidates read `"unmeasured"` | removed |
| `intervening()` | `without_op()` | removed |
| `sweep(param=...)` | `sweep(at=...)` | removed |
| `record_span()` | `span()` | removed |
| `get_model_metadata()` | `log_model_metadata()` | removed |

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
| `direction=` on `Trace.do`, `Trace.attach_hooks`, `tl.when`, `tl.observers.tap`, `tl.stats.narrate`, `tl.sweep` | the site: `tl.grad(site)`, or `tl.grad(site) \| site` for both | removed |
| `tl.stats.aggregate(target=)` | metric keys are sites (`tl.grad(site)` for a gradient) | removed |
| `tl.observers.span(direction=)` | `span(signal=)` | removed |
| `show_bundle_graph(direction=)` | `show_bundle_graph(signal=)` | removed |
| `FireRecord.direction`, `TapRecord.direction`, `HookContext.direction` | `.signal` | removed |
| `Edit.kind`, `Edit.direction` | `Edit.signal`; `Rule.signal` (derived) | removed |
| `Rule.direction`, `Rule(direction=)` | the target decides the pass: `Rule(tl.grad(site), edit)` for the gradient, `Rule(site, edit)` for the activation; `Rule.signal` (derived) | removed |
| `HookContext.ctx`, `HookContext.run_ctx` | `HookContext.state`, `HookContext.run_state` | removed |

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
