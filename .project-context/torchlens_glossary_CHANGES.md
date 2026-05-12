# TorchLens Glossary Changes
> Generated against `codex/post-backward-megasprint` HEAD on 2026-05-11.
> Compares the pre-refresh `torchlens_glossary.md` (1040 lines, walkthrough-locked
> rename targets only) against the current code state after the
> backward-parity-sprint + alpha.3 sprint AND the post-backward megasprint.
>
> The glossary continues to document *target* names from the rename walkthrough.
> Entries are marked **(impl-current)** when the entry instead documents the
> current implementation name because the rename is not yet locked (e.g. backward
> capture API names that landed live and were never run through the walkthrough).

## Added entries (new symbols since last glossary version)

### Backward-parity sprint and alpha.3
- `tl.bwd_hook(fn)` -- HelperSpec wrapper for backward callbacks.
- `tl.grad_zero(*, force_shape_change=False)` -- gradient-zero helper.
- `tl.grad_scale(factor, *, force_shape_change=False)` -- gradient-scale helper.
- `tl.grad_clip(max_norm, norm_type=2.0)` -- grad_fn clipping helper (mount_shape=tuple).
- `tl.grad_clamp(min=None, max=None)` -- grad_fn elementwise clamp helper.
- `tl.grad_noise(std, *, seed=None)` -- grad_fn Gaussian noise helper.
- `tl.grad_fn(type=None, *, label=None, is_custom=None)` -- backward grad_fn selector.
- `tl.intervening()` -- selector matching grad_fns with no paired forward op.
- `tl.grad_fn_label(name)` -- exact grad_fn-label selector.
- `tl.output(target)` -- output index / role selector for multi-output disambiguation.
- `tl.tap(site, *, direction="forward")` -- direction kwarg added to existing tap.
- `tl.record_span(name, *, direction="both")` -- direction kwarg added.
- `tl.aggregate(model, dataloader, metrics, *, target="out", loss_fn=None)` -- `target` + `loss_fn` added (target accepts `"out"` or `"grad"`).
- `Norm` -- streaming statistic class for `tl.stats.Norm`.
- `Trace.draw_combined(...)` -- new combined forward/backward graph renderer with `intervening_cluster: Literal["upstream","outside","downstream","own"]` kwarg.
- `validate_backward_pass(..., random_seed=None, atol=1e-5, rtol=1e-4)` -- exposed validator parameters and state hygiene contract.
- `Trace.log_backward(loss, **backward_kwargs)` (impl-current; public alias `Trace.backward(loss)` already in glossary).
- `Recording.log_backward(loss, *, keep_grad=None, default_grad=None, retain_graph=None, create_graph=False)` -- fastlog backward capture.
- `Recorder.log_backward(loss, *, keep_grad=None, default_grad=None, retain_graph=None, create_graph=False)` -- fastlog Recorder backward.
- `gradient_postfunc` -- silent kwarg alias for `grad_transform` on `tl.trace(...)`.
- `Recording.grad_records`, `grad_by_pass`, `grad_by_label`, `grad_by_grad_fn_label`, `keep_grad_repr` -- fastlog backward result fields.
- `GradientRecord` -- new fastlog dataclass for captured gradient samples.
- `GradRecordContext` -- predicate context object for `keep_grad` callables.
- `GradFnLog.is_intervening: bool` -- replaces former `has_op` field (now a deprecated derived property).
- `_grad_fn_param_refs: dict[str, str]` -- trace-level AccumulateGrad attribution map (capture-time).
- `_param_log_by_pid: dict[int, str]` -- trace-level capture-time `id(param) -> address` registry.
- `tl.show_bundle_graph(..., direction="backward")` -- now functional (previously raised).
- `HelperMountError(HookSiteCoverageError)` -- raised when helpers mount on incompatible selector universes (e.g. forward helper on backward selector).
- `UnclassifiedSelectorError(SiteResolutionError)` -- raised when selector lacks an explicit direction bucket.
- `SelectorCompositionError(SiteResolutionError)` -- raised on cross-direction selector composition.

### Post-backward megasprint
- `ModuleCallLog.outputs: list[OpLog]` -- new structured field listing the actual output OpLogs for this call.
- `ModuleCallLog.output_structure: ContainerSpec | None` -- captured container shape for multi-output modules.
- `ModuleLog.outputs: list[OpLog]` -- aggregated outputs across calls (single-call passthrough).
- `ModuleLog.output_structure: ContainerSpec | None` -- aggregated container structure for single-call modules.
- `MultiOutputModuleError(ValidationError, ValueError)` -- raised for ambiguous singular access on multi-output module calls; ModuleCallLog `out`, `grad`, `out_shape`, etc. raise this instead of `ValueError`.
- `OpLog.multi_output_role: str | None` -- semantic role tag for this Op's position in a multi-output container (none, by container-path index, or named role).
- `tl.func(name, *, output=None)` -- `output` kwarg added for multi-output disambiguation.
- `tl.output(target)` -- new top-level selector (also listed under backward-parity but landed P1 of post-backward).
- `tl.halt(reason="")` callable, `HaltSignal(BaseException)` -- fastlog early-abort signal.
- `tl.fastlog.halt`, `tl.fastlog.HaltSignal` -- fastlog public exports.
- `Recording.halted: bool`, `Recording.halt_reason: str | None`, `Recording.halts_by_pass: dict[int, str]` -- fastlog halt state on results.
- `LayerGradReport` dataclass -- module-output gradient parity report returned by `_validate_layer_grads`.
- `validate_backward_pass(..., validate_layer_grads=False, layer_grad_atol=None, layer_grad_rtol=None)` -- per-module-output validation toggle and tolerances.
- Coverage buckets on `LayerGradReport`: `covered`, `mismatched`, `skipped_no_first_leaf`, `skipped_module_less`, `skipped_no_grad`, `skipped_identity_output`, `skipped_root_module`.
- `AppendStreamingNotSupportedError(ValidationError, ValueError)` -- append rerun cannot mutate active streamed activation blobs.
- `AppendStateValidationWarning(TorchLensInterventionWarning)` -- warning when validators skip fresh checks on stacked appended traces.
- `AppendBatchDependenceError(ValidationError, ValueError)` -- append cannot prove helper or grad batch independence.
- `AppendMismatchError(ValidationError, ValueError)` -- chunked append candidate incompatible with base log (formerly hinted).
- `Trace.append_history: list[dict[str, Any]]` -- per-append provenance ledger.
- `Trace.is_appended: bool` -- True when trace currently holds appended chunks.
- `Trace._append_sequence_id: int` -- monotonic id incremented on each append; cleared on non-append rerun, preserved through `replace_run_state_from`.
- `MLX backend` -- `torchlens.backends.mlx.MLXBackend` adds wrappers for Conv2d, normalization layers, Embedding, Dropout, MultiHeadAttention, reductions, shape ops, and activations.
- MLX `intervention_ready=True` and `save_grads=True` are hard-rejected with `NotImplementedError`.
- MLX `_backend_name: str` survives portable save/load (preserved through `.tlspec`).
- MLX optional dep pinned `mlx>=0.26,<0.27`.
- `tl._io.scrub._ScrubOptions.unsupported_tensor_records: list[dict[str, str]]` -- sidecar audit collector; `scrub_for_save` now returns a 3-tuple `(scrubbed_state, blob_specs, unsupported_tensor_records)`.
- Internal rename: `_add_backward_hook` -> `_add_tensor_backward_hook` (capture-side helper in `backends/torch/tensor_tracking.py`).
- `LayerLog.corresponding_grad_fn` / `GradFnLog.corresponding_op_layer` cross-links -- documented in code under `_legacy_field_aliases`; the canonical name is `grad_fn_log` on Op and `op` (returning the LayerLog) on GradFnLog.

### Direction-classification machinery (selectors)
- `_classify_selector_direction(sel)` -- internal helper, but the contract leaks
  through public errors:
  - `forward` bucket: `FuncSelector`.
  - `backward` bucket: `GradFnSelector`, `InterveningSelector`, `GradFnLabelSelector`.
  - direction-agnostic: `Label`, `Module`, `Output`, `Contains`, `Where`,
    `InModule`, `Composite`, `Not`.
  - All others raise `UnclassifiedSelectorError`.

## Updated entries

- `Trace.draw_backward(...)` -- signature confirmed: `(vis_outpath="backward_modelgraph", vis_graph_overrides=None, node_spec_fn=None, collapsed_node_spec_fn=None, vis_node_mode="default", vis_edge_overrides=None, vis_save_only=False, vis_fileformat="pdf", vis_direction="topdown", code_panel=False)`. Glossary already had it but verified surface.
- `tl.draw_backward(...)` -- top-level wrapper now wraps the deprecated alias path through `_moved_draw_backward`.
- `GradFnLog.has_op` -- now a deprecated property issuing `DeprecationWarning`; returns `not self.is_intervening`. The portable spec key is `is_intervening`. Legacy `has_op` pickles are migrated in `__setstate__`.
- `GradFnLog.op` -- typed as `LayerLog | None`. (Glossary had this; verified.)
- `Trace.replace_run_state_from(new_log)` -- now preserves `is_appended`, `_append_sequence_id`, `append_history` through the atomic swap (post-backward fix `071e9e4`). The glossary's deferred entry on `replace_run_state_from` should be removed since the contract is locked.
- `tl.aggregate(...)` -- legacy 3-arg form preserved; new keyword-only `target` and `loss_fn` parameters added. Default `target="out"` matches prior behavior.
- `Trace.summary(...)` -- signature unchanged; remains `Trace.summary(level="overview", *, fields=None, mode="auto", show_ops=False, preset=None, columns=None, include_ops=None, max_rows=200, print_to=None, count_fma_as_two=False)`.
- `tl.fastlog.record(...)` -- backward predicate kwargs documented: `keep_grad`, `default_grad`. `Recorder.log_backward(...)` keyword-only.
- `MultiOutputModuleError` -- inherits both `ValidationError` and `ValueError`. ModuleCallLog `out`, `grad`, `out_shape`, etc. now raise this specifically (not bare `ValueError`).
- `tl.func(name, *, output=None)` -- existing entry needs `output` kwarg.
- `Recording` -- now has lazy `_capture_events`-driven record materialization; halt fields added.
- `Trace.rerun(model, x, *, append=MISSING, ...)` -- now hard-rejects `append=True` on streaming-active traces with `AppendStreamingNotSupportedError` and rejects batch-dependent helpers with `AppendBatchDependenceError`.
- `Trace.is_appended` clears on non-append `rerun`; preserved through `replace_run_state_from`.
- `tl.show_bundle_graph(..., direction="backward")` -- glossary entry's direction enum now functional, previously raised.

## Removed entries

- `GradFnLog.has_op` as a stored field. Replaced by `is_intervening` (with inverted sense). Glossary entry already documents the rename target (line 806 said `is_intervening`). Keep the deprecated `has_op` property entry under a "Deprecated" note.
- `_add_backward_hook` -- internal helper renamed to `_add_tensor_backward_hook`. (Was not in glossary but listed in spec.)
- `gradient_zero` / `gradient_scale` -- not present in code; the helpers landed as `grad_zero` / `grad_scale` from the outset. No rename, just a naming-question resolution to confirm in `_CHANGES.md`.

## Uncertain entries (flagged with `(?)` in glossary)

These are added with a question marker so JMT can decide whether to keep, promote, or reject them. Each appears in the glossary with `(?)`.

- `_grad_fn_param_refs` and `_param_log_by_pid` -- private (underscore-prefixed) trace fields, but they are part of the AccumulateGrad attribution mechanism. Documented because they survive portable save (`_grad_fn_param_refs` is `FieldPolicy.KEEP`) and any external tooling needs to know about them. (?)
- `gradient_postfunc` -- documented as silent alias for `grad_transform`. The alias is intentional, but the glossary may want to call it out separately as a deprecation target or hide it entirely. (?)
- `OpLog.multi_output_role` -- new field whose taxonomy is not yet fully exercised by tests (it currently shadows `multi_output_index` with a semantic role tag like a tuple-element name or dict key). The vocabulary may evolve. (?)
- `Trace._backend_name` -- exposes "torch" vs "mlx" backend; useful but the underscore prefix implies private. JMT may decide whether to promote to `backend_name`. (?)
- `LayerGradReport` -- public dataclass at `torchlens.validation._layer_grad_report.LayerGradReport`. Underscore-prefixed module currently makes it semi-public. (?)
- `ModuleCallLog.outputs` vs `output_layers` -- both now exist (the former is a list of `OpLog`, the latter a list of pass-qualified labels). The walkthrough deltas confirmed both should stay; verifying no glossary regression. (?)
- `corresponding_grad_fn` (deprecated alias on OpLog mapping to `grad_fn_log`) -- still resolvable through `_legacy_field_aliases`. Whether to mention in glossary deferred section. (?)
- `Trace._param_log_by_pid` policy -- `FieldPolicy.DROP` on save (capture-time only). Documenting because validation backward depends on it. (?)

## Inconsistencies found between glossary and code

1. **`MultiOutputModuleError` MRO.** Glossary spec said
   `MultiOutputModuleError(TorchLensError, ValueError)`. Actual MRO is
   `MultiOutputModuleError(ValidationError, ValueError)`. `ValidationError`
   inherits from `TorchLensError` so the spec is consistent in spirit but not
   literally. Glossary entry now reflects the actual MRO.

2. **`tl.bwd_hook(fn)`** signature has only `fn`. Spec listed it in the helper
   suite. Glossary now lists it next to the other backward helpers in the
   intervention-helpers section.

3. **`tl.peek(model, x, layer, stop_after=None)`** -- spec asked whether
   `peek -> pluck` rename is still deferred. **Status: STILL DEFERRED.** Code
   still defines `peek`; `pluck` does not exist. Glossary's Deferred section
   should keep peek listed.

4. **`vis_*` prefix rename** -- spec asked whether still deferred. **Status:
   STILL DEFERRED.** Code still uses `vis_*` parameter names on `tl.trace`,
   `Trace.draw`, and `Trace.draw_backward`. Glossary Deferred section keeps
   them.

5. **`tl.rerun(append=...)` rename target** -- spec asked whether still
   deferred. **Status: STILL DEFERRED.** Code uses `append` keyword (with
   `MISSING` sentinel). No `extend` or `merge` alias.

6. **`replay`/`replay_from` verb rename** -- spec asked whether still
   deferred. **Status: STILL DEFERRED.** No `replay` rename target in code.

7. **`find_sites`/`resolve_sites`** -- spec asked whether still
   deferred. **Status: STILL DEFERRED.** Both methods exist on `Trace`
   (`find_sites` and `resolve_sites`), and `tl.intervention.resolve_sites`
   is publicly exported. The walkthrough hadn't reviewed them yet.

8. **`preview_fastlog`** -- spec asked whether still deferred.
   **Status: PARTIALLY DEFERRED.** Top-level `tl.preview_fastlog` is now a
   deprecated alias pointing to `tl.fastlog.preview`. The "deferred" listing
   in the glossary should be relabeled "moved to fastlog namespace; alias
   retained one cycle."

9. **`Trace.replace_run_state_from` / `Trace.append_run_state_from`** were
   listed in glossary "Deferred items" -- they are now stable public methods.
   Removed from deferred list and added to the Trace methods section.

10. **Coverage bucket names** for `LayerGradReport` should be documented
    verbatim. Glossary lists them in the spec text but they weren't in the
    glossary entry; added now.

11. **`grad_fn_log` (on OpLog)** is the canonical reference and `corresponding_grad_fn` is the deprecated legacy alias (via `_legacy_field_aliases`). The glossary's `grad_fn_log: TorchLens GradFn record corresponding to this Op` entry stays.

12. **`tl.do(...)`, `tl.replay(...)`, `tl.rerun(...)`, `tl.replay_from(...)`** are top-level functions in `torchlens.intervention` -- not just `Trace` methods. The glossary lists them under "Standalone intervention verbs" already; the corresponding `Trace.do`, `Trace.replay`, `Trace.replay_from`, `Trace.rerun` are also documented.

13. **`tl.intervention.AppendMismatchError`, `AppendStreamingNotSupportedError`, `AppendStateValidationWarning`, `AppendBatchDependenceError`, `MultiOutputModuleError`, `HelperMountError`, `UnclassifiedSelectorError`, `SelectorCompositionError`** are all public exceptions through `torchlens.intervention.__all__`. Documented now in glossary's Errors section (new section added).

## Top-3 inconsistencies prioritized for JMT review

1. **`has_op` -> `is_intervening` rename direction.** The glossary entry on
   line 806 reads "True when a grad-fn node has no corresponding forward op."
   That's correct for `is_intervening`. The deprecated `has_op` property
   returns `not self.is_intervening`, so external code that read
   `gf.has_op == True` (meaning "has a corresponding op") must flip to
   `not gf.is_intervening`. Worth noting in user-facing migration docs.

2. **`Trace.replace_run_state_from` / `append_run_state_from` are no longer
   deferred.** They are stable public methods used by the rerun engine and by
   intervention runtime. Removing from the deferred list changes the
   "what's stable" surface for JMT's rename sprint planning.

3. **`tl.func` now takes `output` kwarg.** This changes the selector taxonomy:
   `tl.func("relu", output=0)` is the canonical multi-output disambiguator.
   The glossary's selector signature is updated. This subtly changes the
   `FuncSelector.selector_value` shape from `str` to `dict[str, Any]` when
   `output` is supplied.
