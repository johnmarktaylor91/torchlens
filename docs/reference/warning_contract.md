# Warning contract (S-18)

TorchLens warnings that steer user decisions carry a stable machine-readable
`code` in `warning.fields["code"]` and a structured remedy in
`warning.fields["remedy"]` (derived at the one `TorchLensWarning` chokepoint
from the authored `Remedy: ...` message tail, exactly like the error base).
Consumers branch on `fields["code"]`, never on message text.

This table is the closed vocabulary of CONTRACTED warning codes. The
alias-resolving warn-site census and the lockstep gates live in
`tests/composition_expectations/` (S-18): every `code=` passed at a
`warnings.warn` construction site must have a row here, every row here must
resolve to a live warn site, and the uncoded-site count is a monotone
ratchet burning down to zero. A warning attached to a wrong or unusable
result is a GAP, not a disclosure (compo memo D3); silently overriding an
explicit user request is a typed conflict, never a warning.

| Code | Warning | Remedy class |
|---|---|---|
| `annotation_sweep_sites_skipped` | A default `*_evolution` sweep skipped saved sites that are not stimulus-indexed (buffer overwrites, leading-axis mismatches); processing them would fabricate stimulus-space results (neuro memo D4/D5) | Select stimulus-indexed sites explicitly; an explicit `save=` selection of a skipped site raises the full teaching refusal |
| `extraction_position_ids_derived` | `extract_dataset` derived `position_ids` from a non-right-aligned attention mask and passed them to the model's forward; derived positions match single-sequence forwards exactly for absolute-position models and change nothing for rotary models (extract memo D5, disclosed once per run) | Right-pad the batch (`padding_side="right"`) or include explicit `position_ids` in each batch to silence the derivation |
| `batchnorm_train_stats_mutated` | Tracing ran the model's real forward and train-mode norm layers with `track_running_stats=True` advanced their running statistics in place (momentum applies even inside `torch.no_grad` / `inference_only=True`); fires once per process | `model.eval()` before observational captures, or snapshot `state_dict()` to roll statistics back; training-through-trace users ignore |
| `failed_capture_release_incomplete` | After a FAILED capture TorchLens could not fully remove its instrumentation from the model (a secondary release failure); the model may keep instance-level forward wrappers and fail to pickle | Call `tl.release_model(model)` once the underlying condition is resolved |
| `module_filter_zero_saved` | The `module_filter` third save gate suppressed every payload the save selection picked: the returned trace carries full metadata but zero saved activations (the natural nn.Module-shaped filter matches nothing because the predicate receives an op-record namespace) | Write the filter against op-record fields (e.g. `lambda op: op.func_name == 'linear'`), or drop `module_filter` |
| `stop_after_never_fired_callable` | A callable or ambient (`tl.experimental.stop_after` context manager) stop-after site never fired: the capture ran the full forward and the outputs are the model's real outputs, not a frontier (selector-shaped explicit sites REFUSE typed instead: `stop_after_never_fired`) | Check the stop_after site against the executed model, or drop stop_after |
| `recipes_unknown_provider` | `activate_entrypoint_recipes(names=...)` was asked for entry-point names that are not installed; the unknown names were skipped and every installed provider name is listed (disclosed, never silent) | Activate an installed provider name from `installed_recipe_providers()`, or install the provider distribution first |
| `rerun_zero_fire` | A rerun hook plan entry fired at zero sites on the new inputs; the rerun completed but those interventions were silent no-ops (the flagship silent-wrongness signal, compo memo 5.4) | Resolve the target sites against the rerun trace (`trace.resolve_sites`) before re-applying, or route the edit through the push engine (`fork().do(...)`), which validates sites at plan time |
| `runnable_random_init_run` | A weight-free runnable artifact (the default `include_weights=False`) is executing on RANDOM `torchlens_role_init_v2` state: outputs come from a freshly initialized model, not the captured one, and the report's `verified` attests path faithfulness against that random state (WT1 A-IV item 17, lane A08) | Re-save with `tl.save(trace, path, level='runnable', include_weights=True)`, or bind real weights with `trace.load_state_dict(state_dict)` before `run()` |
| `dense_subspace_full_axis_edit` | `do()` received a subspace selection whose basis supports the ENTIRE bound axis: under the documented set semantics the edit rewrites every element of the axis, never the component along the direction (list-A row 8 point-of-use disclosure; the projection-valued edit is a named fork, not a shipped capability) | Use a sparse direction (or raise `tol=`) to target a subset, or accept the full-axis edit knowingly |
| `patch_campaign_all_identical` | An activation-patching campaign fired at every site but EVERY fire replaced the site value with an identical tensor (clean == corrupted everywhere), so the published table is guaranteed to equal the corrupted baseline -- usually a facet anchored on an input-derived op rather than the computation it names | Inspect `tl.facets.facet_coverage(trace)` and re-anchor the facet before publishing the table as a null result |

Adding or renaming a warning code updates this table, the
`docs/reference/error_refusal_contract.md` code table (the package-wide
`code=` literal scanner reads that one), and the S-18 lockstep baseline in
the same change.
