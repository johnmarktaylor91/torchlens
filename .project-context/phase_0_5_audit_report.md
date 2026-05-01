# Phase 0.5 intervention API audit report

## Sources checked

- `00_CURATED_FEATURE_LIST_FINAL.md` §6 lines 298-345.
- `intervention_api_2_16_shipped.md` memory note.
- `PLAN.md` effective v5.2 intervention spec.
- `V5_2_REDRAFT_NOTES.md`.
- `IMPLEMENTATION_PLAN_FINAL.md` pre-work and Phase 5a sections.

## Verified consistent: mutating verbs

The final plan says 2.16.0 ships six mutating verbs on `ModelLog`: `set`, `attach_hooks`, `do`, `replay`, `rerun`, and `fork`. This matches `PLAN.md` §1.5/§5 and the memory note list. The memory note headline says "5 mutating verbs" but immediately lists all six; the final plan already corrected that drift as an R2/H1 note.

## Verified consistent: helpers

The 14 shipped helpers match across `PLAN.md` and the memory note: `zero_ablate`, `mean_ablate`, `resample_ablate`, `steer`, `scale`, `clamp`, `noise`, `project_onto`, `project_off`, `swap_with`, `splice_module`, `bwd_hook`, `gradient_zero`, and `gradient_scale`.

## Verified consistent: new work needed

The curated §6 NEW work still missing from the shipped 2.16.0 surface matches the final plan Phase 5a split: sequential hook composition/scoped handles, `tl.sites(...)`, `tl.tap(...)`, `tl.record_span(...)`, multi-input `torchlens.experimental.session(model)`, direct-effect `torchlens.experimental.freeze_module(layer)`, and consolidated `validate(scope=...)` including intervention validation. The related but non-launch or namespaced work remains outside the shipped 2.16.0 surface: causal-trace/logit-lens/contrastive/trainable recipes, `tl.viz.causal_trace_heatmap`, `torchlens.report.log_value`, LLM sequence selectors, and the noise/baseline mini-DSL without overloading the shipped `noise(std, ...)` helper.

## Drift found

No vault-level drift requiring an edit was found. The only wording drift is the memory note sentence "5 mutating verbs" despite listing six; the final implementation plan already records the correction.
