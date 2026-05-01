# validation/ - Implementation Guide

## core.py
- `validate_saved_activations()` is the main saved-forward replay entry point.
- `_validate_single_layer()` handles one layer's replay and perturbation.
- `_execute_func_with_restored_state()` restores RNG/autocast state around replay.
- `_perturb_layer_activations()` is bounded by `MAX_PERTURB_ATTEMPTS`.
- Validation requires saved function args for replay; check callers preserve
  `save_function_args=True`.

## backward.py
- `validate_backward_pass()` compares TorchLens backward capture against stock autograd.
- Keep tolerances and loss handling in sync with `capture/backward.py`.
- Backward-specific kwargs are routed through `validate(..., scope="backward")`.

## consolidated.py
- `validate(model, input_args, scope=...)` is the top-level 2.x dispatcher.
- Valid scopes are `forward`, `backward`, `saved`, and `intervention`.
- Reject scope-specific kwargs early when they do not apply.

## exemptions.py
Registries:
- `SKIP_VALIDATION_ENTIRELY`
- `SKIP_PERTURBATION_ENTIRELY`
- `STRUCTURAL_ARG_POSITIONS`
- `CUSTOM_EXEMPTION_CHECKS`

Posthoc checks handle bool outputs, casts, `__setitem__`, small tensor coincidences, all
inf/NaN tensors, and special-value args.

## invariants.py
- `MetadataInvariantError` is the public invariant failure type.
- `check_metadata_invariants()` should fail loudly on broken graph/log structure.
- Keep invariants aligned with primary conditional fields, not only legacy THEN views.

## __init__.py Schema Checks
- `validate_tlspec()` only validates unified `.tlspec` manifests.
- Legacy `v2.16_*` formats return without schema validation.
- Manifest schema lives at `torchlens/schemas/tlspec_manifest_v1.json`.

## Known Limitations
- bfloat16 tolerance remains tighter than dtype epsilon in some replay paths.
- Quantized tensors can still hit unsupported tensor operations in comparisons.
- Replay behavior under autocast depends on captured autocast state coverage.
- Selective `layers_to_save` validation needs saved parents or an exemption path.
