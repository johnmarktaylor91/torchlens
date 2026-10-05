# torchlens/validation architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

## What This Does

Validates TorchLens captures at several levels: saved forward outs, backward capture,
metadata invariants, intervention readiness, and unified `.tlspec` manifest schema.

## Files

| File | Purpose |
|------|---------|
| `core.py` | Saved-out replay, perturbation checks, arg reconstruction |
| `backward.py` | Backward-pass grad-capture validation vs stock autograd |
| `consolidated.py` | Public `validate(..., scope=...)` dispatcher and intervention report |
| `invariants.py` | Metadata invariant categories and `MetadataInvariantError` |
| `_invariants_*.py` | Per-domain invariant implementations, 13 modules (entry, topology, connectivity, conditionals aggregate + conditional base + conditional modules, buffers, modules/params, payloads, equivalence, backward graph/flow/domain); `_invariants_conditionals.py` is the largest conditional module and rebinds through `invariants.py` |
| `_pristine.py` | Pristine-oracle helpers (fixwave-5 R75-1): the untouched-baseline capture the replay oracle compares against |
| `exemptions.py` | Replay/perturbation exemption registries and dynamic checks |
| `_index_domain.py` | Index-domain slot/domain facts and in-domain index perturbations (rotation, single-entry move) for the gather/scatter/embedding/cross_entropy family |
| `_integer_mod_proof.py` | Saved-call proof that an integer `% +-1` is identically zero (used by the `integer_mod_unit_divisor` posthoc decision) |
| `_completeness_backstop.py` | `completeness_backstop_counts`: dispatcher-witness census vs captured ops, including the module-forward-owned drop rule |
| `status.py` | Replay-validation status objects |
| `diagnostics.py` | Structured replay-failure diagnostics (add-only relative to pass/fail) |
| `_layer_grad_report.py`, `_stock_layer_grads.py`, `_output_walk.py` | Layer-grad oracle report, stock-autograd grad collection, output-tree walking |
| `__init__.py` | Public validation exports plus `.tlspec` manifest schema validation |

## Validation Scopes

`torchlens.validate(model, x, scope=...)` accepts:
- `"forward"` - calls `validate_forward_pass()`.
- `"saved"` - validates saved outs on a `Trace` path.
- `"backward"` - validates first-class backward capture.
- `"intervention"` - currently runs forward-like checks and returns intervention-axis details.
- `"receptive_field"` - captures an armed trace and samples receptive/projective
  tri-state validations (own dispatch and gate in `consolidated.py`).

Legacy top-level shims for `validate_forward_pass`, `validate_backward_pass`, and
`validate_saved_outs` forward to this package.

## Forward Replay Flow

1. Run the model for ground truth output on PRISTINE torch: installed
   torchlens wrappers are removed for this forward (restored from the
   pre-decoration originals ledger) and reinstalled afterwards, so the
   ground-truth oracle is independent of the wrapper layer it checks
   (R75-1; backward validation's stock-autograd pass does the same). If
   the wrappers cannot be removed because a capture is active, validation
   refuses rather than blessing a wrap-state-dependent ground truth.
2. Run `trace(..., capture=CaptureOptions(layers_to_save="all", save_arg_values=True))`
   (the flat kwargs are deprecated aliases that warn).
3. Check logged output matches ground truth.
4. Walk backward from outputs, replaying each saved operation from saved parents.
5. Perturb parents to ensure output sensitivity unless exempt.
6. Optionally run metadata invariants.

## Invariants

`check_metadata_invariants(trace)` checks structural and semantic consistency: model-log
self consistency, graph topology, Op/Layer fields, recurrence, branching,
module hierarchy, params, buffers, equivalence, ordering, distances, connectivity, and lookup
keys. Invariants are part of the postprocess regression net.

## .tlspec Validation

`validate_tlspec(path)` validates unified manifests against
`schemas/tlspec_manifest_v{schema_version}.json`, selecting the schema version declared by the
artifact.
Only the two legacy 2.16 INTERVENTION formats (`v2.16_intervention`,
`v2.16_intervention_with_kind`) are accepted without schema validation. Legacy 2.16
MODEL-LOG bundles are NOT loadable: they sit below the tlspec v6 / torchlens 2.33
rehydration floor and refuse with `ArtifactVersionBelowFloorError`.

## How It Connects

Validation reads `data_classes/`, uses original torch functions for replay, and calls public
user functions for fresh captures. It must stay independent of visualization and optional
bridge extras.

## core.py

- `validate_saved_outs()` is the main saved-forward replay entry point.
- `validate_parents_of_saved_layer()` handles one layer's replay and perturbation.
- `_execute_func_with_restored_state()` restores RNG/autocast state around replay.
- `_perturb_layer_outs()` is bounded by `MAX_PERTURB_ATTEMPTS`.
- Validation requires saved function args for replay; check callers preserve
  `save_arg_values=True`.

## backward.py

- `validate_backward_pass()` compares TorchLens backward capture against stock autograd.
- Keep tolerances and loss handling in sync with `backends/torch/backward.py`.
- Backward-specific kwargs are routed through `validate(..., scope="backward")`.

## consolidated.py

- `validate(model, input_args, scope=...)` is the top-level 2.x dispatcher.
- Valid scopes are `forward`, `backward`, `saved`, `intervention`, and
  `receptive_field` (own dispatch, gate, and tri-state return; see
  `_validate_receptive_field_scope`).
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
  It is IMPLEMENTED in `_invariants_entry.py` (with the per-domain checks in
  the sibling `_invariants_*.py` modules); `invariants.py` only REBINDS the
  entry function — editing `invariants.py` to change check behavior is a no-op.
- Keep invariants aligned with primary conditional fields, not only legacy THEN views.

## __init__.py Schema Checks

- `validate_tlspec()` only validates unified `.tlspec` manifests.
- Only legacy 2.16 intervention formats skip schema validation; model-log bundles below tlspec v6 refuse to load.
- Manifest schemas live at `torchlens/schemas/tlspec_manifest_v{schema_version}.json`;
  validation selects the version declared by each artifact.

## Known Limitations

- bfloat16 tolerance remains tighter than dtype epsilon in some replay paths.
- Quantized tensors can still hit unsupported tensor operations in comparisons.
- Replay behavior under autocast depends on captured autocast state coverage.
- Selective `layers_to_save` validation needs saved parents or an exemption path.
