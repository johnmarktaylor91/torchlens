# validation/ - Replay, Invariants, and Schema Checks

## What This Does
Validates TorchLens captures at several levels: saved forward activations, backward capture,
metadata invariants, intervention readiness, and unified `.tlspec` manifest schema.

## Files

| File | Purpose |
|------|---------|
| `core.py` | Saved-activation replay, perturbation checks, arg reconstruction |
| `backward.py` | Backward capture validation against stock autograd |
| `consolidated.py` | Public `validate(..., scope=...)` dispatcher and intervention report |
| `invariants.py` | Metadata invariant categories and `MetadataInvariantError` |
| `exemptions.py` | Replay/perturbation exemption registries and dynamic checks |
| `__init__.py` | Public validation exports plus `.tlspec` manifest schema validation |

## Validation Scopes
`torchlens.validate(model, x, scope=...)` accepts:
- `"forward"` - calls `validate_forward_pass()`.
- `"saved"` - validates saved activations on a `ModelLog` path.
- `"backward"` - validates first-class backward capture.
- `"intervention"` - currently runs forward-like checks and returns intervention-axis details.

Legacy top-level shims for `validate_forward_pass`, `validate_backward_pass`, and
`validate_saved_activations` forward to this package.

## Forward Replay Flow
1. Run the model for ground truth output.
2. Run `log_forward_pass(..., layers_to_save="all", save_function_args=True)`.
3. Check logged output matches ground truth.
4. Walk backward from outputs, replaying each saved operation from saved parents.
5. Perturb parents to ensure output sensitivity unless exempt.
6. Optionally run metadata invariants.

## Invariants
`check_metadata_invariants(model_log)` checks structural and semantic consistency: model-log
self consistency, graph topology, LayerPassLog/LayerLog fields, recurrence, branching,
module hierarchy, params, buffers, equivalence, ordering, distances, connectivity, and lookup
keys. Invariants are part of the postprocess regression net.

## .tlspec Validation
`validate_tlspec(path)` validates unified manifests against `schemas/tlspec_manifest_v1.json`.
Older 2.16 intervention/model-log formats are accepted without schema validation so legacy
artifacts remain loadable.

## How It Connects
Validation reads `data_classes/`, uses original torch functions for replay, and calls public
user functions for fresh captures. It must stay independent of visualization and optional
bridge extras.
