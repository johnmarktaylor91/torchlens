# validation/ — Forward Replay & Metadata Invariants

## What This Does
Validates that logged activations are correct by replaying each operation with saved
parent tensors and checking outputs match. Also provides 18 metadata invariant checks
(A-R) that verify structural consistency of the entire ModelLog.

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `core.py` | 808 | BFS orchestration, forward replay, perturbation, arg preparation |
| `exemptions.py` | 482 | 4 data-driven exemption registries + posthoc checks (16 exemption conditions) |
| `invariants.py` | 1505 | 18 metadata invariant categories (A-R), Phase 1 structural + Phase 2 semantic |

## Validation Flow (`validate_saved_activations`)

```
1. Run model(input) for ground truth output
2. Run log_forward_pass(model, input, layers_to_save="all", save_function_args=True)
3. Verify ground truth matches logged output
4. Backward BFS from outputs — for each layer:
   a. Verify parent layers match saved args (arg position check)
   b. Replay function with saved parents -> check output matches
   c. Perturb each parent -> verify output changes
5. Return True if all checks pass
```

Tolerance: `MAX_FLOATING_POINT_TOLERANCE = 3e-6`

## core.py — Key Functions
- `validate_saved_activations()` — Entry point. Runs ground truth + logging + BFS validation.
- `_validate_single_layer()` — Replay + perturbation for one layer.
- `_execute_func_with_restored_state()` — Runs function with saved RNG/autocast state.
- `_perturb_layer_activations()` — Bounded by `MAX_PERTURB_ATTEMPTS=100` for int/bool tensors.
- `_deep_clone_tensors()` — Recursive tensor cloning in nested structures.

## exemptions.py — 4 Registries

| Registry | Purpose | Examples |
|----------|---------|----------|
| `SKIP_VALIDATION_ENTIRELY` | Nondeterministic output | `empty_like` |
| `SKIP_PERTURBATION_ENTIRELY` | All args structural | `expand_as`, `zeros_like`, `ones_like` (~12 funcs) |
| `STRUCTURAL_ARG_POSITIONS` | Specific positions insensitive | `cross_entropy`, `index_select`, `scatter_` |
| `CUSTOM_EXEMPTION_CHECKS` | Dynamic logic | `__getitem__`, `__setitem__`, `lstm`, `interpolate` |

Posthoc checks (after execution): bool output, `to()` casting, `__setitem__` same-shape,
small tensor coincidence, all-inf/NaN, special-value args (all-zeros/all-ones).

## invariants.py — 18 Checks (A-R)

**Phase 1 (A-L) — Structural:**
- A: model_log self-consistency (counts, timing)
- B: special layer lists (input/output/buffer flag <-> list bidirectionality)
- C: graph topology (parent-child bidirectionality)
- D: layer_pass_log fields (shape/dtype, function callable)
- E: recurrence invariants
- F: branching invariants
- G: LayerPassLog <-> LayerLog cross-references
- H: module-layer containment
- I: module hierarchy (address tree, pass keys)
- J: param cross-references
- K: buffer cross-references
- L: equivalence symmetry

**Phase 2 (M-R) — Semantic:**
- M: graph ordering (creation_order unique/monotonic)
- N: loop detection consistency (recurrent_group symmetry)
- O: distance invariants (conditional on `mark_input_output_distances`)
- P: graph connectivity
- Q: module containment logic
- R: lookup key consistency

Entry: `check_metadata_invariants(model_log)` or `model_log.check_metadata_invariants()`.
Raises `MetadataInvariantError(check_name, message)` on first failure.

## Known Bugs & Limitations
- **BFLOAT16-TOL**: `MAX_FLOATING_POINT_TOLERANCE = 3e-6` is 2,600x too tight for bfloat16
  (epsilon ~7.8e-3). Validation always fails for bfloat16 models with `allow_tolerance=True`.
- **QUANTIZED-CRASH**: `tensor_nanequal()` calls `.isinf()` which raises AttributeError on
  quantized tensors. Validation crashes for quantized models.
- **VALIDATE-STATE-RESTORE**: `validate_forward_pass` (user_funcs.py:535-552) saves model
  state_dict but `load_state_dict()` only runs on success path. If forward pass raises,
  model params remain mutated. Needs try/finally.
- **INVARIANT-COND-THEN**: No invariant check for `cond_branch_then_children` <->
  `conditional_then_edges` consistency. No check that same-layer ops agree on `in_cond_branch`.
- **Bug #151**: Replay crash -> silently passes. `_execute_func_with_restored_state` catches
  ANY exception and returns None; caller treats None as valid.
- **Bug #150**: Crashes on unsaved parents when `layers_to_save` is selective.
- **Autocast**: Context not captured during logging — replay runs outside autocast,
  producing different precision. Tests skip validation for autocast models.

## Gotchas
- Validation requires `save_function_args=True` — without it, replay can't reconstruct inputs.
- `_check_arglocs_correct_for_arg` uses `torch.all(x == 0)` / `torch.all(abs(x) == 1)`
  guards (not mean-based heuristic).
- Layers with `func_applied=None` (unused model inputs) must early-return True.
- `posthoc_perturb_check` returning True on first special arg (all-zeros/all-ones) is
  correct — any single special arg can explain output invariance.

## Related
- [capture/](../capture/CLAUDE.md) — Produces the data being validated
- [data_classes/](../data_classes/CLAUDE.md) — ModelLog/LayerPassLog structure
- `user_funcs.py` — `validate_forward_pass()` is the public API
