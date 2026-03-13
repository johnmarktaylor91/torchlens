# validation/ — Implementation Guide

## core.py — Key Functions
- `validate_saved_activations()` — Entry point. Runs ground truth + logging + BFS validation.
- `_validate_single_layer()` — Replay + perturbation for one layer.
- `_execute_func_with_restored_state()` — Runs function with saved RNG/autocast state.
- `_perturb_layer_activations()` — Bounded by `MAX_PERTURB_ATTEMPTS=100` for int/bool tensors.
- `_deep_clone_tensors()` — Recursive tensor cloning in nested structures.

Tolerance: `MAX_FLOATING_POINT_TOLERANCE = 3e-6`

## exemptions.py — 4 Registries

| Registry | Purpose | Examples |
|----------|---------|----------|
| `SKIP_VALIDATION_ENTIRELY` | Nondeterministic output | `empty_like` |
| `SKIP_PERTURBATION_ENTIRELY` | All args structural | `expand_as`, `zeros_like`, `ones_like` (~12 funcs) |
| `STRUCTURAL_ARG_POSITIONS` | Specific positions insensitive | `cross_entropy`, `index_select`, `scatter_` |
| `CUSTOM_EXEMPTION_CHECKS` | Dynamic logic | `__getitem__`, `__setitem__`, `lstm`, `interpolate` |

Posthoc checks (after execution): bool output, `to()` casting, `__setitem__` same-shape,
small tensor coincidence, all-inf/NaN, special-value args (all-zeros/all-ones).

## Known Bugs & Limitations
- **BFLOAT16-TOL**: `MAX_FLOATING_POINT_TOLERANCE = 3e-6` is 2,600x too tight for bfloat16
  (epsilon ~7.8e-3). Validation always fails for bfloat16 models with `allow_tolerance=True`.
- **QUANTIZED-CRASH**: `tensor_nanequal()` calls `.isinf()` which raises AttributeError on
  quantized tensors. Validation crashes for quantized models.
- **VALIDATE-STATE-RESTORE**: `validate_forward_pass` (user_funcs.py:535-552) saves model
  state_dict but `load_state_dict()` only runs on success path. If forward pass raises,
  model params remain mutated. Needs try/finally.
- **INVARIANT-COND-THEN**: No invariant check for `cond_branch_then_children` <->
  `conditional_then_edges` consistency.
- **Bug #151**: Replay crash -> silently passes. `_execute_func_with_restored_state` catches
  ANY exception and returns None; caller treats None as valid.
- **Bug #150**: Crashes on unsaved parents when `layers_to_save` is selective.
- **Autocast**: Context not captured during logging — replay runs outside autocast,
  producing different precision.

## Gotchas
- Validation requires `save_function_args=True` — without it, replay can't reconstruct inputs.
- Layers with `func_applied=None` (unused model inputs) must early-return True.
- `posthoc_perturb_check` returning True on first special arg (all-zeros/all-ones) is
  correct — any single special arg can explain output invariance.
