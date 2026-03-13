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

## Validation Flow

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

## How It Connects

Called by `user_funcs.py:validate_forward_pass()` and `validate_saved_activations()`.
Reads data from `data_classes/` (ModelLog, LayerPassLog). Uses original torch functions
for replay (not decorated versions). The 18 invariant checks (A-R) serve as a structural
regression test for the entire postprocess pipeline.

## Invariants (A-R)

**Phase 1 (A-L) — Structural:**
A: model_log self-consistency, B: special layer lists, C: graph topology,
D: layer_pass_log fields, E: recurrence, F: branching, G: LayerPassLog↔LayerLog,
H: module-layer containment, I: module hierarchy, J: params, K: buffers, L: equivalence

**Phase 2 (M-R) — Semantic:**
M: graph ordering, N: loop detection consistency, O: distances, P: connectivity,
Q: module containment logic, R: lookup key consistency
