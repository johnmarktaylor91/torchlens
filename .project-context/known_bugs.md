# Known Bugs

## Phase 14 Cleanup Status

Status: refreshed after Phase 14 bug sweep.

Fixed or verified in Phase 14:
- `BFLOAT16-TOL`: `tensor_nanequal()` now uses dtype-aware tolerance for
  `bfloat16` and `float16`.
- `FUNC-CALL-LOC-LEAK`: `FuncCallLocation` snapshots signature/docstring
  metadata and immediately releases `_frame_func_obj`.
- `ARG-KWARGS-MISSING`: static `FUNC_ARG_SPECS` now covers common keyword
  tensor paths for `linear`, convolutions, norms, attention, `cat`/`stack`,
  and `where`.
- `COND-THEN-MULTIPASS`: rolled `LayerLog` conditional child views merge
  pass-level THEN/ELIF/ELSE children.
- `INVARIANT-COND-THEN`: metadata invariants reject stale
  `cond_branch_then_children` projections.
- `HASH-COLLISION`: deterministic barcodes use SHA-256 prefixes instead of
  Python `hash()` truncation.
- `VALIDATE-STATE-RESTORE`: `validate_forward_pass()` restores model state in
  a `finally` path.
- Validation issues: LSTM perturbation exemption, duplicate same-parent arglocs,
  and identity-output validation have targeted regressions.

Deferred or intentionally not changed:
- `ELK-IF-THEN` / `ELK-EDGE-LABEL-DEDUP`: ELK remains an internal-only renderer
  escape hatch. Graphviz is the supported conditional edge-label path.
- Large visualization direct-SVG work and broader model-prep perf sweeps are
  deferred because they exceed the Phase 14 bug-fix scope.
