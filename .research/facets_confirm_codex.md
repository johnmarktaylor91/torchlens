## VERDICT: SATISFIED (P1 buildable)

## Remaining blockers (if any) — each with a concrete resolution

None.

## Residual notes (non-blocking)

- Round-1 blocking findings #1-#6 are adequately resolved in v2:
  grad is now an explicit capability with typed `MissingGradient`; paired grad_fn input-side gradients are dropped
  from P1; transform power is split into fail-closed capability classes; attention reconstruction is moved to P3
  with the needed capture prerequisites; registry behavior is made trace-owned and immutable; and the multi-output
  work is reframed as verification plus `.facets` exposure rather than a risky capture rewrite.
- The de-risked P1 sequence is coherent: P1a builds the immutable snapshot and structural-output naming first,
  P1b defines `FacetSpec` plus read/grad and missing-gradient behavior, and P1c migrates recipes with an explicit
  capability inventory.
- The `FacetSpec` ABI and capability taxonomy are tight enough for P1 implementation. The conservative default is
  clear: only provably op-anchored structural specs may claim grad, and non-op/parameter/computed facets fail closed
  to read-only or missing.
- Implementation should make two conservative choices explicit in tests/docs: the exact `MissingGradient` surface
  (returned sentinel versus raised-on-access sentinel) and the total ordering used for "weakest link" transform-chain
  classification. These are specification polish items, not P1 blockers, because v2 already defines the safe behavior.
