# attribution/onebackward/ -- one-backward reads (lane F04)

One suppressed `autograd.grad` over alias-deduped GradientEdges resolved
from EXISTING op fields; results ride the immutable `ReadTable` with honesty
as columns. Doc of record: `docs/reference/onebackward_reads.md`; design of
record: the M(reads) tri-lab memo. Spellings DOCUMENTED-UNSTABLE pending the
naming sprint; error codes are the contract.

| Module | Owns |
|---|---|
| `_accessor.py` | site -> `(node, slot)` index over existing fields; cache; closed liveness gate; id-rejoin fallback |
| `_suppress.py` | `read_suppressed` context (generalized RF probe suppression) + pinned-ref tripwire |
| `_targets.py` | `seed(...)` edge-seeded family; tensor/callable family; teaching refusals |
| `_engine.py` | the one `autograd.grad` pattern; CONE-GROUPED chunked batched VJPs; plan-vs-refusal |
| `_frozen.py` | `frozen=` three-state contract; slot-targeted prehooks; FORK-1 detector (both branches behind `FROZEN_DEFAULT_DISCLOSURE_BRANCH`) |
| `_facet_bridge.py` | semantic-evidence-only MLP-out resolver; frozen-policy registry (the R10 seam) |
| `_read.py` | orchestration: population law (D9/D10), reductions (D15), provenance |
| `_table.py` | `ReadRow`/`ReadTable`/`TableProvenance`; pandas adapter; scalar `read_table_v1` persistence |
| `_edge_plumbing.py` | `GradInputUseMap` fail-closed contract; origin-namespaced pass stamps (deferred-EAP plumbing) |
| `_errors.py` | `ReadError` / `ReadInternalError` (codes on `fields["code"]`) |

## Invariants a change here must never break

- `materialize_grads=False`, `allow_unused=True`, `retain_graph=True`,
  always. autograd `None` -> `status='unreachable'`; a numeric zero stays
  `ok`. Never fabricate a zero for an unreachable row.
- Batched chunks group by CONE KEY: a mixed-cone `is_grads_batched` call
  returns fabricated ZERO rows for not-upstream pairs (measured; pinned in
  `tests/test_onebackward_hookband.py`).
- Every read backward runs inside `read_suppressed`; nothing here may run a
  wrapped torch op or an unsuppressed backward on a live trace.
- The frozen resolver consumes facet evidence only -- a name-substring
  filter is a defect (guard pin in `tests/test_onebackward_frozen.py`).
- No persisted trace fields: caches ride WeakKeyDictionaries; the ReadTable
  artifact is standalone JSON through the bounded reader. F04 has NO
  field_intent rows -- adding one is a C07-owner amendment, not a lane edit.
- Explicit enumeration is a contract; implicit population is a filter (D9,
  both here and at the `by=` door in `torchlens/selection_values.py`).

Tests: `tests/test_onebackward_*.py` (core, hookband, refusals, frozen,
bydoor, edgeplumbing, realmodel).
