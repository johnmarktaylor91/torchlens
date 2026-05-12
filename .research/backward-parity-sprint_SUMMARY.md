# Backward-Parity + alpha.3 Sprint Summary

completed_at: 2026-05-12T08:05:56-04:00
branch: codex/backward-parity-P1-alpha3
status: ready for JMT review and merge

## Sprint Goal

This sprint closed the backward-parity and alpha.3 capture gap by moving
TorchLens closer to a single capture IR for forward and backward workflows,
hardening backward validation, adding backward-aware visualization,
intervention, observer, fastlog, and stats surfaces, and preserving the 2.x
public version line with no TorchLens version bump.

## Planning Trajectory

The plan went through 14 adversarial critique rounds across redundant Claude and
Codex labs before the v15 final freeze. Early rounds closed synthesis and
mechanism gaps; middle rounds converged on AccumulateGrad attribution,
validation viability, selector composition, and observer semantics; later rounds
tightened implementation-edge details such as resolver direction inference,
fastlog forward-to-backward grad_fn joins, selector serialization, and static
preflight behavior. AD-32 intentionally deferred the per-layer-gradient oracle
after repeated correctness hazards in the proposed designs.

## Phase Summary

P1, 8 commits: finished alpha.3 capture plumbing by adding `LiveOpRecord`
projection/write sites, draining records through postprocess Step 0, renaming
backward metadata from `has_op` to `is_intervening`, adding capture-time
AccumulateGrad attribution, and tightening parity helpers.

P2, 4 commits: hardened `validate_backward_pass` with `random_seed`,
`validate_metadata`, state_dict restore, grad clearing, train-mode parity
coverage, and backward graph metadata invariants across public wrapper paths.

P3, 4 commits: added combined forward/backward visualization through
`Trace.draw_combined`, module-aware backward clustering, dashed correspondence
edges, intervening grad_fn placement controls, regression tests, and sample PDF
artifacts.

P4, 5 commits: added the backward selector DSL (`tl.grad_fn`,
`tl.intervening`, `tl.grad_fn_label`), resolver direction inference, live
grad_fn helper dispatch, `grad_clip` / `grad_noise` / `grad_clamp`, and
backward selector/helper tests.

P5, 3 commits: added `gradient_postfunc` as a silent alias for
`grad_transform`, fastlog backward gradient capture through
`Recording.log_backward` / `Recorder.log_backward`, predicate-mode
forward-to-backward attribution, `aggregate(target="grad", loss_fn=...)`, and a
streaming norm stat.

P6, 5 commits: added `tap(direction="backward"|"both")` and
`record_span` direction support, wired backward observer normalization through
hook plans, added backward bundle visualization regression coverage, and filed
the deferred per-layer-gradient oracle follow-up.

P7, 10 commits including block/fix commits: ran the final full T2 sweep,
resolved prior T2 blockers in P7B/P7C, consolidated the CHANGELOG, and prepared
the final sprint docs for review.

## Deferred Items

- AD-32 per-layer-gradient oracle for backward validation.
- Unified backward bundle supergraph beyond the existing per-member regression
  coverage.
- Full-repo `ruff check . --fix` remains sensitive to unrelated dirty notebook
  state; the P7 required scoped `ruff check torchlens/ --fix` is green.

## Final Test Counts

- Smoke: `pytest tests/ -m smoke -x --tb=short` -> 194 passed, 2206
  deselected, 117 warnings in 19.43s.
- T2 non-slow: `pytest tests/ -m "not slow" -x --tb=short` -> 2165 passed,
  24 skipped, 209 deselected, 2 xfailed, 934 warnings in 579.58s.
- Static: `ruff check torchlens/ --fix` -> all checks passed.
- Static: `mypy torchlens/` -> success, no issues in 199 source files.

## Branch Metrics

- Commits since `main`: 39 at summary creation.
- Files touched: 75.
- LOC delta: 75 files changed, 6012 insertions(+), 277 deletions(-).
- Total time spent: approximately 10 hours of implementation and wrap-up time
  recorded or inferred from phase docs, excluding the earlier research and
  adversarial planning loop.

## Branch Status

Ready for JMT review and merge: yes.

Suggested merge strategy: fast-forward. This is a single sprint branch with no
parallel branch work expected.

## Follow-Up Sprint Candidates

- Implement the AD-32 per-layer-gradient oracle once a lower-risk oracle design
  is agreed.
- Revisit unified backward bundle supergraph rendering after the current
  per-member path has been reviewed.
- Clean the unrelated notebook syntax issue so full-repo `ruff check . --fix`
  can be used again without scope exceptions.
