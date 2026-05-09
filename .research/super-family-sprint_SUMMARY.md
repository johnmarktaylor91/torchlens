# super-family-sprint -- SUMMARY

## What shipped
- Phase 1 `fbf17d4` -- Trace-side `Accessor[T]` base for log accessors.
- Phase 2 `d25d9a2` -- Bundle-side `Super[T]` base plus `_TensorBearing` mixin.
- Phase 3 `3232674` -- Bundle-side `SuperAccessor[T, S]` base.
- Phase 4 `46d230e` -- Folded `multi_trace/` into `intervention/_super/` and `_topology/`.
- Phase 5 `005a72d` -- Added remaining Super* classes and Bundle accessors.
- Phase 6 `10ca3f2` -- Added Bundle structural-agreement predicates.
- Phase 7 `d0e7db0` -- Added `bundle.at(label)` auto-detect dispatcher.
- Phase 8 `this commit` -- Closed docs, memory, summary, and final verification.

## Architectural decisions made
- Super[T] generic base + per-kind extensions, NOT N hand-written classes
- _TensorBearing mixin keeps tensor-stacking behavior reusable for future tensor-bearing Super classes
- SuperAccessor[T, S] base parametrized by Trace-side type and Super wrapper class
- multi_trace/ folded into intervention/ for cohesion (Bundle and Super* live together)
- Universal rule: every sub-Trace class has a Super counterpart at Bundle level (no piecemeal exceptions)
- Naming: Super[T] kept; pairs with existing Supergraph; revisit compound nouns (SuperModuleCall, SuperGradFnCall) in naming-sprint v3

## Files changed (total, post-hoc count)
`git diff main..HEAD --shortstat`: 417 files changed, 26850 insertions(+), 19783 deletions(-)

## Test counts
- Pre-sprint smoke: 170/170 (per Phase 1 commit message)
- Post-sprint smoke: 170/170 passed (`pytest tests/ -m smoke -x --tb=short`)
- Post-sprint bundle: 78/78 passed (`pytest tests/ -k "bundle" -x --tb=short`)
- Post-sprint tier-2 (excluding pre-existing nnsight + rendering colon failures): 1993 passed, 21 skipped, 224 deselected, 2 xfailed (`pytest tests/ -m "not slow" --ignore=tests/test_migrations.py --deselect tests/test_param_log.py::TestVisualizationParams -x --tb=short -q`)

## Residual / known issues
- tests/test_migrations.py::test_migration_example_runs[from_nnsight.md-...] fails on main due to nnsight env -- not introduced by this sprint
- tests/test_param_log.py::TestVisualizationParams::* fails on working tree due to unrelated rendering.py edits -- not introduced by this sprint

## Next-sprint setup
The following naming-sprint v3 / cleanup-sprint items are now easier because of this sprint:
- peek -> pluck rename (Accessor[T] base from Phase 1 makes trace.orphans accessor trivial when keep_orphans flag lands)
- Layer/op-level wrappers for trace-level intervention verbs (Phase 5 plumbing supports these)
