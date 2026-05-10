# module-containment-refactor -- Summary

## Sprint goals

Replace TorchLens module containment's tensor-entry/exit thread replay with direct
module-stack snapshots at op creation time. Preserve user-facing module metadata,
loop grouping behavior, save/load compatibility, and fast/exhaustive capture alignment.
Codify the ordering invariants so future capture changes cannot silently reintroduce
module-attribution drift.

## Phases shipped

- `cee50a4` - Phase 0a, extracted the shared module-stack helper.
- `a539469` - Phase 0b, added the module-containment field-equality harness and snapshots.
- `8258229` - Phase 1, added shadow stack capture behind the engine flag.
- `10942ba` - Phase 2, switched production capture to hook-stack and deleted thread replay.
- `22b355f` - Phase 3, reduced Step 6 to suffix-only `equivalence_class` mutation.
- `c8e84e8` - Phase 4, bumped internal IO format and dropped boundary-thread fields on load.
- `<this commit>` - Phase 5, codified ordering, exception-safety, and backward-link regressions.

## Net code delta

The requested `0a..HEAD` range is not a valid revision in this checkout, so the
pre-sprint commit `f120375` is the range boundary used here.

`git log --oneline f120375..HEAD`:

```text
c8e84e8 chore(io): bump portable state schema; drop boundary-thread fields
22b355f refactor(postprocess): simplify Step 6 to suffix-only mutation
10942ba refactor(capture): switch to hook_stack engine; delete thread-replay code
8258229 refactor(capture): add shadow module-stack capture, gated by engine flag
a539469 test(capture): add module-containment field-equality harness (16 fixtures)
cee50a4 refactor(decoration): extract module-stack helper
```

`git diff --shortstat f120375..HEAD`:

```text
43 files changed, 12909 insertions(+), 333 deletions(-)
```

Net before Phase 5: +12,576 lines, dominated by JSON regression snapshots. Phase 5 adds
304 test lines, 27 in-repo documentation lines, and this summary/state update.

## Behavioral improvements documented

- Factory/internal ops created inside submodules now keep the enclosing module frame at
  op creation instead of relying on later replay inference.
- Factory ops created in the root forward now correctly remain root-level instead of
  inheriting downstream child-module containment.
- User forward-hook replacement tensors are explicitly documented as post-exit ops:
  the original module output is marked replaced, while the replacement op is root-level.

## Known residuals

- Fixture 14 retains the documented xfailed divergence around synthetic hook replacement
  ancestry for `linear_1_4`.
- Dynamically registered buffers are observable as the creation op that produced the
  tensor, not as a separate `buffer_address` source node.
- `pytest tests/ -m "not slow" -q` still has baseline failures outside this phase:
  one nnsight migration source-inspection failure and three visualization parameter-label
  assertions in `tests/test_param_log.py::TestVisualizationParams`.

## Final verification

```text
pytest tests/test_module_containment_equality.py -x -q
17 passed, 1 warning

pytest tests/test_module_containment_save_load.py -x -q
4 passed, 1 warning

pytest tests/test_module_stack_identity_exhaustive.py -x -q
1 passed, 1 warning

pytest tests/test_module_stack_buffer_ordering.py -x -q
1 passed, 1 warning

pytest tests/test_module_stack_exception_safety.py -x -q
1 passed, 1 warning

pytest tests/test_module_stack_user_hook_ordering.py -x -q
2 passed, 1 warning

pytest tests/test_backward_attribution_unchanged.py -x -q
1 passed, 2 warnings

pytest tests/ -m "smoke" -x -q
170 passed, 2112 deselected, 111 warnings

pytest tests/ -m "not slow" -q
4 failed, 2042 passed, 25 skipped, 209 deselected, 2 xfailed, 924 warnings
```

## Pointers

- Audit: `.research/module-containment-audit_REPORT.md`
- Plan: `.research/module-containment-refactor_PLAN.md`
