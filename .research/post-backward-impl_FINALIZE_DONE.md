# Post-Backward Megasprint Finalize Block

created_at: 2026-05-12T17:08:49-04:00
state: BLOCKED
branch: codex/post-backward-megasprint
head_at_block: 02693a3741c1c6dcdeeb15ed656bdf5cd10e62e6
commits_since_main_at_block: 34
loc_delta_at_block: 4352 insertions(+), 243 deletions(-)
files_touched_at_block: 66
ready_for_review_merge: no
suggested_merge_strategy: blocked until full T2 is green

## Block Reason

The full T2 sweep stopped on the first command:

```bash
pytest tests/ -m "not slow" -x --tb=short
```

This failed in
`tests/test_intervention_phase7.py::test_replace_run_state_preserves_relationship_and_spec_fields`.
The failure is not one of the listed pre-existing blockers. The known notebook ruff syntax
blocker was not involved because ruff was not reached.

Per the finalize instructions, no changelog consolidation, SUMMARY completion, STATE DONE update,
or merge-ready marker was written after this non-pre-existing failure.

## T2 Results

### pytest tests/ -m "not slow" -x --tb=short

Result: failed.

Summary:

```text
collected 2465 items / 211 deselected / 2254 selected
FAILED tests/test_intervention_phase7.py::test_replace_run_state_preserves_relationship_and_spec_fields
= 1 failed, 912 passed, 6 skipped, 211 deselected, 178 warnings in 168.81s (0:02:48) =
```

Failure excerpt:

```text
________ test_replace_run_state_preserves_relationship_and_spec_fields _________
tests/test_intervention_phase7.py:284: in test_replace_run_state_preserves_relationship_and_spec_fields
    assert log.is_appended is True
E   AssertionError: assert False is True
E    +  where False = Trace(name='kept', model_class='ReluAdd', layers=4, run_state=PRISTINE).is_appended
```

Likely root cause to investigate:

The test expects `_replace_run_state`-style rerun bookkeeping to preserve the append relationship
metadata on the retained log. The current branch clears or fails to restore `is_appended` for this
path, leaving the trace in `run_state=PRISTINE` with `is_appended=False`.

### ruff check torchlens/ --fix

Result: not run because pytest failed first and the instructions require stopping on any
non-pre-existing T2 failure.

### mypy torchlens/

Result: not run because pytest failed first and the instructions require stopping on any
non-pre-existing T2 failure.

## Deferred Finalize Tasks

- CHANGELOG Unreleased consolidation.
- `.research/post-backward-megasprint_SUMMARY.md`.
- `.research/post-backward-megasprint_STATE.md` transition to DONE.
- Merge-ready finalization after full T2 pass.
