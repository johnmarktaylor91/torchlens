# Post-Backward Megasprint Finalize Block

created_at: 2026-05-12T16:55:14-04:00
state: BLOCKED
branch: codex/post-backward-megasprint
head_at_block: b611fe7f4eeb68ac6635da5c3816b8535cdda712
commits_since_main_at_block: 32
loc_delta_at_block: 4279 insertions(+), 243 deletions(-)
files_touched_at_block: 61
ready_for_review_merge: no
suggested_merge_strategy: blocked until T2 is green

## Block Reason

The full T2 sweep stopped on the first command:

```bash
pytest tests/ -m "not slow" -x --tb=short
```

This failed in `tests/test_capture_events_parity.py::test_pickle_byte_equal_pre_m6[tiny_mlp]`.
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
FAILED tests/test_capture_events_parity.py::test_pickle_byte_equal_pre_m6[tiny_mlp]
==== 1 failed, 284 passed, 1 skipped, 211 deselected, 97 warnings in 20.23s ====
```

Failure excerpt:

```text
___________________ test_pickle_byte_equal_pre_m6[tiny_mlp] ____________________
tests/test_capture_events_parity.py:430: in test_pickle_byte_equal_pre_m6
    pytest.fail(f"Trace pickle differs for {model_name}: {diffs[:5]!r}")
E   Failed: Trace pickle differs for tiny_mlp: ["trace.layer_dict_main_keys['input_1'].parent_layer_log: keys ['multi_output_role']", "trace.layer_dict_main_keys['input_1'].parent_layer_log: keys differ ['multi_output_role']", "trace.layer_dict_main_keys['linear_1_1'].parent_layer_log: keys differ ['multi_output_role']", "trace.layer_dict_main_keys['linear_2_3'].parent_layer_log: keys differ ['multi_output_role']", "trace.layer_dict_main_keys['linear_3_5'].parent_layer_log: keys differ ['multi_output_role']"]
```

Likely root cause to investigate:

`multi_output_role` is present on materialized `LayerLog` records in one side of the
pre-M6 pickle parity comparison but absent on the other side. The parity fixture or the
capture/materialization path needs to be updated so both compared traces serialize the same
field set.

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
