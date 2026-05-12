# Post-Backward Megasprint Finalize Block

created_at: 2026-05-12T17:24:10-04:00
state: BLOCKED
branch: codex/post-backward-megasprint
head_at_block: 071e9e4fa1997ecd23afca5cedf2eb85956629fd
commits_since_main_at_block: 36
loc_delta_at_block: 4358 insertions(+), 242 deletions(-)
files_touched_at_block: 66
ready_for_review_merge: no
suggested_merge_strategy: blocked until full T2 is green

## Block Reason

The full T2 sweep stopped on the first command:

```bash
pytest tests/ -m "not slow" -x --tb=short
```

This failed in
`tests/test_io_export.py::test_tabular_exports_round_trip_csv_and_json[module_pass]`.
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
FAILED tests/test_io_export.py::test_tabular_exports_round_trip_csv_and_json[module_pass]
= 1 failed, 979 passed, 7 skipped, 211 deselected, 196 warnings in 174.24s (0:02:54) =
```

Failure excerpt:

```text
__________ test_tabular_exports_round_trip_csv_and_json[module_pass] ___________
tests/test_io_export.py:269: in test_tabular_exports_round_trip_csv_and_json
    surface.to_json(json_path, orient="records")
torchlens/data_classes/module_log.py:424: in to_json
    self.to_pandas().to_json(filepath, orient=orient, **kwargs)
../../anaconda3/envs/py311/lib/python3.11/site-packages/pandas/core/generic.py:2635: in to_json
    return json.to_json(
../../anaconda3/envs/py311/lib/python3.11/site-packages/pandas/io/json/_json.py:241: in to_json
    ).write()
../../anaconda3/envs/py311/lib/python3.11/site-packages/pandas/io/json/_json.py:292: in write
    return ujson_dumps(
E   OverflowError: Maximum recursion level reached
```

Likely root cause to investigate:

`ModulePassLog.to_json()` delegates to `self.to_pandas().to_json(...)`. The `module_pass`
surface now appears to include a recursive or pandas-unserializable object after the recent
module output changes, causing pandas' JSON writer to overflow while serializing records.

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
