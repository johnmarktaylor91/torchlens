# Post-Backward Megasprint Finalize Done

created_at: 2026-05-12T18:11:00-04:00
state: DONE
branch: codex/post-backward-megasprint
final_review_base_sha: ffa73f6ebfe6dcdf4f1a4532dad7ebb9ff331601
last_code_fix_sha: 54ff9f2
ready_for_review_merge: yes
suggested_merge_strategy: review, then squash or merge per project policy

## Final Regression Sweep

The requested full T2 sweep without early stop was run before fixes:

```bash
pytest tests/ -m "not slow" --tb=no -q 2>&1 | tee /tmp/t2_sweep.log
```

It found 12 non-pre-existing failures. Those were fixed across tabular export,
multi-output parameterized-op invariants, module-containment snapshots, and a
stale API-budget assertion.

The full T2 sweep was rerun after fixes:

```text
pytest tests/ -m "not slow" --tb=no -q
2228 passed, 24 skipped, 211 deselected, 2 xfailed, 942 warnings in 1549.81s
```

## Quality Gates

```text
ruff check torchlens/ --fix
All checks passed!

mypy torchlens/
Success: no issues found in 203 source files

pytest tests/ -m smoke -x --tb=short
199 passed, 2266 deselected, 119 warnings in 37.56s
```

Whole-tree ruff was intentionally not run because the known pre-existing
notebook syntax blocker remains in `notebooks/torchlens_in_10_minutes.ipynb`
cell 27.

## Finalize Artifacts

- `CHANGELOG.md` Unreleased notes consolidated for post-backward behavior and
  API surface.
- `.research/post-backward-megasprint_SUMMARY.md` records the final sprint
  summary and gate outputs.
- `.research/post-backward-megasprint_STATE.md` is marked DONE.

## Ready For Review

The branch is ready for review. `final_review_base_sha` records the finalized
branch state immediately before this marker commit; the marker commit itself is
the terminal finalize artifact.
