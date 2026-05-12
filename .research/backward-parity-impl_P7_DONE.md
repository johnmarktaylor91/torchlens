# Phase P7 BLOCK -- Final wrap-up

blocked_at: 2026-05-12T06:39:21-04:00
branch: codex/backward-parity-P1-alpha3
head_before_p7_block_commit: bcfe094bcb9bd0888e62a16a32cc468861f81a94
commits_since_main_before_p7_block_commit: 29
files_touched_since_main_before_p7_block_commit: 63
loc_delta_before_p7_block_commit: 5573 insertions(+), 260 deletions(-)
ready_for_review_merge: no
suggested_merge_strategy: blocked until T2 is green

## Result

P7 did not declare the sprint done. The required T2 gate failed on the first
command, so the remaining wrap-up tasks were intentionally skipped per the P7
instruction: "If ANY test fails, STOP and write a BLOCK note instead of
declaring the sprint done."

## T2 gate

Command:

```bash
pytest tests/ -m "not slow" -x --tb=short
```

Result: FAIL.

Pytest collected 2399 items, deselected 209, skipped 1, and selected 2190.
It stopped after 1 failure:

```text
tests/test_activation_postfunc_refactor.py::test_train_mode_out_postfunc_detach_rejected
FAILED: DID NOT RAISE <class 'torchlens._training_validation.TrainingModeConfigError'>
```

Full failure excerpt:

```text
_________________ test_train_mode_out_postfunc_detach_rejected _________________
tests/test_activation_postfunc_refactor.py:136: in test_train_mode_out_postfunc_detach_rejected
    with pytest.raises(tl.TrainingModeConfigError, match="disconnected from the autograd graph"):
E   Failed: DID NOT RAISE <class 'torchlens._training_validation.TrainingModeConfigError'>
```

Final pytest summary:

```text
===== 1 failed, 5 passed, 1 skipped, 209 deselected, 58 warnings in 1.98s ======
```

The remaining T2 commands were not run because the first T2 gate failed:

```bash
ruff check torchlens/ --fix
mypy torchlens/
```

## Tasks not performed

- CHANGELOG consolidation was not performed.
- `.research/backward-parity-sprint_SUMMARY.md` was not written.
- `.research/backward-parity-sprint_STATE.md` was not marked DONE.
- Optional sample gallery refresh was not performed.
- Final review/merge readiness was not declared.

## Phase summaries from prior commits

- P1: LiveOpRecord IR and postprocess Step 0 materialization, including the
  `has_op` to `is_intervening` rename and AccumulateGrad attribution support.
- P2: Backward validation metadata plumbing and related public API forwarding.
- P3: Combined forward/backward visualization groundwork and attribution for
  backward graph display.
- P4: Backward selector DSL and gradient helper parity.
- P5: Fastlog backward gradient capture, `gradient_postfunc` aliasing, and
  `aggregate(target="grad", loss_fn=...)` support.
- P6: Observer direction support, backward bundle visualization regression
  coverage, validation hardening, and deferred per-layer-grad oracle tracking.

## Assumptions

- I treated the T2 failure-stop instruction as stronger than the general
  cleanup instruction to consolidate docs, because declaring DONE after a known
  non-slow failure would be misleading.
- I did not modify or revert unrelated dirty work already present in the
  worktree.

## Suggested next step

Fix the regression where `train_mode=True` plus a detaching `out_postfunc`
does not raise `TrainingModeConfigError`, then rerun the full P7 T2 command
sequence before resuming changelog, summary, STATE DONE, and final branch
review notes.
