# Phase P7 BLOCK -- Final wrap-up

blocked_at: 2026-05-12T06:53:44-04:00
branch: codex/backward-parity-P1-alpha3
head_before_p7_block_update: 20b7fd0cc357001915f569f202c5af2ca2d784ad
commits_since_main_before_p7_block_update: 31
files_touched_since_main_before_p7_block_update: 65
loc_delta_before_p7_block_update: 5714 insertions(+), 260 deletions(-)
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
tests/test_api_surface.py::test_all_size_exactly_40
FAILED: AssertionError: assert 46 == 40
```

Full failure excerpt:

```text
___________________________ test_all_size_exactly_40 ___________________________
tests/test_api_surface.py:82: in test_all_size_exactly_40
    assert len(torchlens.__all__) == 40
E   AssertionError: assert 46 == 40
E    +  where 46 = len(['trace', 'fastlog', 'load', 'save', 'do', 'replay', ...])
E    +    where ['trace', 'fastlog', 'load', 'save', 'do', 'replay', ...] = torchlens.__all__
```

Final pytest summary:

```text
===== 1 failed, 74 passed, 1 skipped, 209 deselected, 66 warnings in 2.13s =====
```

The current top-level `torchlens.__all__` has 46 names. The six sprint-added
public names causing the budget mismatch are:

```text
grad_fn
intervening
grad_fn_label
grad_clip
grad_noise
grad_clamp
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
- P7B: Restored train-mode `out_postfunc` validation after the earlier P7 block,
  but the full T2 rerun is now blocked by API-surface budget drift.

## Assumptions

- I treated the T2 failure-stop instruction as stronger than the wrap-up
  instruction, because marking the sprint DONE after a known non-slow failure
  would be misleading.
- I did not modify or revert unrelated dirty work already present in the
  worktree.

## Suggested next step

Decide whether the Phase 1a top-level API budget should remain exactly 40 or
be updated to include the six backward-parity public names. Then rerun the full
P7 T2 command sequence before resuming changelog, summary, STATE DONE, and final
branch review notes.
