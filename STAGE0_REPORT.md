# Stage 0 capture-characterization oracle report

## Result

The Stage 0 capture oracle now characterizes all 30 matrix cases without worker crashes. The
goldens were regenerated in full from this worktree, the no-update differential suite passes,
and the repository smoke suite shows no capture-semantics regression. No files under
`torchlens/` were changed.

Import verification resolved `torchlens.__file__` to:

```text
/home/jtaylor/.claude/worktrees/torchlens-cu-stage0/torchlens/__init__.py
```

## Matrix

| Model axis | Cases | Configurations |
| --- | ---: | --- |
| `plain_cnn` | 14 | `exhaustive`, `predicate_live`, `record`, `two_pass_negative`, `lookback_trace`, `intervene_trace`, `intervene_record`, `halt_trace`, `halt_record`, `backward_trace`, `backward_record`, `disk_exhaustive`, `disk_predicate`, `disk_record` |
| `train_batchnorm` | 3 | `exhaustive`, `predicate_live`, `two_pass_negative` |
| `recurrent` | 3 | `exhaustive`, `record`, `two_pass_negative` |
| `conditional` | 2 | `exhaustive`, `predicate_live` |
| `in_place` | 2 | `exhaustive`, `predicate_live` |
| `mutating_pre_hook` | 2 | `exhaustive`, `two_pass_negative` |
| `tiny_transformer` | 2 | `exhaustive`, `record` |
| `failing_conditional` | 2 | `failed_trace`, `failed_record` |
| **Total** | **30** | |

The matrix covers exhaustive capture, live selectors, recording, structure-dependent two-pass
selection, lookback, intervention, halt, backward capture, disk storage, recurrent execution,
data-dependent control flow, in-place mutation, pre-hook mutation, train-mode buffer mutation,
and failed/partial capture.

## Fixes

### Saved-activation payload guard

`_payload_projection` now checks `has_saved_activation` before reading `op.out`. Unsaved ops in
the lookback trace therefore remain topology-only observations instead of triggering the strict
payload accessor's `ValueError`. The other raw activation read used to seed backward capture is
also explicitly guarded. The audit found no unguarded `transformed_out` reads; event snapshots
read metadata only, and recording/gradient payload reads were already conditioned on payload
presence.

`plain_cnn__lookback` now records a complete outcome with one user-forward invocation.

### Capture failure recording

`_capture_once` now catches exceptions raised by `_run_capture`, recovers a partial recording or
partial trace when TorchLens attached one, and otherwise keeps `product=None`. All projections
tolerate an absent product. `_outcome_projection` faithfully records the exception type and
message, so an unexpected failure in any ordinary case changes strict ground truth and fails the
golden comparison instead of crashing its worker.

For `train_batchnorm__two_pass_negative`, the recorded Stage 0 reading is:

- outcome: `failed`
- exception: `ValueError`
- message identifies a changed computational graph and `save_new_outs` failure
- user-forward invocations: `2`
- producer history: `exhaustive`, then `fast`

## Wart classification

Only these four categories appear under `expected_to_change`:

1. `producer_path`: internal producer identity is migration mechanics, not a model fact.
2. `two_pass_double_execution`: negative-index selection currently replays the user forward;
   its invocation totals and resulting observable side effects must fall to exactly once.
3. `predicate_lossy_event_fields`: the predicate producer currently loses or defaults parameter,
   argument-edge, and related richness metadata; migration should replace those sentinel values
   with faithful data.
4. `stateful_two_pass_outcome`: present only for
   `train_batchnorm__two_pass_negative`. Its second pass sees BatchNorm running-state mutations,
   diverges, and fails. Exactly-once migration should remove the second pass and produce a clean
   single-pass success, which is a wart fix rather than a regression.

The BatchNorm carve-out moves only that case's `outcome` from `ground_truth`; the current failure
is pinned in its golden. The comparator accepts a changed reading only when it is a complete,
non-failed, non-halted outcome, the `fast` producer is gone, and the two-pass side-effect wart
shows one forward invocation. A different failure remains rejected. Every other case keeps its
outcome in strict `ground_truth`.

Executed operations, topology, shapes, dtypes, values, payload identities, gradients, disk
facts, stable side effects, and all non-carved-out outcomes remain strict ground truth. A
dedicated test asserts that no other case owns the stateful-outcome carve-out.

## Invocation-count pins

The regenerated goldens record:

- all four `two_pass_negative` cases: `forward_invocations == 2`
- all live-selector (`predicate_live`) cases: `forward_invocations == 1`
- `plain_cnn__two_pass_negative`: complete success with two invocations
- `recurrent__two_pass_negative`: complete success with two invocations
- `mutating_pre_hook__two_pass_negative`: complete success with two invocations
- `train_batchnorm__two_pass_negative`: recorded failure with two invocations

This preserves the load-bearing Stage 0 distinction between replay-based selection and live
single-pass selection.

## Non-vacuity demonstration

A temporary harness-only perturbation replaced the projected final-op `layer_type` with
`capture_oracle_non_vacuity_probe` for `plain_cnn__exhaustive`. The no-update golden comparison
failed at strict `actual["ground_truth"] == golden["ground_truth"]` with exit status 1. After
reverting the perturbation, the same case passed, and the probe string was confirmed absent.
No golden was modified during this experiment.

The permanent unit-level non-vacuity test also mutates an in-memory ground-truth event identity
and confirms that the comparator raises `AssertionError`.

## Golden and test results

- Golden regeneration:
  `30 passed, 4 deselected, 1 warning in 293.06s`; 30 JSON goldens exist, with no missing or
  extra matrix cases and no worker crashes.
- Full oracle, final no-update run:
  `34 passed, 1 warning in 297.11s`.
- Targeted wart-boundary, invocation-pin, and permanent non-vacuity checks:
  `3 passed, 31 deselected, 1 warning in 0.14s`.
- Explicit temporary non-vacuity probe: expected failure,
  `1 failed, 1 warning in 9.41s`, exit status 1; after revert,
  `1 passed, 1 warning in 9.21s`.
- Ruff: `All checks passed!` A pre-existing malformed-`noqa` warning was reported in
  `torchlens/backends/torch/prehook_provenance.py`; it is outside this task's scope.
- Mypy: `Success: no issues found in 314 source files`.
- Smoke suite:
  `337 passed, 5 skipped, 4717 deselected, 57 warnings in 274.14s`.

One initial full-oracle verification run exposed a stale structural assertion that expected
`predicate_live` to report the `predicate` producer even though all current trace-live cases
faithfully report `exhaustive`. The assertion was corrected to the observed Stage 0 reading; it
was not removed or relaxed. The final full-oracle result above is from the rerun after that fix.

## Assumptions and concerns

- The explicit request for this root report is treated as the sole allowed artifact outside
  `tests/capture_oracle/`.
- TorchLens warning output during the BatchNorm failure is retained as current behavior; the
  worker's final JSON line remains parseable.
- Tracking medians are intentionally broad relative checks and are not semantic gates.
- No capture semantics, public API, or files under `torchlens/` were modified.
