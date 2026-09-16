# Runnable TLSPEC Stage 2a Report

## Changes

- Added `FunctionCallRef.func_id: FunctionRegistryKey | None = None` in
  `torchlens/ir/events.py`, preserving all existing construction sites through the `None` default.
- Reused the existing `CapturedArgTemplate.func_id` value in
  `torchlens/backends/torch/ops.py`. The callable key is still derived exactly once, by the existing
  `_build_args_template()` path; Stage 2a does not add or move a registry-key computation.
- Copied `FunctionCallRef.func_id` into cooked `Op.func_id` in
  `torchlens/postprocess/_materialize.py`.
- Added `Op.func_id` to `LAYER_PASS_LOG_FIELD_ORDER`, the Op slot/default-fill state, and
  `PORTABLE_STATE_SPEC` with `FieldPolicy.KEEP` in lockstep.
- Updated the backend-parity FIELD_ORDER/dataframe/schema fixture for the intentional cooked-Op
  schema addition. No capture-oracle golden was modified.
- Excluded only `function.func_id` from the Stage-0 capture oracle's generic nested population
  projection. This keeps runnable identity metadata out of frozen model facts while the sealed event
  and cooked Op retain it.

Committed change size: 7 files, 29 insertions, 9 deletions. The JSON fixture accounts for 11
insertions and 8 deletions (three `func_id` schema entries plus its digest); production and oracle
characterization code account for 18 insertions and 1 deletion.

## Assumptions

- “Reuse the existing computed key” means `func_id` is populated only where the pre-existing
  argument-template path already computes `CapturedArgTemplate.func_id`. Source/synthetic paths that
  do not compute a callable key retain the contract default of `None`; no second computation was
  introduced.
- The explicit Stage 2a cooked-Op schema change requires the backend-parity schema fixture to gain
  `func_id`. This fixture is distinct from the frozen Stage-0 capture-oracle goldens, which remained
  byte-identical and untouched.

## Test results

- `ruff check . --fix`: passed. Final `ruff check .`: passed.
- `mypy torchlens/`: passed — `Success: no issues found in 321 source files`.
- Slow capture oracle: 30 passed, 4 deselected; all committed `ground_truth` values matched with zero
  diff. No capture-oracle golden was regenerated or edited.
- Non-slow capture oracle: 4 passed, 30 deselected.
- Smoke suite: 337 passed, 5 skipped, 4,731 deselected.
- FIELD_ORDER/schema/tlspec/save-load/round-trip selection: 244 passed, 9 skipped, 4,820 deselected.
- Focused field-policy/schema coverage: 50 passed.
- Direct `.tlspec` probe: two ordinary compute-op `FunctionRegistryKey` values populated and survived
  save/load equality.
- Full non-slow suite completed with full output in
  `/tmp/torchlens-stage2a-not-slow.txt`: 4,686 passed, 105 skipped, 276 deselected, 3 xfailed, and 3
  known non-code failures in 7,064.21 seconds:
  - `tests/test_callbacks_lightning.py::test_maybe_profile_releases_activation_tensor_deterministically`
    — known flaky deterministic-GC assertion; the weak-referenced activation remained live.
  - `tests/test_menagerie_dependency_mapping.py::test_effdet_d6_catalog_row_has_pip_mapping` — known
    menagerie catalog/data construction failure caused by the deprecated `torch.stft(...,
    return_complex=False)` warning being raised.
  - `tests/test_menagerie_structural_digest.py::test_structural_fingerprint_is_deterministic_for_menagerie_sample[82]`
    — known menagerie artifact failure caused by the unsupported `FixedCategorical` output-container
    warning being raised.

## Controversial choices

- The backend-parity schema golden was updated because it intentionally fingerprints
  `OP_LOG_FIELD_ORDER`, dataframe columns, and portable-state keys. Excluding `func_id` there would
  weaken the required schema-compatibility gate. The capture oracle instead has a narrow,
  named exclusion for `function.func_id`, preserving all 30 frozen ground-truth payloads exactly.

## Concerns

- None for Stage 2a. The sparse producer, descriptor/registry/DAG emission, `run()`, and
  `load_state_dict()` remain deliberately unimplemented for Stage 2b and later work.

## Knowledge

- The live callable key already existed only inside `CapturedArgTemplate`; promoting that same object
  through `FunctionCallRef` is sufficient for cooked persistence without a new resolver call.
- Synthetic input/buffer field dictionaries bypass callable-template capture, so the event projection
  must accept the `None` default for those entries.
- The Stage-0 oracle recursively characterizes nested dataclass population. A field added to
  `FunctionCallRef` therefore needs an explicit oracle-fact exclusion even when no identity projection
  directly reads it.

## Commit

- `a5fd2c4a9c51d5c623b3b5eb82ffd52ed17b820a` —
  `feat(tlspec): carry callable ids through capture spine`

No push was performed. `STAGE2A_REPORT.md` is intentionally untracked and was not committed.
