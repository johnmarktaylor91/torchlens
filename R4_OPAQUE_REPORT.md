# Round-4 fix: non-tensor input-witness honesty (opaque leaf + bit-exact float)

Branch: `r4fix/opaque-leaf` (off local main `c7584463`). Two related CRITICAL
input-witness honesty holes, both silent-wrong-result / false VERIFIED+ATTESTED tripwire
failures, fixed at the source. Files touched: `torchlens/_io/runnable.py`,
`torchlens/_runnable_execution.py`, `tests/test_tlspec_runnable_opaque_input.py`.

---

## FINDING 1 -- opaque / non-finite-float non-tensor input leaf (my round-4 hunt find)

**Bug:** the round-3 fix witnessed only *grammar-encodable* non-tensor inputs
(bool/int/finite-float/str/None). A leaf **outside** the frozen literal grammar -- enum,
`bytes`, `set`, `complex`, numpy scalar, or a non-finite `inf`/`nan` float -- was witnessed
*value-free* (`encodable=False`, `value=None`) and then never compared. `witness_completeness`
stayed `COMPLETE`, so a changed opaque leaf that steered unobserved Python control flow made
the run report `VERIFIED` + `ATTESTED` over a numerically WRONG replayed path, `poisoned=False`,
zero diagnostics.

**Root cause:** `_io/runnable.py` appended `_input_literal_witnesses` *after* `completeness`
was computed and never downgraded it; value-free leaves are `continue`d in
`_runnable_execution.py:_input_literal_contract_checks`, and
`_path_faithfulness` returns `VERIFIED` whenever `witness_completeness is COMPLETE`.

**Fix (narrow, reuses existing incomplete-witness machinery -- no new public API):**
- `_io/runnable.py` `_input_literal_witnesses` now returns `(witnesses, saw_opaque_leaf)`;
  `saw_opaque_leaf` is `True` whenever any leaf is not grammar-encodable.
- At the producer call site, when `saw_opaque_leaf` and `completeness is COMPLETE`, downgrade
  to the existing `WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE` (an opaque leaf can steer
  an unobserved predicate we cannot re-verify). It does NOT fail producer preflight, so the
  artifact still saves and runs -- it just reports honestly.
- `_runnable_execution.py` `_numeric_attestation_check` now short-circuits to `NOT_APPLICABLE`
  when `descriptor.witness_completeness is not COMPLETE` (incomplete coverage can only be
  UNVERIFIABLE, so a byte-exact attestation would dishonestly bless a possibly-wrong path).

**Result:** `_path_faithfulness` (unchanged mechanism) now returns `UNVERIFIABLE` and
attestation is `NOT_APPLICABLE`, `poisoned=True`, for any opaque/non-finite-float leaf --
whether the leaf changed or not (an unchanged opaque leaf still *runs*, honestly UNVERIFIABLE,
never a false VERIFIED).

**Before/after (repros `/tmp/repro_opaque_leaf.py`, `/tmp/repro_opaque_details.py`):**

| case | before | after |
|------|--------|-------|
| changed enum leaf | VERIFIED + ATTESTED, output `[3,4]` vs true `[20,30]` | UNVERIFIABLE + NOT_APPLICABLE, poisoned |
| changed `inf` float leaf | VERIFIED + ATTESTED, wrong | UNVERIFIABLE + NOT_APPLICABLE, poisoned |
| unchanged enum leaf | VERIFIED + ATTESTED | UNVERIFIABLE + NOT_APPLICABLE (honest: can't attest) |
| `witness_completeness` | COMPLETE | INCOMPLETE_UNOBSERVED_PREDICATE |

---

## FINDING 2 -- signed-zero / NaN finite-float witness (rival Codex find, folded in)

**Bug:** `_runnable_execution.py:_literal_leaf_equal` compared floats with `==`. So
`-0.0 == +0.0` is `True` -- a changed sign-bit input steering control flow (e.g.
`math.copysign(1.0, direction) < 0`, `1/x`) passed the witness and false-VERIFIED+ATTESTED the
wrong replayed path. (`nan == nan` is `False`, which would false-diverge an unchanged NaN, but
Finding 1 already routes non-finite floats to UNVERIFIABLE, so that arm is defensive.)

**Root cause:** ordinary `==` is not IEEE-identity-correct at signed zero and NaN.

**Fix:** compare finite floats by their **IEEE-754 bit pattern** via `struct.pack(">d", x)`
(added `import struct`). `-0.0` bytes differ from `+0.0` -> a sign flip is a divergence; a NaN
equals a NaN with the same bits. Bool/int/type-strict logic unchanged; ordinary finite floats
behave exactly as before.

**Before/after (repro `/tmp/torchlens_r4_signed_zero_input.py`):**

| case | before | after |
|------|--------|-------|
| capture `-0.0`, run `+0.0` (copysign branch) | VERIFIED + ATTESTED, output `[12]` vs fresh `[-8]` | DIVERGES (`PathDivergenceError`) |
| capture `-0.0`, run `-0.0` | verified | VERIFIED + ATTESTED (bits match) |
| normal finite float change | diverges | diverges (unchanged) |
| normal finite float unchanged | verify+attest | verify+attest (unchanged) |

Interaction note: an unchanged NaN input no longer false-diverges from a `nan == nan`
self-comparison; per Finding 1 it is opaque, so it runs UNVERIFIABLE (not DIVERGED, not a false
VERIFIED). Test `test_unchanged_nan_leaf_does_not_falsely_diverge` locks this.

---

## Hard-constraint compliance
- A run whose opaque/non-finite input DIFFERS from capture: never VERIFIED/ATTESTED ->
  UNVERIFIABLE + NOT_APPLICABLE only. Output not falsely blessed (poisoned, no attestation
  check emitted).
- Changed signed-zero (finite) input: DIVERGES (matches round-3 finite-float behaviour).
- No regression: encodable bool/int/finite-float/str inputs still diverge-on-change /
  verify+attest-on-unchanged (round-3 suite `test_tlspec_runnable_python_input.py` 5/5 green;
  dedicated no-regression tests added).
- No validation/honesty check weakened; the fix STRENGTHENS coverage (COMPLETE -> UNVERIFIABLE
  for opaque leaves; `==` -> bit-exact for floats). No new public API (reused existing enum
  member + attestation gate).

## New tests (`tests/test_tlspec_runnable_opaque_input.py`, 11)
Finding 1: `test_changed_enum_leaf_is_unverifiable_not_attested`,
`test_unchanged_enum_leaf_runs_unverifiable_not_verified`,
`test_non_finite_inf_leaf_is_unverifiable_not_attested`,
`test_non_finite_nan_leaf_is_unverifiable_not_attested`,
`test_bytes_leaf_is_unverifiable_not_attested`,
`test_set_leaf_is_unverifiable_not_attested`,
`test_encodable_float_path_unregressed`.
Finding 2: `test_signed_zero_change_diverges`,
`test_unchanged_nan_leaf_does_not_falsely_diverge`,
`test_normal_finite_float_change_still_diverges`,
`test_normal_finite_float_unchanged_still_verifies`.

## Verification
- ruff check: All checks passed. ruff format: edited files clean.
- mypy torchlens/: Success, no issues in 326 source files.
- Fast oracle `tests/capture_oracle/ -m "not slow"`: 4 passed.
- New opaque/float suite + round-3 python-input suite: 16 passed.
- Targeted runnable subset + smoke: see below (filled after runs).
- Commit sha: see below.
