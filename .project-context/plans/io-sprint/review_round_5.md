# Adversarial Review — I/O Sprint Plan Round 5

## Verdict

**YELLOW**

Round 5 fixes the Round-4 issues at the design-fork level, and I did not find a new RED-class architecture hole in Fork L, Fork M, the unconditional step-19 `.activation_ref` attachment, or the `expected_sha256` addition. But the plan is still not fully GREEN because a few implementation-facing sections still contradict those new decisions.

## Round 4 Verification

| Finding | Status | Assessment |
| --- | --- | --- |
| R4-1 Validation broken on portable bundles because `func_applied` is dropped | **PARTIAL** | Fork L picks the right rule: portable bundles do not support validation/replay. But the plan still contains contradictory "validation-ready bundle" / "lazy-only" language. See Finding 1. |
| R4-2 No resave contract for `materialize_nested=False` mode | **RESOLVED** | Fork M now explicitly rejects `torchlens.save()` when nested `BlobRef`s are still present and tells the user to run `rehydrate_nested()` first. |
| R4-3 Step-19/20 split left keep-in-memory path un-disk-backed | **PARTIAL** | Fork D now says step 19 always attaches `.activation_ref`, which is the right fix. But IO-S5 still assigns ref patching to step 20. See Finding 2. |
| R4-4 Public API sketch + risk register lagged v4 design | **PARTIAL** | `LazyActivationRef` now includes `expected_sha256`, which fixes the main API sketch gap. But a few bookkeeping sections still describe pre-v5 behavior. See Finding 3. |

## Findings

### 1. MEDIUM — Fork L is not propagated consistently; the plan still promises validation-oriented portable bundles in several places.
**Location:** `plan.md` lines 65, 95, 100, 281-284, 510-511, 612.

**Failure scenario:** An implementer follows the new Fork L section and adds a bundle-load guard, but then also follows the earlier guidance that `include_captured_args=True` is for "validation-ready bundles" and the S6 test text that says validation should raise only on "lazy-only" logs. That yields an implementation where eagerly loaded portable bundles still try to validate, or the docs still tell users bundle validation is supported if they include enough metadata. That directly contradicts Fork L's actual rule: portable-loaded ModelLogs never support `validate_forward_pass()` or `validate_saved_activations()`.

**Recommendation:** Remove the stale validation language everywhere it still appears:
- line 95 should stop saying `include_captured_args=True` is for validation-ready bundles;
- line 100 should stop describing eager nested materialization as useful for validation on portable loads;
- line 65 should stop implying dropped `LayerLog.func_applied` is recoverable for replay on portable bundles;
- S6 should say validation raises on **portable-loaded** ModelLogs, not just "lazy-only" ones.

### 2. MEDIUM — IO-S5 still gives step 20 responsibility that Fork D moved to step 19.
**Location:** `plan.md` lines 147-149, 489-492, 614.

**Failure scenario:** An implementer follows the spec decomposition rather than the higher-level fork prose. IO-S5 still says step 20 "Patches `LazyActivationRef` with final (post-rename) path." If implemented literally, `keep_activations_in_memory=True` logs will finalize the bundle but return without `.activation_ref`, which recreates the exact ambiguity Round 4 flagged.

**Recommendation:** Make IO-S5 match Fork D exactly:
- step 19 should explicitly attach `.activation_ref` with final path and `expected_sha256` for every streamed blob;
- step 20 should only null `.activation` when `keep_activations_in_memory=False`.

### 3. LOW — Several bookkeeping sections still lag the current spec.
**Location:** `plan.md` lines 455, 483, 511, 551, 561, 599.

**Failure scenario:** Test planning drifts because the plan disagrees with itself:
- S1 owns the `PORTABLE_STATE_SPEC` completeness lint, but S7/R2 still describe an older `FIELD_ORDER`/S7 lint model.
- Fork H is now two-layer drift protection, but R12 only mentions manifest-sha checking.
- The version-policy table has 13 rows, while S4 says "all 12 version-policy rows" and the success criteria say "All 14 version-policy rows tested."

**Recommendation:** Sync the bookkeeping sections to the actual v5 design before implementation starts so the tests are derived from one consistent source of truth.

## Summary

This is still close. The actual Round-4 fixes are now present, and I do not see a new structural blocker. But the plan is not GREEN yet because the implementation sections still contain contradictory instructions in two of the exact areas v5 was supposed to settle: validation on portable bundles and step-19/20 ownership. One cleanup pass should be enough.
