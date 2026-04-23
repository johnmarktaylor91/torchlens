# Adversarial Review — I/O Sprint Plan Round 4

## Verdict

**YELLOW**

Round 4 genuinely resolves the 9 Round-3 findings. The Round-3 reviewer’s note was directionally right: Findings 1-4 were fixed cleanly, and I did not find a new RED-class architecture hole in the nested rehydration, scrub completeness, streaming finalization split, per-blob drift detection, symlink rejection, or no-shared-readers model.

This is close. I still found one new high-severity contract break and two medium-severity gaps introduced by the v4 revisions, plus one low-severity document inconsistency.

## Round 3 Verification

| Finding | Status | Assessment |
| --- | --- | --- |
| R3-1 Nested tensor rehydration / `map_location` / RAM blow-up | **RESOLVED** | Materialization moved out of `__setstate__` into a loader that accepts `map_location`, and v4 added `materialize_nested` plus safer defaults for captured args / RNG state. |
| R3-2 Scrub surface not exhaustive | **RESOLVED** | `PORTABLE_STATE_SPEC` is a materially stronger contract than the old FIELD_ORDER-based proof, and the completeness lint now lands in S1 instead of S7. |
| R3-3 Step 19 gate contradictory | **RESOLVED** | Finalization and eviction are now split into step 19 / step 20 with distinct gates. |
| R3-4 Manifest-only drift misses blob tampering | **RESOLVED** | v4 adds per-blob sha256 verification before fast-copy. |
| R3-5 `forward_args_summary` misses kwargs | **RESOLVED** | v4 adds `forward_kwargs_summary` plus recursive formatting rules. |
| R3-6 Streaming unsupported-tensor policy undefined | **RESOLVED** | Streaming is now explicitly always strict. |
| R3-7 Symlink handling undefined | **RESOLVED** | v4 now states a reject-on-symlink policy for save/load/cleanup. |
| R3-8 Scrub lint lands too late | **RESOLVED** | The completeness lint is now in S1. |
| R3-9 Reader thread-safety / ownership model undefined | **RESOLVED** | The plan now picks a concrete no-shared-readers model. |

## Findings

### 1. HIGH — The new scrub contract makes “validation-ready bundles” impossible as written because layer callables are dropped with no restoration path.
**Location:** `plan.md` lines 60-65, 95-101, 467-476; `torchlens/validation/invariants.py` lines 397-398; `torchlens/validation/core.py` line 468.

**Failure scenario:** A user follows the v4 guidance and saves with `include_captured_args=True` because they want a validation-ready bundle, then loads it eagerly and calls `validate_forward_pass()`. The plan now explicitly drops `LayerPassLog.func_applied` and `LayerLog.func_applied`. Worse, line 65 says `LayerLog.func_applied` is recoverable from `LayerPassLog` records, but line 60 drops the LayerPassLog callable too. Current validation requires `callable(lpl.func_applied)` and replay executes `layer.func_applied(...)`. As written, a loaded portable bundle cannot satisfy that contract.

**Recommendation:** Pick one explicit rule:
- preserve a restorable callable identity and rebuild `func_applied` on load for supported ops, or
- declare `validate_forward_pass()` unsupported on portable bundles and remove the “validation-ready bundles” claim tied to `include_captured_args=True`.

### 2. MEDIUM — `lazy=True, materialize_nested=False` has no resave / drift contract for nested `BlobRef` payloads.
**Location:** `plan.md` lines 96-101, 392-399, 468-473.

**Failure scenario:** A user loads a bundle with `lazy=True, materialize_nested=False`, leaving nested tensor-bearing fields such as `captured_args`, `func_rng_states`, or `forward_kwargs` as `BlobRef`-like objects. They then call `model_log.save("b/")`. The v4 fast-copy/drift logic only specifies how `LazyActivationRef` blobs are verified and copied. It does not say how remaining nested `BlobRef`s are copied, re-manifested, or rejected. That leaves the expert-mode resave path underspecified for exactly the new mode v4 introduced.

**Recommendation:** State one rule explicitly:
- `torchlens.save()` rejects ModelLogs that still contain nested `BlobRef`s and requires `rehydrate_nested()` first, or
- nested refs must also carry source-bundle + checksum metadata and participate in the same fast-copy / drift-check pipeline as `LazyActivationRef`.

### 3. MEDIUM — The step-19/20 split still leaves the default streaming path internally inconsistent about whether the returned log is actually disk-backed.
**Location:** `plan.md` lines 147-159, 456-457.

**Failure scenario:** In v4, step 20 is the only step that explicitly sets `.activation_ref`, and it only runs when `keep_activations_in_memory=False`. But line 159 says that with the default `keep_activations_in_memory=True`, the returned ModelLog has activations in memory “AND disk-backed.” As written, the keep-in-memory path finalizes the bundle on disk but never says when any `LazyActivationRef` objects are attached to the in-memory log. That matters for later cleanup/materialization/resave semantics.

**Recommendation:** Decide this explicitly:
- step 19 should always attach `activation_ref` for streamed blobs, and step 20 should only null `.activation`, or
- stop claiming the keep-in-memory return value is disk-backed and document that only the bundle on disk is durable.

### 4. LOW — A few remaining sections still describe pre-v4 behavior.
**Location:** `plan.md` lines 381-389 vs. 466-476; lines 516-519.

**Failure scenario:** An implementer following the public `LazyActivationRef` sketch in §3 omits `expected_sha256`, even though S6 requires it for the new per-blob drift check. The risk register also still says nested containers are always eager-materialized and that `cleanup()` closes live handles, both of which v4 explicitly changed.

**Recommendation:** Sync the public API sketch and risk register to the v4 design so the plan has one consistent source of truth.

## Summary

Round 4 did the important work. The Round-3 blockers are actually fixed, and the plan now looks implementable. I would not call it GREEN yet because the bundle-validation claim is still incompatible with the scrub contract, and the new expert nested-lazy path needs one more explicit save/resave rule. Once those are tightened, this plan is plausibly ready to go GREEN.
