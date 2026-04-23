# Adversarial Review — I/O Sprint Plan Round 3

## Verdict

**RED**

Round 3 is materially tighter than Round 2. Several prior red items are genuinely fixed. The plan is not converged, though: four high-severity design gaps remain in the scrub/reload/streaming contracts, and a few new medium-severity issues were introduced by the v3 revisions.

## Round 2 Verification

| Finding | Status | Assessment |
| --- | --- | --- |
| R2-1 Nested tensor rehydration | **PARTIAL** | v3 finally defines a nested-tensor contract, but it still has a correctness hole around `map_location` and an unbounded RAM blow-up risk because nested tensors are always eager-loaded, even in `lazy=True`. See Finding 1. |
| R2-2 Scrub allow-list holes | **PARTIAL** | Fork A is broader, but the policy/lint still misses real non-`FIELD_ORDER` state and live callables on aggregate objects. See Finding 2. |
| R2-3 Eviction pipeline position | **PARTIAL** | The plan now names "step 19" and final-path refs, but the step gate contradicts the finalize semantics for the default `keep_activations_in_memory=True` workflow. See Finding 3. |
| R2-4 Non-tensor postfunc inconsistency | **RESOLVED** | v3 now states one rule for both streaming and post-hoc save, with explicit pre-write rejection on the post-hoc path. |
| R2-5 Lazy default too aggressive | **RESOLVED** | Flipping `lazy=False` to the default is the conservative choice for the current tensor-centric object model. |
| R2-6 Silent partial bundles | **PARTIAL** | `torchlens.save()` now defaults to `strict=True`, which fixes the original concern for post-hoc bundle save. The streaming path still does not say how unsupported tensors behave. See Finding 6. |
| R2-7 Version parsing + checksum scope | **RESOLVED** | PEP 440 parsing and "sha256 over exact blob file bytes" are now explicit. |
| R2-8 Lazy-resave drift | **PARTIAL** | Manifest-hash drift detection is better than nothing, but it still misses mutated blob bytes when `manifest.json` is unchanged. See Finding 4. |
| R2-9 ModulePassLog summary capture | **PARTIAL** | A preservation point now exists, but it only covers the positional-arg happy path and does not preserve kwargs or nested dict-style inputs coherently. See Finding 5. |
| R2-10 pack_bundle semantics | **RESOLVED** | `pack_bundle` was cut from scope. |
| R2-11 Operational shape | **RESOLVED** | v3 correctly marks the one-file-per-tensor shape as deferred instead of claiming it is solved. |
| R2-12 Exception contract | **RESOLVED** | The plan now clearly states that bundle-layer errors are wrapped in `TorchLensIOError`. |

## Findings

### 1. HIGH — Nested-tensor eager rehydration is still underspecified for `map_location`, and the new default can explode RAM.
**Location:** `plan.md` lines 53-61, 248-253, 332-333, 379-385.

**Failure scenario:** A user loads a large bundle with `torchlens.load(path, lazy=True, map_location="cpu")` where `include_captured_args=True` and many operations captured tensor args, RNG states, or `children_tensor_versions`. Fork A says every nested `BlobRef` is eagerly materialized during `__setstate__`, but `__setstate__` has no `map_location` parameter. S1 simultaneously says `rehydrate.py` does the inverse load work. Those are two different designs. Even if the implementation routes around `__setstate__`, v3 still forces eager load of every nested tensor in both lazy and eager modes, so "lazy" bundles with tens of thousands of captured-arg tensors can still blow RAM immediately.

**Recommendation:** Move all blob materialization out of `__setstate__` and into a loader/rehydrator entrypoint that receives `map_location`. Then either:
- add a separate flag for nested tensor materialization, or
- make bundle-save defaults less aggressive for captured args/RNG state,

so `lazy=True` is not defeated by auxiliary tensor payloads.

### 2. HIGH — The scrub policy is still not actually exhaustive, and the proposed lint is not strong enough to prove it.
**Location:** `plan.md` lines 31-49, 331, 395; `torchlens/data_classes/model_log.py` lines 166-166, 230-230, 246-246, 270-283; `torchlens/data_classes/cleanup.py` lines 58-82; `.project-context/research/io-audit-codex.md` lines 62-62, 90-92, 123-145.

**Failure scenario:** The plan claims an exhaustive scrub policy guarded by `FIELD_ORDER` enumeration, but the current object graph already contains important state outside `FIELD_ORDER`: `ModelLog` internals such as `_optimizer`, `_saved_gradients_set`, `_module_metadata`, `_module_forward_args`, `_module_logs`, `_buffer_accessor`, `_mod_*`; `LayerPassLog.parent_layer_log`; and `LayerLog` extras not listed in `LAYER_LOG_FIELD_ORDER`. On top of that, Fork A only deny-lists `activation_postfunc` and `func_applied` on some classes. `LayerLog` still carries both callables today, copied from the first pass. The result is a portable-bundle scrub pass that can still serialize live callables or transient backrefs, or silently rely on raw pickle success for state that should have been normalized.

**Recommendation:** Stop using `FIELD_ORDER` enumeration as the proof of scrub completeness. Define an explicit serialized-state schema per class, including out-of-band/transient attributes, and deny-list all callables/backrefs/transient build state across:
- `ModelLog`,
- `LayerPassLog`,
- `LayerLog`,
- `ModuleLog`,
- `ParamLog`,
- helper objects such as `FuncCallLocation`.

At minimum, `LayerLog.func_applied`, `LayerLog.activation_postfunc`, `LayerPassLog.parent_layer_log`, and the current non-`FIELD_ORDER` `ModelLog` internals need explicit policy.

### 3. HIGH — Step 19 is internally contradictory for the default streaming workflow.
**Location:** `plan.md` lines 97-106, 368-375, 447.

**Failure scenario:** Fork D says Step 19 is gated on `keep_activations_in_memory=False`, but the same section also says that when `keep_activations_in_memory=True` Step 19 still writes `manifest.json`, writes `metadata.pkl`, and renames the temp dir, only skipping activation eviction. S5 repeats the gated-only-when-false version. Those cannot both be true. If the gate is implemented literally, the default streaming path never finalizes the bundle. If the gate is ignored, the named "eviction" step is doing two unrelated jobs: bundle finalization and optional in-memory eviction.

**Recommendation:** Split the current Step 19 into:
- an always-run bundle-finalization step, and
- a conditional activation-eviction substep.

If you want to keep one step number, the gate must be rewritten so finalization always happens and only the null-out/`LazyActivationRef` patching is conditional.

### 4. HIGH — Manifest-only source-drift detection still misses tampered blob files.
**Location:** `plan.md` lines 174-180, 311-318, 381-385, 435.

**Failure scenario:** A user does `model_log = torchlens.load("a/", lazy=True)`. Another process later overwrites `a/blobs/0000000042.safetensors` with bad bytes but leaves `a/manifest.json` untouched. `model_log.save("b/")` re-hashes only `manifest.json`, sees the same digest, and fast-copies the corrupted blob into `b/`. The v3 defense catches manifest edits, not bundle-content drift.

**Recommendation:** Record and re-verify content, not just manifest bytes. Practical options:
- capture a root digest over manifest + referenced blob checksums at load time, or
- re-hash every source blob named in the manifest before fast-copy, or
- require materialization/rewrite instead of raw copy for lazy resave.

### 5. MEDIUM — `forward_args_summary` does not fully preserve the data that current GC deletes.
**Location:** `plan.md` lines 83-86, 341-344; `torchlens/postprocess/finalization.py` lines 418-424, 691-695.

**Failure scenario:** A module is called with tensor kwargs only, or with a dict/list structure in `forward_args`. `_build_module_logs()` currently clears both `forward_args` and `forward_kwargs`. v3 adds a single `forward_args_summary: str`, but the described format only summarizes positional args. A kwargs-heavy module can end up with a misleadingly empty or incomplete pandas row after GC.

**Recommendation:** Preserve both sides explicitly:
- add `forward_kwargs_summary`, or
- define `forward_args_summary` as a recursive summary over the full `(args, kwargs)` call surface.

The summary format also needs to say how dict/list nesting is represented.

### 6. MEDIUM — The new `strict=True` default fixes post-hoc save, but streaming still has no unsupported-tensor contract.
**Location:** `plan.md` lines 119-135, 236-245, 368-375, 424.

**Failure scenario:** A model is captured with `log_forward_pass(..., save_activations_to=path)`, and one activation is sparse, meta, nested, or a tensor subclass. Fork E clearly defines `strict` behavior for `torchlens.save()`, but the streaming API has no `strict` parameter and S5 never says whether `tensor_policy.py` is consulted mid-pass. One implementation will abort on first unsupported tensor; another will silently skip; a third may write a partial bundle and keep going. v3 claims this finding is fixed, but only the post-hoc save path has an explicit contract.

**Recommendation:** Extend the strict/lenient policy to streaming explicitly. Either:
- add a streaming `strict` flag, or
- state that streaming always behaves as `strict=True`,

and cover it in S5/S7 tests.

### 7. MEDIUM — Symlink handling is still undefined for bundle load and temp-dir cleanup.
**Location:** `plan.md` lines 19-24, 115, 156-160, 255-259, 369-375.

**Failure scenario:** A bundle directory contains symlinked `metadata.pkl` or `blobs/*.safetensors`, or a sibling `<path>.tmp.*` entry is a symlink. The loader and `cleanup_tmp()` are both filesystem-facing operations, but the plan never says whether symlinks are rejected, followed, or cleaned. A symlinked blob escapes the "bundle root only" assumption; a symlinked temp dir can turn cleanup into deletion outside the intended scope.

**Recommendation:** State and test a hard rule: reject symlinks anywhere in bundle roots and in discovered temp dirs. `cleanup_tmp()` should only recurse into real directories directly owned under the expected parent.

### 8. MEDIUM — The promised scrub-policy guard lands too late to protect the scrub work.
**Location:** `plan.md` lines 49, 331, 395, 445-449.

**Failure scenario:** S1 introduces the scrubber; S4 depends on it for bundle save/load; the scrub-policy lint is not scheduled until S7. That means the guard advertised in Fork A is absent during the actual design/build rounds where scrub completeness can regress. If the sprint is implemented incrementally, the "lint-protected" claim is false until the end.

**Recommendation:** Move a minimal scrub completeness test into S1, and keep only cross-feature integration coverage in S7.

### 9. MEDIUM — The plan acknowledges open safetensors handles, but not their thread-safety or ownership model.
**Location:** `plan.md` lines 379-383, 428.

**Failure scenario:** `LazyActivationRef.materialize()` is implemented with cached `safetensors` readers or mmaps, and another thread calls `ModelLog.cleanup()` while materialization is in flight or while another ref still depends on a shared handle. v3 says cleanup closes handles before nulling refs, but does not say whether readers are per-ref, shared, locked, or thread-safe.

**Recommendation:** Pick one concrete model:
- no long-lived shared readers at all, or
- shared readers guarded by locks/refcounts,

and document whether lazy materialization is thread-safe.

## Checks With No Substantive Finding

- **`lazy=False` default itself:** conservative and acceptable for this sprint. The real issue is the unconditional eager load of nested tensor blobs, not the top-level default.
- **Manifest `relative_path` when a bundle directory is moved:** no blocker found if load resolves `relative_path` against the bundle root passed to `torchlens.load()`. A fresh load after moving the directory should still work. The remaining portability issue is only for long-lived in-memory refs after the bundle is moved post-load.
- **"Step 19" numeric ordering:** the current pipeline is imperative code, not a name-based registry. The numbering itself is not the problem. The real issue is the contradictory gate/finalize contract in Finding 3.

## Summary

Round 3 fixed several real issues from Round 2: the lazy default flip, the strict default flip for post-hoc save, the tar/`pack_bundle` scope cut, the version/checksum clarifications, and the explicit exception contract. The plan is still not ready for implementation as written because the scrub surface, nested rehydration contract, streaming finalization gate, and lazy-resave drift protection are not yet sound.

If Findings 1-4 are fixed cleanly, this plan is close enough that the next review could plausibly go GREEN.
