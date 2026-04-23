# Adversarial Review — I/O Sprint Plan Round 2

## Verdict

**RED**

Round 2 is materially better than Round 1, but it does **not** actually resolve all 23 prior findings. The largest remaining gap is the bundle contract for **non-activation tensor fields**: the plan now says it will scrub tensors out of `captured_args`, `children_tensor_versions`, RNG state, and module forward args, but it never defines how those blobs are rehydrated or eagerly materialized on load. That is still a format-design hole, not an implementation detail.

Several revisions also introduced new ambiguity: the new eviction step has no exact placement in the current 18-step postprocess pipeline, lazy refs appear to be created before the temp-dir rename, the default save policy still silently drops unsupported tensors, and the new `pack_bundle` utility is underspecified.

## Round 1 Verification

| # | Orig Sev | Status | Assessment |
|---|----------|--------|------------|
| 1 | CRITICAL | **PARTIAL** | Scrubbing is broader, but the plan still lacks a rehydration/materialization contract for nested tensor fields and still omits some arbitrary-object fields. See Findings 1-2. |
| 2 | CRITICAL | **PARTIAL** | Eviction no longer happens in `save_tensor_data()`, but the new postprocess hook still has no concrete step number or state-transition contract. See Finding 3. |
| 3 | CRITICAL | **RESOLVED** | Gradient streaming was cut; post-hoc gradient bundling is now the stated scope. |
| 4 | CRITICAL | **RESOLVED** | Tar/tar.gz was removed from the core save/load path. |
| 5 | CRITICAL | **RESOLVED** | Opaque blob ids replace raw layer labels in filenames. |
| 6 | CRITICAL | **PARTIAL** | Non-tensor `activation_postfunc` is only addressed for streaming, and even there the enforcement point is internally inconsistent. See Finding 4. |
| 7 | HIGH | **RESOLVED** | `.activation` vs `.activation_ref` split is now coherent. |
| 8 | HIGH | **PARTIAL** | Default-lazy load still changes the default behavior of loaded logs while the compatibility audit is narrow. See Findings 1 and 5. |
| 9 | HIGH | **RESOLVED** | `safetensors` is now required for the bundle path. |
| 10 | HIGH | **PARTIAL** | The manifest is much stronger, but checksum scope and some index semantics are still underspecified. See Finding 7. |
| 11 | HIGH | **RESOLVED** | Temp-dir uniqueness and local-filesystem atomicity scope are now stated. |
| 12 | HIGH | **RESOLVED** | Streaming is restricted to the exhaustive pass; the known-bad fast path is no longer reused. |
| 13 | HIGH | **PARTIAL** | The supported-tensor matrix exists, but the default policy still silently emits partial bundles. See Finding 6. |
| 14 | HIGH | **PARTIAL** | Plain-pickle support is more explicit, but line 42 says it is "not deprecated" while line 420 promises a deprecation warning for old plain pickles. The compatibility contract is still fuzzy. |
| 15 | HIGH | **PARTIAL** | A version-policy table exists, but nightly/pre-1.0/interpreter edge cases are still unspecified. See Finding 7. |
| 16 | HIGH | **UNRESOLVED** | The operational-shape problem remains: directory bundle plus one blob per tensor is still the default. `pack_bundle` only papers over it after the fact. See Finding 11. |
| 17 | HIGH | **PARTIAL** | Corruption coverage is broader, but exact exception contracts are still not specified. See Finding 12. |
| 18 | HIGH | **RESOLVED** | `frames/` was removed from the bundle layout. |
| 19 | MEDIUM | **PARTIAL** | `ModulePassLog.to_pandas()` was added to scope, but the plan still does not say how it preserves forward-arg summaries after current GC clears them. See Finding 9. |
| 20 | MEDIUM | **RESOLVED** | Dependencies and LoC expectations are more realistic than v1. |
| 21 | MEDIUM | **RESOLVED** | `compression="zstd"` was removed. |
| 22 | MEDIUM | **RESOLVED** | `cleanup_tmp` now has an owner spec. |
| 23 | MEDIUM | **PARTIAL** | Lazy-bundle re-save semantics are defined for deletion, but not for source-bundle drift or mutation. See Finding 8. |

## Findings

### 1. CRITICAL — The bundle contract still does not define how nested tensor fields are rehydrated or eagerly materialized.
**Location:** `plan.md` lines 31-32, 39, 49-55, 280-281, 326-333.

**Failure scenario:** A user saves a log with `include_captured_args=True` and `save_rng_states=True`, then loads it with `torchlens.load(path, lazy=False)` and calls `validate_forward_pass()`. The plan says tensors inside `captured_args`, `captured_kwargs`, `forward_args`, `forward_kwargs`, `func_rng_states`, and `children_tensor_versions` become `BlobRef` stubs, and `__setstate__` restores those stubs as `LazyActivationRef`. But the only eager-materialization contract in the plan is `LayerPassLog.materialize_activation()`, which touches `.activation` only. Validation and replay code need real tensors inside those nested containers, not activation refs embedded in lists/dicts.

**Recommendation:** Define a real blob-field schema now. At minimum:
- distinguish activation, gradient, captured-arg, child-version, RNG-state, and module-arg blobs;
- store a path-aware stub that can be reinserted into nested containers;
- specify eager-load behavior for **all** tensor-bearing fields, not just `.activation`;
- or, if bundle v1 is activation-only, explicitly exclude validation-state fields from the archival contract.

### 2. HIGH — The scrub contract still leaves arbitrary-object holes.
**Location:** `plan.md` lines 31-37, 68, 280-281.

**Failure scenario:** A layer's `func_config` contains a tensor subclass or user object, or a module's `extra_attributes` contains arbitrary data from module introspection. The scrub contract only names `func_applied`, `activation_postfunc`, `_param_ref`, `FuncCallLocation._frame_func_obj`, and "extra_attributes on ModelLog". It never covers `func_config`, `ModuleLog.extra_attributes`, or other arbitrary-object fields already present in the current object graph. `metadata.pkl` can still fail to serialize or can round-trip inconsistently.

**Recommendation:** Expand the scrub contract from a few named fields to a full allow-list/deny-list over the actual `FIELD_ORDER` surfaces. `func_config` and module `extra_attributes` need an explicit policy: either recursively scrub/normalize them or reject them for portable bundle save.

### 3. HIGH — The new eviction step is still not placed concretely in the pipeline, and the ref path likely races the final rename.
**Location:** `plan.md` lines 80-81, 249-251, 319, 327, 396-397.

**Failure scenario:** During streaming, `_evict_streamed_activations` creates `LazyActivationRef` objects while the writer is still targeting `<path>.tmp.<uuid>`. Step 6 then renames the temp dir to `<path>`. Unless the ref stores the final destination path rather than the live temp path, the returned `ModelLog` now holds stale bundle paths. Separately, the text keeps calling this the "18-step pipeline" while also saying the new step is "appended last", without stating whether it runs before or after `_set_pass_finished`.

**Recommendation:** Name the exact order in the plan. Example: "Step 18 = `_evict_streamed_activations`; Step 19 = `_set_pass_finished`." Also define whether refs are created with the final bundle path or patched after rename.

### 4. HIGH — Non-tensor `activation_postfunc` is still only half-specified.
**Location:** `plan.md` lines 92-100, 303-311, 316-323.

**Failure scenario:** A user saves a log post hoc with `torchlens.save(model_log, path)` after running `activation_postfunc=lambda t: t.cpu().numpy()`. The streaming path has a proposed guard, but the ordinary bundle save path does not. Even within streaming, the text says the error is raised "at forward start" and then immediately switches to a runtime-first-layer check. Those are materially different behaviors for temp-dir cleanup and user surprise.

**Recommendation:** Pick one rule and apply it to both save paths. If bundle v1 is tensor-only, reject non-tensor postfunc output **before any blob is written** in both streaming and post-hoc save.

### 5. HIGH — Default-lazy load is still too aggressive for the audited compatibility surface.
**Location:** `plan.md` lines 45-57, 59-63, 205, 326-333.

**Failure scenario:** A user loads a normal activation bundle with the default `lazy=True` and then uses an existing TorchLens method outside the short audit list. The plan only audits validation, reprs, `to_pandas()`, and cleanup. Any other public or semi-public method that expects tensors still sees `.activation is None` or nested refs and breaks on the new default path.

**Recommendation:** For Phase 4, make `lazy=False` the default unless the compatibility audit is exhaustive and explicit. Lazy load is a useful feature; it should not become the surprising default while the object graph is still tensor-centric.

### 6. HIGH — The supported-tensor matrix still defaults to silent partial bundles.
**Location:** `plan.md` lines 169-183, 191-200, 308-311, 373.

**Failure scenario:** A model produces a sparse, quantized, nested, or DTensor activation. `torchlens.save()` succeeds by default, emits a warning, and silently drops that activation. The user now has a bundle that looks valid but is incomplete by default. That is especially risky for validation, regression testing, and CI artifacts.

**Recommendation:** Flip the default. `strict=True` should be the default for bundle save; lenient skip behavior should be an explicit opt-in for power users who knowingly accept partial archives.

### 7. MEDIUM — Manifest integrity and version semantics still have undefined edge cases.
**Location:** `plan.md` lines 107-122, 128-158, 308-309.

**Failure scenario:** A bundle is produced on a nightly PyTorch build (`2.x.dev...`), by a pre-1.0 TorchLens release, or under a different Python implementation. The version-policy table only sketches major/minor behavior and does not say how these version strings are parsed. Likewise, `sha256` is recorded, but the plan never states whether it hashes the exact safetensors file bytes, a logical tensor payload, or something else.

**Recommendation:** Specify normalized version parsing and checksum scope explicitly. The simplest workable rule is: "sha256 is computed over the exact blob file bytes on disk after write, and version comparisons use normalized PEP 440 parsing where available, else string fallback with warning."

### 8. HIGH — Lazy-bundle re-save still does not protect against sidecar drift.
**Location:** `plan.md` lines 253-268, 311, 331-333, 385.

**Failure scenario:** A user lazy-loads `a/`, another process mutates one sidecar or rewrites the source bundle, and then the user calls `model_log.save("b/")`. The plan's fast path just `shutil.copy`s blobs from the current source bundle if it "exists and is intact". That can yield a new bundle whose copied sidecars no longer match the in-memory metadata originally loaded from `a/`.

**Recommendation:** Record source-manifest digests or mtimes at load time and re-verify them before fast-copy. If the source drifted, either materialize and re-write or fail hard.

### 9. MEDIUM — `ModulePassLog.to_pandas()` still has no concrete data-preservation point.
**Location:** `plan.md` lines 66-68, 287-291.

**Failure scenario:** The current code clears `ModulePassLog.forward_args` and `.forward_kwargs` after `_build_module_logs` to release tensor references. The plan promises pandas rows with forward-arg shape metadata, but never states where that summary is captured before those fields are nulled.

**Recommendation:** Add an explicit summary field to `ModulePassLog` during module-log construction, or compute/export that summary inside `_build_module_logs` before the existing GC step. As written, the plan promises data that the current lifecycle discards.

### 10. MEDIUM — `pack_bundle` reintroduces a new archive surface without a contract.
**Location:** `plan.md` lines 346-352, 382.

**Failure scenario:** Users rely on `torchlens.pack_bundle()` for single-file archival, but tar behavior for symlinks, permissions, zero-length files, sparse files, and overwrite/unpack conflicts is unspecified. The plan removed tar from the core path, then added it back as a utility without defining its semantics.

**Recommendation:** Either cut `pack_bundle` from Phase 4 or specify the tar contract and tests explicitly. Do not smuggle a second archive format back into scope via S8.

### 11. MEDIUM — Round 1 finding #16 is still not actually resolved.
**Location:** `plan.md` lines 14-25, 348, 382-383.

**Failure scenario:** Tiny models still save as multi-file directories, and large models still create one file per tensor by default. `pack_bundle` only helps after the directory already exists; it does nothing for inode pressure or directory-walk cost during save/load.

**Recommendation:** Stop claiming this finding is resolved. Either defer it explicitly or change the default operational shape. Documentation is not the same as resolution.

### 12. MEDIUM — Corruption handling is broader than v1, but the exception boundary is still unspecified.
**Location:** `plan.md` lines 119-121, 338-340, 421, 446.

**Failure scenario:** A truncated safetensors blob raises one backend exception, a corrupted `metadata.pkl` raises another, and a checksum mismatch raises `TorchLensIOError`. Callers now have to guess which exceptions to catch, even though the change log claims exact exception types are part of the spec.

**Recommendation:** Specify the exception contract in the plan: either all bundle-corruption cases are wrapped in `TorchLensIOError`, or the plan lists which backend exceptions intentionally leak through.
