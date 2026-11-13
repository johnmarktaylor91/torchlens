# _io/ - Implementation Guide

This package is the untrusted-artifact save/load surface: a loaded `.tlspec` bundle is
hostile input, and every load path here is fail-closed.

## bundle.py
- `save()` and `load()` (typed overloads) are the main entry points behind `trace.save()`/`tl.load()`.
- `_RenameAwareUnpickler` reads `metadata.pkl` through the default-deny `SafeBundleUnpickler`
  allowlist while preserving the locked `_RENAMED_PICKLE_GLOBALS` class/module remap; each
  remapped target is still gated through the allowlist.
- `cleanup_tmp()` removes stale partial-save temp directories.
- `_staging.refuse_stale_load()` runs at the top of `load()` (top-level path only): a sibling
  `<name>.tmp.*` staging folder at least as new as the target refuses `newer_save_unfinished`.
  `_staging.settle_failed_staging()` is how the non-bundle savers keep a failed overwrite's
  staging folder PARTIAL; `sibling_staging_dirs()` is the one escaped, symlink-refusing glob.
- `_root_manifest.load_by_root_manifest()` routes merged, fastlog and extraction directories by
  their root manifest before `load()` parses a trace manifest. Merged, fastlog and current-format
  manifests first pass the release floor (`raise_if_saved_by_older_major`: a `torchlens_version`
  stamp below `FIRST_FORMAT3_RELEASE` refuses); extraction takes it in `load_extraction`.

## _safe_unpickle.py
- `SafeBundleUnpickler` is the restricted default-deny unpickler (class allowlist, mirrors
  `torch.load(weights_only=True)` intent; closes the load-time `__reduce__`/`os.system` RCE).
- Foreign custom callables are NEVER imported at unpickle by default: they load as inert
  `_DeferredForeignCallable` placeholders that fail closed if called. Execution requires the
  explicit `trust_custom_callables=True` opt-in or the narrower
  `allowed_custom_callable_modules` allowlist, which stays enforced even alongside broad trust.
- Trust keys on the resolved object's REAL module, never the pickled path; known-dangerous
  modules (`os`, `sys`, `subprocess`, `builtins`, `importlib`, ...) are hard-denied.
- `format_contract.refuse_pre_3_module_path()` (called from `find_class`) refuses a pickled
  module path only torchlens 2.x wrote, with `ArtifactTooOldError`; that includes any path at or
  below a public package's submodule 3.0 made private (the old types module of `tl.intervention`).
- No file bytes ever reach `torch.load`: `torch.storage._load_from_bytes` (the storage leg of
  torch's default tensor reduction) is refused by name. Plain metadata tensors (annotation
  values, small raw inputs/outputs) are written by `_canonical_pickle.py` through the
  TorchLens-owned inline reduction in `_inline_tensor.py` (raw bytes, dtype, shape,
  `requires_grad`, device), which validates every argument, checks the byte length exactly
  and yields a non-resizable storage. Older files that embedded a tensor refuse typed with a
  re-save remedy.
- The load runs on the C `pickle.Unpickler` behind a static pre-scan of the opcode stream
  (`_pickle_prescan.py`): the pre-scan resolves every global through the same `find_class`
  policy, runs the allocation belt on every statically known construction, classifies every
  BUILD target and refuses stream corruption; the C phase hands REDUCE callables back wrapped
  in a `GuardedCallable` that runs the belt on the real arguments. A stream shape the pre-scan
  cannot classify runs through `_PurePythonBundleUnpickler`, the same policy per opcode on the
  pure-Python machine. The belt text lives once, in `_construction_refusal` / `_build_refusal`;
  the belt's vocabulary (the torch I/O type denylist, the storage and tensor constructor sets,
  `_alloc_refusal_reason`, `_SAFE_EXPLICIT_GLOBALS`, `_safe_getattr`) lives in `_alloc_belt.py`
  and is re-exported by `_safe_unpickle.py` for the hardening tests that pin it.
  `tests/test_unpickle_c_path.py` is the equivalence proof between the two machines. The
  fast-path unpickler must die with its load: the C memo is not traversed by the cyclic
  collector on every supported Python, so `load()` clears it and the dry-run planner releases
  its resolver and probe classes when the dry run ends (a retained unpickler pinned every loaded
  object through its owner's first `gc.collect()`).
- The opcode-count ceiling (`bundle._MAX_METADATA_PKL_OPCODES`) is enforced by the pre-scan on
  every load through `max_opcodes=`; the load sites map `OpcodeCeilingExceeded` to the
  `metadata_object_count_exceeded` artifact error.

## manifest.py / tlspec.py
- `Manifest`, `TensorEntry`, `Provenance`; `enforce_version_policy()` gates artifact versions;
  `torchlens._utils.hashing.sha256_file()` hashes blobs.
- `_TlSpecWriter` writes unified bundles; `coerce_tlspec_save_level()` validates save levels.

## scrub.py / rehydrate.py / accessor_rebuild.py

`_scrub_identities.py` holds the pure metadata identity normalization and record joins used by `scrub.py`.
- `scrub_for_save()` turns live trace state into portable state (tensor blobification via
  `BlobSpec`, policy-driven per-field handling).
- `rehydrate_trace()` / `rehydrate_nested()` rebuild loaded state; `rebuild_trace_accessors()`
  restores accessor objects.

## runnable.py / runnable_load.py
- Save side: `build_sparse_run_descriptor()`, `with_weight_payload()`, `with_activation_payload()`,
  `sparse_descriptor_to_json()`.
- Load side: `parse_sparse_run_descriptor()`, `preflight_sparse_run_descriptor()`,
  `attach_sparse_run_readiness()`, `validate_witness_obligations()`.
- `runnable_coherence.py`: r83 registry-coherence names (`_callable_registry_contradiction()`;
  a private `torch._C._fft`/`_linalg`/`_special` key also accepts its public binding name).
- Contract of record: `docs/reference/runnable_tlspec_contract.md`.

## payload_codec.py / lazy.py / streaming.py
- `PayloadCodec` protocol with per-backend codecs (`TorchPayloadCodec`, `JaxPayloadCodec`,
  `TinygradPayloadCodec`, `MlxPayloadCodec`, `PaddlePayloadCodec`, `TFPayloadCodec`,
  `NullPayloadCodec`); resolve via `get_payload_codec()`, extend via `register_payload_codec()`.
  A codec author MUST route device tokens through `_artifact_strings.py`'s
  device-token gate — it is mandatory, not advisory (it closed an
  arbitrary-file-write via a `disk:<path>` tinygrad device string).
- `LazyActivationRef` (lazy.py) defers tensor payload loads; `BundleStreamWriter` (streaming.py)
  streams blobs to disk.

## Small support modules
- `__init__.py`: `ArtifactError`, `ArtifactTooOldError` (rehydration floor),
  `FieldPolicy`, `read_tlspec_version()`.
- `_artifact_strings.py`: artifact-string validation incl. the MANDATORY
  device-token gate (see payload_codec above).
- `_canonical_pickle.py`: the hash-seed-independent canonical metadata pickler
  (split out of its former host module in fixwave-4).
- `_json.py`: bounded JSON reads (`loads_bounded()`, `load_bounded()`, `read_bounded()`) with
  size/depth refusal on untrusted text.
- `paths.py`: `reject_symlink_path()` and bundle blob path resolution.
- `state_keys.py`: `refuse_callable_shadowing_state_keys()`, `static_class_attr()`.
- `tensor_policy.py`: `is_supported_for_save()` returning `Ok`/`SkipReason`/`FailReason`.
- `_durability.py`: `fsync_file()`/`fsync_dir()`/`fsync_tree()`.
- `_torch_symbols.py`: re-export of `torch_attr` (canonical impl in `utils/_torch_symbols`,
  kept here to avoid import cycles).

## Local invariants
- Fail-closed everywhere: unknown pickle globals raise, oversized/deep JSON refuses,
  symlinked bundle members are rejected.
- Artifacts older than the current tlspec (saved by torchlens 2.x) refuse with `ArtifactTooOldError`
  (drop-not-resurrect; no field-alias ladders).
- The sparse runnable core must stay tensor-value-free:
  `assert_sparse_core_has_no_tensor_payload()` enforces it at descriptor build.

The 3.0 runnable writer uses `format3/runnable.py` and its `runnable_families.py` adapter.
It shares the artifact tensor-key counter with `format3/payloads.py`; the descriptor and
state families live under `runnable/`. The storage and persistence enums live in the
leaf `_field_policy.py`, with the record package preserving its `StorageKind` re-export.
Save preparation lives in `_save_gates.py`; runnable wire storage has no record or execution
binding dependency. Runtime component attachment and family binding belong to
`_format3_runnable_bind.py`, above the pure `format3/runnable.py` carrier.

Save preparation in `_save_gates.py` is L3 bundle assembly. The format 3 runnable carrier consumes prepared JSON and tensor families at L1; it reads the frozen wire version names from the `_runnable` vocabulary through one deferred import in `component_versions()`.

Typed recipe restoration and predicate re-arming share `intervention/_rerun_predicate.py`; `intervention/_save.py` delegates decoding there.

Lazy payload index derivation and BlobRef lookup live with `LazyActivationRef` in `lazy.py`. Resave and nested materialization derive the index from validated format 3 record cells, and resave uses the same codec/chunk writer as eager payloads.

`format3/_record_refs.py` owns the closed row-reference resolution pass used by the reader.
