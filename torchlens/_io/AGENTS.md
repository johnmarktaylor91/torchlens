# _io/ - Implementation Guide

This package is the untrusted-artifact save/load surface: a loaded `.tlspec` bundle is
hostile input, and every load path here is fail-closed.

## bundle.py
- `save()` and `load()` (typed overloads) are the main entry points behind `tl.save()`/`tl.load()`.
- `_RenameAwareUnpickler` reads `metadata.pkl` through the default-deny `SafeBundleUnpickler`
  allowlist while preserving the locked `_RENAMED_PICKLE_GLOBALS` class/module remap; each
  remapped target is still gated through the allowlist.
- `cleanup_tmp()` removes stale partial-save temp directories.

## _safe_unpickle.py
- `SafeBundleUnpickler` is the restricted default-deny unpickler (class allowlist, mirrors
  `torch.load(weights_only=True)` intent; closes the load-time `__reduce__`/`os.system` RCE).
- Foreign custom callables are NEVER imported at unpickle by default: they load as inert
  `_DeferredForeignCallable` placeholders that fail closed if called. Execution requires the
  explicit `trust_custom_callables=True` opt-in or the narrower
  `allowed_custom_callable_modules` allowlist, which stays enforced even alongside broad trust.
- Trust keys on the resolved object's REAL module, never the pickled path; known-dangerous
  modules (`os`, `sys`, `subprocess`, `builtins`, `importlib`, ...) are hard-denied.

## manifest.py / tlspec.py
- `Manifest`, `TensorEntry`, `Provenance`; `enforce_version_policy()` gates artifact versions;
  `sha256_of_file()` hashes blobs.
- `_TlSpecWriter` writes unified bundles; `coerce_tlspec_save_level()` validates save levels.

## scrub.py / rehydrate.py / accessor_rebuild.py
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
- `__init__.py`: `TorchLensIOError`, `ArtifactVersionBelowFloorError` (rehydration floor),
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
- Artifacts older than tlspec v6 refuse with `ArtifactVersionBelowFloorError`
  (drop-not-resurrect; no field-alias ladders).
- The sparse runnable core must stay tensor-value-free:
  `assert_sparse_core_has_no_tensor_payload()` enforces it at descriptor build.
