# Final Adversarial Review: `main..codex/io-sprint`

**Verdict:** REQUEST_CHANGES

## Findings

1. **CRITICAL** — `torchlens/_io/bundle.py:1024`, `torchlens/_io/bundle.py:1054`, `torchlens/_io/rehydrate.py:385`, `torchlens/_io/lazy.py:65`

   `manifest.json`'s `relative_path` is trusted as-is and only checked for existence / symlink status. Nothing enforces that it stays inside the bundle root or even under `blobs/`. A malicious manifest can point to `../outside.safetensors` and `tl.load()` will happily read it; I reproduced this locally with a temp fixture. This defeats the sprint's symlink-hardening story and is a real path-traversal bug.

   **Recommendation:** Normalize each manifest path with `resolve()`, require it to stay under `<bundle>/blobs/`, and reject absolute paths or `..` escapes in every load/materialize path (`load`, eager verify, lazy materialize, fast-copy source validation).

2. **HIGH** — `torchlens/user_funcs.py:362`, `torchlens/user_funcs.py:389`, `torchlens/postprocess/__init__.py:199`

   `log_forward_pass(..., layers_to_save=[...], save_activations_to=...)` is broken in the two-pass path. The writer is created during the metadata-only exhaustive pass, finalized at postprocess step 19 before any requested activations are re-saved, and never reattached for the fast pass. Result: the returned log has in-memory activations only, `activation_ref` stays unset, `keep_activations_in_memory=False` is ignored, and the on-disk bundle is empty. I reproduced this locally.

   **Recommendation:** Either reject `save_activations_to` when `layers_to_save` requires the two-pass path, or plumb streaming through the fast pass so the writer is created/finalized around the pass that actually saves activations.

3. **HIGH** — `torchlens/_io/bundle.py:272`, `README.md:205`

   `tl.load()` still does a raw `pickle.load()` of `metadata.pkl`. That means loading an untrusted portable bundle can execute arbitrary code, but the new public save/load docs do not warn about that anywhere visible. Given that portable bundles are now a first-class advertised API, this is a security footgun.

   **Recommendation:** Add an explicit warning to the README and API docstrings that `tl.load()` must only be used on trusted bundles, or replace `metadata.pkl` with a non-executable metadata format.

4. **MEDIUM** — `torchlens/_io/bundle.py:119`, `torchlens/_io/bundle.py:877`, `torchlens/_io/scrub.py:188`

   `save()` rejects any model log containing nested `BlobRef`s before it knows whether those fields are being persisted. That blocks valid flows like `lazy=True, materialize_nested=False` followed by `save(..., include_captured_args=False)` or `include_rng_states=False`, even though scrub would drop those fields anyway. I reproduced this locally.

   **Recommendation:** Make the nested-blob preflight aware of `include_captured_args` / `include_rng_states`, or move the check after effective field-policy resolution so dropped fields do not spuriously fail the save.

5. **LOW** — `tests/test_io_streaming.py:121`, `tests/test_io_streaming.py:143`, `tests/test_io_streaming.py:167`

   The streaming integration suite only covers the single-pass `layers_to_save="all"` path. There is no test for `save_activations_to` combined with selective layer requests, which is why the two-pass regression above was missed.

   **Recommendation:** Add coverage for selective streaming in both `keep_activations_in_memory=True` and `False` modes, and add a traversal test for malicious manifest `relative_path` values.

## Confirmed

- Step 19 runs unconditionally when `_activation_writer` is present, and step 20 is separately gated on `not _keep_activations_in_memory` (`torchlens/postprocess/__init__.py:199-206`, `:261-265`).
- Streaming is actually strict: `BundleStreamWriter` rejects `strict=False`, and `write_blob()` always calls `is_supported_for_save(..., strict=True)` (`torchlens/_io/streaming.py:64-81`, `:160-172`).
- Fork L's validation guard covers both `ModelLog.validate_saved_activations()` and the `ModelLog.validate_forward_pass` alias, for eager and lazy portable loads (`torchlens/validation/core.py:45-65`, `:68-99`; `torchlens/data_classes/model_log.py:704-705`).
- Fork K's no-shared-handles claim looks accurate: lazy refs call `safetensors.torch.load_file()` per materialization and do not retain file handles (`torchlens/_io/lazy.py:67-108`).
