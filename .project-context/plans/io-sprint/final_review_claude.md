# Final Review (Claude): `main..codex/io-sprint`

**Verdict:** REQUEST_CHANGES (aligned with Codex)

## Tests & Gates
- `ruff check .`: clean
- `mypy torchlens/`: clean (59 source files)
- `pytest tests/ -m "not slow"`: 1054 passed, 21 skipped, 0 failed, ~18 min
- `pytest tests/ -m smoke`: 28 passed

End-to-end smoke: `log_forward_pass -> torchlens.save -> torchlens.load` round-trips on a toy Conv2d model, activations bit-exact, to_pandas returns a 61-col frame.

## Confirmed independently

- **CRITICAL path traversal (Codex #1)**: reproduced. Tampered `manifest.json` with `relative_path = "../outside.safetensors"` + matching sha256 is happily accepted by `tl.load()`. Read outside the bundle root succeeds. This is a real security bug.
- **HIGH two-pass streaming (Codex #2)**: reproduced. `log_forward_pass(m, x, layers_to_save=['conv2d_1_1', 'conv2d_2_3'], save_activations_to=bundle, keep_activations_in_memory=False)` produces an empty `blobs/` directory, empty manifest tensors list, `keep_activations_in_memory=False` is silently ignored, and `activation_ref` is never set. Users get a corrupt bundle on disk and a ModelLog that claims it streamed when it didn't.

## My additional findings

1. **MEDIUM** — `torchlens/_io/bundle.py:266` emits `UserWarning: Bundle contains unreferenced extra files in blobs/` when a legitimate blob file is present but its manifest entry was edited away. The warning fires on the path-traversal exploit above, *before* the load succeeds. The warning should be escalated to `TorchLensIOError` when the unreferenced file's name collides with a persisted-but-redirected entry, or at least tightened from `UserWarning` (ignorable) to `warnings.warn(..., UserWarning)` with `stacklevel` set so users see it. Current behavior lets a tampered bundle load with only a print-to-stderr warning.

2. **MEDIUM** — `README.md:205` "Saving and Loading" section never explicitly tells the user that portable bundles use `pickle.load` internally, nor that `tl.load()` must only be used on trusted input. Good practice and the adversarial review both require this disclosure.

3. **LOW** — Plan Fork F specifies 13 version-policy rows; `tests/test_io_bundle.py` exercises all of them via monkeypatching. But the test file in a couple of cases asserts `TorchLensIOError` without checking the message content names the mismatched version string. For user-facing quality, these messages are exactly where actionability matters. Not a blocker.

4. **LOW** — `torchlens/_io/rehydrate.py` has no top-of-file warning that the S6 `rehydrate_nested` function silently no-ops on `lazy=True, materialize_nested=True` (the default) because there are no remaining nested BlobRefs. This is correct behavior but will confuse a user who calls it expecting "do something." A 1-line docstring clarifying "only does work when `materialize_nested=False` was set at load time" would prevent that.

5. **LOW** — `LazyActivationRef.materialize()` at `torchlens/_io/lazy.py:67` opens the file three times: once via `pathlib.exists()`, once via `sha256_of_file()`, once via `safetensors.torch.load_file()`. Not a correctness issue, but for huge tensors (multi-GB) this means 3x IOPS. Consider reading once into memory then verifying checksum + tensor.

## Confirmed contracts (matches plan)

- Fork A PORTABLE_STATE_SPEC: verified lint test is in S1 (`tests/test_io_scrub_policy.py`) and passes on every live instance attr across 7 classes.
- Fork B lazy=False default: confirmed in public API signature.
- Fork C six pandas surfaces + forward_args_summary/forward_kwargs_summary on ModulePassLog: confirmed.
- Fork D step 19 always runs + step 20 conditional: confirmed in `postprocess/__init__.py:199-206, 263-265`. **BUT** only exercised in the single-pass `layers_to_save="all"` path — fails in two-pass (Codex #2 above).
- Fork E strict=True default: confirmed in `torchlens.save` signature.
- Fork F 13-row version policy: confirmed implemented in `manifest.py::enforce_version_policy`.
- Fork G TorchLensIOError wrapping: confirmed, all bundle-layer errors funnel through.
- Fork H two-layer drift: confirmed at `bundle.py:631-680`.
- Fork J symlink rejection: confirmed in `bundle.py` (7 call sites) and `streaming.py:85`.
- Fork K no-shared-handles: confirmed in `lazy.py::materialize` (open-read-close per call).
- Fork L portable-bundle replay guard: confirmed both `validate_forward_pass` and `validate_saved_activations` paths at `validation/core.py:59, 98`.
- Fork M nested BlobRef resave rejection: confirmed — though the check is slightly over-eager (see Codex #4).

## Summary

The sprint's in-scope code paths work. 1054 tests pass. API ergonomics are clean: `tl.save(log, path)` + `tl.load(path)` + `log.to_pandas()` read as expected.

The two issues that block release:
1. **Path traversal CVE in bundle load** — CRITICAL security bug, any user loading untrusted bundles is vulnerable.
2. **Two-pass streaming is broken** — HIGH functional bug, selective streaming produces empty bundles and silent misbehavior. This path isn't tested at all (Codex LOW #5).

Both are narrow enough to fix in a single follow-up spec (~100-150 LoC). I'd dispatch **IO-S9: security + two-pass streaming** as the final task before merging this branch to main.
