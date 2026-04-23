# TorchLens I/O Sprint Plan (Round 2)

**Branch**: `codex/io-sprint`
**Goal**: comprehensive, frictionless I/O for ModelLog and its children — pickle, pandas/CSV/Parquet/JSON, activation-to-disk (including on-the-fly streaming during forward pass), symmetric loading.
**Out of scope**: visualization (graphviz, ELK, dagua).
**Inputs**: `.project-context/research/io-audit-claude.md`, `.project-context/research/io-audit-codex.md`, `.project-context/plans/io-sprint/review_round_1.md`.
**This revision**: addresses all 23 findings of `review_round_1.md`. Change log at bottom.

---

## 1. Design Forks — Resolved

### Fork A. Archival format
**Decision**: **directory bundle only** (no tar this sprint) containing a scrubbed-state pickle, safetensors-only tensor blobs, and a rich manifest. `safetensors` is a **hard dependency** of the bundle path.

Bundle layout:
```
<bundle_dir>/
  manifest.json            # see schema below — authoritative index
  metadata.pkl             # ModelLog with tensor fields stubbed to blob ids; scrubbed of non-portable objects
  blobs/
    0000000001.safetensors
    0000000002.safetensors
    ...                    # one file per saved tensor, filename is stable opaque blob id
```

Filenames are zero-padded monotonically-assigned blob ids (`0000000001.safetensors`). Human-readable layer labels (which contain `:` and other filesystem-unsafe characters) live only inside `manifest.json`, never in paths. Addresses review finding #5.

**Pickle scrubbing contract** (addresses finding #1):
Before `metadata.pkl` is written, a scrub pass walks the ModelLog state and:
- Replaces `LayerPassLog.activation` / `.gradient` / `.children_tensor_versions` tensors with `BlobRef(blob_id)` stubs.
- Replaces any tensors found inside `captured_args` / `captured_kwargs` / `forward_args` / `forward_kwargs` / `func_rng_states` with `BlobRef(blob_id)` stubs (unless `include_captured_args=False`, in which case they are dropped).
- Drops `func_applied` (live torch callable) — recorded by `func_name` string already.
- Drops `activation_postfunc` (user callable, may be a lambda) — recorded as `activation_postfunc_repr` string if present.
- Drops `_param_ref` on ParamLog (pins an `nn.Parameter`, not portable).
- Clears `FuncCallLocation._frame_func_obj` (already done by disabled-source path; make unconditional on save).
- Clears any `extra_attributes` on ModelLog flagged non-serializable (explicit allow-list).

The scrubbed state is versioned (`io_format_version=1`); `__setstate__` restores `BlobRef` stubs as `LazyActivationRef` and reconstructs dropped fields with sentinel values (`func_applied=None`, `activation_postfunc=None`, `_param_ref=None`). Existing `__getstate__` logic (strip weakrefs) continues to run on the pickle layer.

**Plain-pickle backward compat** (addresses finding #14):
`pickle.dump(model_log, f)` and `pickle.load(f)` continue to work. In this path, the existing `__getstate__` hooks run (weakref strip only) — no scrubbing — so the limitations of the current pickle surface are unchanged (callables and `_param_ref` remain). Users who want the portable, scrubbed representation use `torchlens.save(path)`. Plain pickle is kept supported, not deprecated, for at least this release cycle. Test coverage: `tests/test_io_plain_pickle.py`.

### Fork B. Default load granularity & API shape
**Decision**: **lazy default when bundle has activations; eager otherwise. `.activation` stays `tensor | None`. Lazy refs live on a separate field `.activation_ref`.**

Addresses findings #7 and #8.

Semantics after `torchlens.load(path, lazy=True)`:
- For each LayerPassLog that had a saved activation:
  - `.activation` is **None**
  - `.activation_ref` is a `LazyActivationRef` carrying shape/dtype/device_at_save/blob_id
  - `.tensor_shape`, `.tensor_dtype`, `.tensor_memory`, `.has_saved_activations` are all populated (metadata survives without materialization)
- `LayerPassLog.materialize_activation(*, map_location="cpu") -> torch.Tensor` reads from disk, stores into `.activation`, and returns it. `.activation_ref` remains for idempotent re-materialization.
- `torchlens.load(path, lazy=False)` eagerly materializes every activation during load — equivalent to calling `materialize_activation()` on every layer.

Existing TorchLens internal code that reads `.activation.shape` etc. continues to work on eager-loaded bundles (the common case) and fails loudly (AttributeError on None) on lazy-loaded bundles with clear error messaging. We **do not** auto-materialize on internal accesses — finding #8 is addressed by explicit eager opt-out plus a documented lazy-compatibility list.

**Lazy-compatibility audit** (part of S6): the following internal methods are verified lazy-safe or are updated to call `materialize_activation()` as needed:
- `validate_forward_pass` (reads `.activation`) — adds explicit check; raises a clear error if activations are lazy and `materialize=False` was passed.
- `validate_saved_activations` — same.
- String reprs and `to_pandas()` — already read only metadata fields, not `.activation`. No change.
- `cleanup()` — becomes lazy-aware: closes any open safetensors file handles, then nulls out refs.

### Fork C. Pandas-export scope
**Decision**: DataFrames at **six log levels**: layer-pass (ModelLog), aggregate layer, module-pass (new), module, param, buffer.

Adds `ModulePassLog.to_pandas()` to resolve finding #19 — includes `forward_args` / `forward_kwargs` shape metadata (tensors replaced by shape/dtype strings for pandas rows).

### Fork D. Streaming-save architecture
**Decision**: during forward pass, write the sidecar immediately at `save_tensor_data()` completion; **do not** evict the in-memory tensor until a new **post-pass eviction step** runs inside postprocess. Selective streaming hooks the exhaustive pass only, not fast-path refresh sites.

Addresses findings #2 and #12.

Concrete flow:
1. `log_forward_pass(..., save_activations_to=path)` creates ModelLog with `_activation_writer = BundleStreamWriter(path + ".tmp")`.
2. Inside `LayerPassLog.save_tensor_data()` (layer_pass_log.py:455-501), after the existing clone / detach / output_device / `activation_postfunc` pipeline, the writer receives `(blob_id, self.activation)` and writes `blob_id.safetensors` to `<path>.tmp/blobs/`. `self.activation` is **unchanged** at this point.
3. Writer is invoked **only from the exhaustive-pass call site** (`capture/output_tensors.py:866-900`). The fast-path overwrite sites (`output_tensors.py:608-644`, `source_tensors.py:316-322`) are NOT instrumented for streaming (those support `save_new_activations` which has known issues; streaming is a first-pass feature).
4. Postprocess runs as usual with real tensors in memory.
5. New postprocess step **`_evict_streamed_activations`** (added to the 18-step pipeline) runs last if `keep_activations_in_memory=False`: for each layer with a streamed sidecar, replaces `.activation` with None, sets `.activation_ref` to a `LazyActivationRef` pointing at the sidecar.
6. Writer writes `manifest.json` and `metadata.pkl` atomically at the end, then renames `<path>.tmp` → `<path>`.

**Atomicity scope** (addresses finding #11):
- Temp-dir suffix is random (`<path>.tmp.{uuid4}`) so two concurrent writers never collide.
- `<path>` must not exist at start (fail fast with `FileExistsError`). User can pass `overwrite=True` to allow replacement; replacement does rename-old-then-rename-new, accepting a small window of inconsistency.
- Rename + fsync documented as local-same-filesystem best-effort. Cross-filesystem and NFS atomicity explicitly NOT claimed; docstring warns.

**Unsupported-writer state & cleanup** (addresses finding #22):
- `torchlens.cleanup_tmp(path)` is a public utility added in S5 that searches `<path>.tmp.*` siblings and removes them (after confirmation prompt unless `force=True`).
- On exception during streaming: the `.tmp.<uuid>` directory is kept, a warning naming the path is emitted, and the writer emits `<path>.tmp.<uuid>/PARTIAL` sentinel file so `cleanup_tmp` can identify it safely.

**Non-tensor activation_postfunc guard** (addresses finding #6):
If `save_activations_to` is set AND `activation_postfunc` is passed AND postfunc output is not a `torch.Tensor`, raise `TorchLensIOError` at forward start with a clear message. This check is a one-liner on the first call (after postfunc applied to sentinel input-shape tensor) or runtime-first-layer — latter is simpler: the sink validates the type of the first tensor it sees; if not `torch.Tensor`, the whole write aborts and `.tmp` directory is kept.

**Gradient streaming** (addresses finding #3):
Gradients are **NOT** streamed during backward in this sprint. Rationale: backward is user-initiated, may not happen, may fail halfway, and lacks the single instrumentation point that forward has.

Post-hoc gradient bundling IS supported: `torchlens.save(model_log, path, include_gradients=True)` after `log_forward_pass` + user-run backward. If any LayerPassLog has `.gradient` populated, it is written to `blobs/NNNNNNNNNN.safetensors` with `kind="gradient"` in the manifest. `include_gradients=True` with no populated gradients is a no-op (not an error).

This preserves the user promise (grads can be archived) without taking on a gradient-streaming design this sprint.

### Fork E. Backward compat & version policy
**Decision**: version-tagged manifest + default-fill `__setstate__` + explicit mismatch policy table.

Addresses findings #14 and #15.

Version policy table:
| Scenario | Load behavior |
|----------|---------------|
| `io_format_version` == current | Load normally |
| `io_format_version` < current | Load with default-fill, emit `DeprecationWarning` naming missing fields |
| `io_format_version` > current | Hard-fail with `TorchLensIOError`: "bundle requires torchlens >= X; refusing to load" |
| `torchlens_version` < current, same format version | Load, emit info log |
| `torchlens_version` > current, same format version | Load with `UserWarning` |
| `torch_version` major mismatch (e.g. 1.x ↔ 2.x) | Hard-fail with `TorchLensIOError`: "bundle produced with torch X; incompatible major" |
| `torch_version` minor mismatch | Load with `UserWarning` |
| `python_version` major mismatch | Load attempt; if pickle opcode fails, raise with a descriptive error referencing the version |
| Safetensors backend absent | Hard-fail: "bundle requires safetensors; pip install safetensors" |
| Manifest missing or unparseable | Hard-fail |
| Blob file missing from manifest index | Hard-fail with the specific missing blob_id |
| Blob checksum mismatch | Hard-fail |
| Unknown extra files in bundle | Warning, continue |

---

## 2. Manifest Schema v1

```json
{
  "io_format_version": 1,
  "torchlens_version": "1.2.0",
  "torch_version": "2.5.0",
  "python_version": "3.11.5",
  "platform": "linux-x86_64",
  "created_at": "2026-04-23T13:05:00Z",
  "bundle_format": "directory",
  "n_layers": 152,
  "n_activation_blobs": 152,
  "n_gradient_blobs": 0,
  "tensors": [
    {
      "blob_id": "0000000001",
      "kind": "activation",
      "label": "conv2d_1_1:2",
      "relative_path": "blobs/0000000001.safetensors",
      "backend": "safetensors",
      "shape": [1, 64, 112, 112],
      "dtype": "float32",
      "device_at_save": "cuda:0",
      "layout": "strided",
      "bytes": 3211264,
      "sha256": "3f5c..."
    }
  ],
  "unsupported_tensors": [
    {"label": "sparse_layer:1", "reason": "sparse_coo layout not supported this release"}
  ]
}
```

Addresses finding #10.

---

## 3. Supported Tensor Matrix

Addresses finding #13.

| Tensor kind | Status | Behavior on save |
|-------------|--------|------------------|
| Dense CPU / CUDA (`float16`, `bfloat16`, `float32`, `float64`, `int*`, `uint8`, `bool`, `complex64`, `complex128`) | Supported | Written via safetensors |
| Non-contiguous | Supported | `.contiguous()` called before write; documented |
| Sparse COO / CSR | Not supported this sprint | Skipped with warning, entry added to `unsupported_tensors`, `.activation_ref` stays None |
| Quantized | Not supported | Same as above |
| Meta tensors (`device="meta"`) | Skipped | No data to persist; metadata preserved in manifest only |
| Nested tensors | Not supported | Skipped with warning |
| Tensor subclasses (unknown) | Not supported | Skipped with warning |
| DTensor / FSDP local shard | Not supported | Skipped with warning; user must gather first |
| Complex dtypes (complex32) | Not supported (safetensors limitation) | Skipped with warning |

`torchlens.save()` with unsupported tensors never errors; it writes what it can and records what it skipped in `unsupported_tensors`. Loading such a bundle produces LayerPassLogs with `activation=None` and a clear "skipped at save time" note on load. This is a deliberate choice for defensive UX — exceptions would make one weird layer break a whole 1000-layer archive.

The strict alternative (`strict=True` kwarg) raises on first unsupported tensor. Default is lenient.

---

## 4. Public API Surface

### Top-level `torchlens`
```python
torchlens.save(
    model_log: ModelLog,
    path: str | Path,
    *,
    include_activations: bool = True,
    include_gradients: bool = True,    # no-op if no gradients populated
    include_captured_args: bool = True,
    overwrite: bool = False,
    strict: bool = False,
) -> None

torchlens.load(
    path: str | Path,
    *,
    lazy: bool = True,
    map_location: str | torch.device = "cpu",
) -> ModelLog

torchlens.cleanup_tmp(
    path: str | Path,
    *,
    force: bool = False,
) -> list[Path]    # returns list of cleaned-up .tmp.* dirs
```

**Removed from v1 plan**: `format="tar"`, `compression="zstd"` (findings #4 and #21).

### On `ModelLog`
```python
model_log.save(path, **kwargs)          # sugar for torchlens.save(self, path, **kwargs)
ModelLog.load(path, **kwargs)           # classmethod

model_log.to_pandas() -> pd.DataFrame
model_log.to_csv(path, **kwargs)
model_log.to_parquet(path, **kwargs)
model_log.to_json(path, *, orient="records")
```

### On accessors (all six gain .to_csv / .to_parquet / .to_json)
```python
model_log.layers.to_pandas() / .to_csv / .to_parquet / .to_json
model_log.modules.to_pandas() / ...
model_log.params.to_pandas() / ...     # NEW
model_log.buffers.to_pandas() / ...    # NEW

module_log.to_pandas() / ...           # exists (one row per layer inside a single module)
module_pass_log.to_pandas() / ...      # NEW — one row per module pass
```

### Streaming at capture time (`log_forward_pass`)
```python
log_forward_pass(
    ...,
    save_activations_to: str | Path | None = None,
    keep_activations_in_memory: bool = True,
    activation_sink: Callable[[str, torch.Tensor], None] | None = None,
)
```
- `save_activations_to=path`: writes a full bundle as capture progresses; returns a lazy-enabled ModelLog.
- `activation_sink`: low-level callable for custom destinations (e.g. S3, shared memory); mutually-exclusive with `save_activations_to`.
- `keep_activations_in_memory=False`: post-pass eviction step replaces `.activation` with None and sets `.activation_ref`.

### Re-saving a lazy-loaded bundle (addresses finding #23)
```python
# lazy-loaded bundle, user modified nothing:
model_log = torchlens.load("a/", lazy=True)
model_log.save("b/")           # copies blobs from a/ to b/ without materializing (fast path)

# lazy-loaded bundle, user materialized some activations:
t = model_log.layer_list[3].materialize_activation()
# hypothetically: t = user modifies activation in place — we don't support that cleanly
model_log.save("b/")           # re-writes only materialized layers; un-materialized ones are copied from source bundle

# lazy-loaded bundle, source bundle deleted mid-session:
# materialize anything not already in memory first, or save() will raise clearly
```

Implementation: `save()` iterates layers. For each that has `.activation` populated: normal write. For each that is lazy-only and `.activation_ref.source_bundle_path` exists and is intact: `shutil.copy` the sidecar file to the new bundle. For each that is lazy-only and source is gone/corrupt: raise with instruction to materialize first. Edge-case covered by integration test.

---

## 5. Spec Decomposition

Quality gates mandatory for every spec: `ruff check . --fix && mypy torchlens/ && pytest tests/ -m smoke -x --tb=short`. LoC estimates revised upward per finding #20.

### Group 1 — Foundations (parallel-safe on different files)

**IO-S1. Pickle hardening: scrubbing + version tagging + accessor reconstruction**
- Add `torchlens/_io/__init__.py` with `IO_FORMAT_VERSION = 1`, `TorchLensIOError`, `BlobRef` NamedTuple.
- Add `torchlens/_io/scrub.py`: `scrub_for_save(model_log, include_captured_args=True) -> (scrubbed_dict, list[BlobSpec])`. Walks the ModelLog, produces scrubbed pickle state and a catalog of tensors needing blob persistence. This is the single source of truth for what-gets-scrubbed.
- Expand ModelLog `__setstate__` to: read version, default-fill, rebuild `_module_logs` and `_buffer_accessor` (share helper extracted from `postprocess/finalization.py:671-681`), restore `BlobRef` → `LazyActivationRef`.
- Add `__setstate__` default-fill to LayerPassLog, LayerLog, ParamLog, BufferLog, ModuleLog, ModulePassLog.
- Files: `torchlens/_io/__init__.py` (NEW), `torchlens/_io/scrub.py` (NEW), `torchlens/_io/accessor_rebuild.py` (NEW, shared by pickle + bundle), `torchlens/data_classes/{model_log,layer_pass_log,layer_log,param_log,buffer_log,module_log}.py`, `torchlens/postprocess/finalization.py` (extract helper only).
- Tests: `tests/test_io_pickle.py` (NEW) — round-trip scrubbed ModelLog; accessors rebuilt correctly; default-fill from forged older pickle; version hard-fail on newer; plain-pickle unchanged.
- Size: **~360 LoC** (up from 240).

**IO-S2. ParamAccessor.to_pandas(), BufferAccessor.to_pandas(), ModulePassLog.to_pandas()**
- Mirror `LayerAccessor.to_pandas` (`layer_log.py:665-687`).
- FIELD_ORDER constants for param, buffer, module-pass.
- Files: `torchlens/data_classes/{param_log,buffer_log,module_log}.py`, `torchlens/constants.py`.
- Tests: `tests/test_io_pandas.py` (NEW) — schema shape; empty accessor; NaN; `forward_args` shape-stringification on module-pass rows.
- Size: **~200 LoC** (up from 160, adds module-pass).

### Group 2 — Tabular export + bundle (serial: S3 after S2, S4 after S1)

**IO-S3. to_csv / to_parquet / to_json export wrappers**
- Wrappers on every DataFrame producer: ModelLog, LayerAccessor, ModuleAccessor, ModuleLog, ModulePassLog, ParamAccessor, BufferAccessor.
- `pyarrow` optional via `pip install torchlens[io]`; clear ImportError with install hint if `to_parquet` called without it.
- Files: `torchlens/data_classes/{interface,layer_log,module_log,param_log,buffer_log}.py`, `pyproject.toml` (extras).
- Tests: `tests/test_io_export.py` (NEW) — round-trip CSV / parquet / JSON; ImportError on parquet without pyarrow; schema stability.
- Size: **~220 LoC**.

**IO-S4. Bundle save (eager post-hoc), manifest v1, supported-tensor matrix, integrity checks on load**
- `torchlens/_io/bundle.py`: `save(model_log, path, ...)` and `load(path, ...)`.
- `torchlens/_io/manifest.py`: schema class, writer, reader, validator.
- `torchlens/_io/tensor_policy.py`: implements the supported-tensor matrix, returns `(ok|skip_reason)` for each tensor.
- Safetensors required. Missing → hard-fail with install hint.
- Manifest `tensors` table populated; sha256 checksum per blob; shape/dtype/device_at_save recorded.
- Load-time integrity: verify every manifest blob_id exists on disk; checksum on open (lazy — done at materialization for lazy loads, at load for eager); version policy table enforced.
- Files: `torchlens/_io/{bundle,manifest,tensor_policy}.py` (NEW); `torchlens/data_classes/model_log.py` (sugar methods); `torchlens/__init__.py`; `torchlens/user_funcs.py`.
- Tests: `tests/test_io_bundle.py` (NEW) — eager save/load round-trip; unsupported tensor lenient + strict; version mismatch table; corrupted manifest; missing blob file; checksum tampering; disk-full at various points (monkeypatched); overwrite semantics; re-save of lazy bundle (fast-copy path).
- Size: **~420 LoC** (up from 320).

### Group 3 — Streaming + lazy (serial: S5 after S4, S6 after S5)

**IO-S5. Streaming-save during forward pass + eviction postprocess step + cleanup_tmp utility**
- `torchlens/_io/streaming.py`: `BundleStreamWriter` opens `<path>.tmp.<uuid>/blobs/`, receives `(blob_id, tensor)` calls, finalizes manifest + metadata.pkl + rename at end.
- Instrument `LayerPassLog.save_tensor_data()` (layer_pass_log.py:455-501) with writer callout; only activated by the **exhaustive-pass call site** (`output_tensors.py:866-900`). Fast-path overwrite sites remain untouched.
- New postprocess step `_evict_streamed_activations` appended to the 18-step pipeline (gate: `keep_activations_in_memory=False`).
- Non-tensor `activation_postfunc` check: on first tensor passed to sink, `assert isinstance(t, torch.Tensor)`; else abort with clear error and retain `.tmp.<uuid>`.
- `torchlens.cleanup_tmp(path, force=False)` finds `<path>.tmp.*` siblings with `PARTIAL` sentinel, removes after confirmation.
- Files: `torchlens/_io/streaming.py` (NEW), `torchlens/_io/bundle.py` (finalize handoff), `torchlens/data_classes/layer_pass_log.py`, `torchlens/data_classes/model_log.py` (holds `_activation_writer`, updated cleanup logic), `torchlens/user_funcs.py` (new kwargs), `torchlens/postprocess/finalization.py` (new eviction step), `torchlens/postprocess/__init__.py` (step registration).
- Tests: `tests/test_io_streaming.py` (NEW) — sidecars written during pass (not after); mid-pass exception leaves `.tmp.<uuid>` with `PARTIAL`; `activation_sink` callable path; non-tensor postfunc rejected; `keep_activations_in_memory=False` evicts post-pass, not mid-pass (verify postprocess sees tensors); cleanup_tmp finds and removes partial dirs.
- Size: **~380 LoC** (up from 270).

**IO-S6. Lazy activation references + lazy-compatibility audit**
- Flesh out `torchlens/_io/lazy.py`: `LazyActivationRef(blob_id, shape, dtype, device_at_save, source_bundle_path)` with `materialize(map_location="cpu")`.
- `LayerPassLog.materialize_activation(*, map_location="cpu") -> torch.Tensor`: materializes, stores into `.activation`, keeps `.activation_ref`, returns.
- Audit list: verify `validate_forward_pass`, `validate_saved_activations`, cleanup, string reprs, and `to_pandas()` against lazy logs. Update where needed with clear error messages (no silent auto-materialize).
- Cleanup: closes any open safetensors mmap handles before nulling refs.
- Re-save of lazy bundle: `save()` copies sidecars when source still intact; materializes-then-writes otherwise; raises clearly if source gone and not materialized.
- Files: `torchlens/_io/lazy.py` (EXPAND from S5 stub), `torchlens/data_classes/layer_pass_log.py`, `torchlens/data_classes/cleanup.py`, `torchlens/validation/invariants.py`, `torchlens/_io/bundle.py` (lazy-resave path).
- Tests: `tests/test_io_lazy.py` (NEW) — shape/dtype accessible without materialize; materialize correctness (bit-for-bit equality with eager); map_location works; open-handle cleanup; resave fast-copy; resave with source-bundle deleted raises clearly; validation APIs error clearly on lazy-only logs.
- Size: **~340 LoC** (up from 240).

### Group 4 — Integration tests + docs (parallel-safe)

**IO-S7. Integration + corruption test suite**
- End-to-end: capture → stream-save → lazy-load → to_parquet round-trip.
- Corruption: truncated safetensors, missing blob, corrupt pickle, tampered manifest, stale blob counts.
- Forged older-format pickle loads with warning.
- DataParallel / DDP sanity (skipped if no CUDA): confirms capture path remains single-process (torchlens is not DP-safe; explicit warning recorded).
- Files: `tests/test_io_integration.py` (NEW), minor updates to `test_save_new_activations.py`.
- Size: **~360 LoC** (up from 280).

**IO-S8. Docs + bundle utilities**
- Docstrings on all new public APIs with copy-pasteable examples.
- `torchlens/_io/pack.py`: `torchlens.pack_bundle(bundle_dir, out_tar)` and `unpack_bundle(out_tar, into_dir)` — **eager-only tar archival utility** for users who want a single file. Explicitly not random-access; documented.
- README section (short) under "Save and Load".
- `.project-context/knowledge/io_architecture.md`: bundle layout, manifest schema, tensor matrix, version policy.
- Files: `torchlens/_io/__init__.py`, `torchlens/_io/pack.py` (NEW), `torchlens/__init__.py`, `torchlens/data_classes/model_log.py`, `README.md`, `.project-context/knowledge/io_architecture.md` (NEW).
- Size: **~180 LoC** (up from 120, adds pack utility).

---

## 6. Spec Dependency Graph

```
S1 ─┬─> S4 ─┬─> S5 ──> S6 ──> S7
    │       │                  ▲
S2 ─┴─> S3 ─┘                  │
                        S8 ────┘ (docs depend on all)
```

Total LoC: ~2460 (up from 1830 in v1).

---

## 7. Risk Register

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Unsupported tensor layouts (sparse/meta/DTensor) | Skipped with warning by default; `strict=True` raises; manifest records all skipped. |
| R2 | `func_applied`/`activation_postfunc` pickle fragility | Scrubbed at save time; null sentinels on load; recorded by name/repr string. |
| R3 | Disk-full / crash mid-stream | `.tmp.<uuid>/` preserved with `PARTIAL` sentinel; `cleanup_tmp` utility provided. |
| R4 | Lazy refs leak into DataFrames | Exporters only read metadata fields. Assertion in tests that no `LazyActivationRef` repr appears in DataFrame cells. |
| R5 | Lazy cleanup vs live file handles | `cleanup()` closes safetensors handles before nulling. Tested. |
| R6 | Pickle-format drift in N+1 | `io_format_version` policy table; migration helpers required before bumping version. |
| R7 | CUDA bundle loaded on CPU host | `map_location="cpu"` default; safetensors loader accepts it. |
| R8 | Streaming writer blocks event loop | Synchronous writes; documented. Async out of scope. |
| R9 | pyarrow install weight | Optional via extras; clear error if used without. |
| R10 | Large bundle on tiny model feels heavy | `pack_bundle` utility for single-file archival; docs explain directory-vs-tar. |
| R11 | Thousands of tiny blobs on huge model | Documented; chunked-blob design deferred to a future sprint. |
| R12 | NFS / Windows rename semantics | Docstring scopes atomicity claim to local same-fs. Non-local FS guidance in docs. |
| R13 | User deletes source bundle after lazy load | `materialize_activation` raises clear error; `save()` raises clearly if asked to resave without materialization. |
| R14 | DataParallel / DDP multiple capture paths | Torchlens is already single-process-only (existing `warn_parallel`). Documented; streaming inherits that constraint. |

---

## 8. Execution Plan

| Round | Specs | Gate |
|-------|-------|------|
| 1 | S1 ‖ S2 | pickle round-trip green; param/buffer DataFrames green |
| 2 | S3 (after S2) ‖ S4 (after S1) | bundle save/load round-trip green; all export formats green |
| 3 | S5 (after S4) | streaming green; eviction doesn't break postprocess |
| 4 | S6 (after S5) | lazy load + materialize green; validation APIs updated |
| 5 | S7 ‖ S8 | integration regression + docs |
| 6 | Phase 5 dual review (me + `/codex:review`) |

---

## 9. Stop-and-Ask Triggers

- End of this plan revision — **NOW** (present to user for approval)
- Any public API signature change beyond additions
- Bundle format bump (io_format_version increase) in the future
- End of Phase 5 (which review findings to fix)

---

## 10. Success Criteria (binary)

- [ ] `torchlens.save(model_log, path)` + `torchlens.load(path)` round-trip including all supported tensor layouts
- [ ] Six log-level DataFrames + `.to_csv/.to_parquet/.to_json` on every one
- [ ] `log_forward_pass(..., save_activations_to=path)` streams during pass; bundle openable immediately after forward
- [ ] `keep_activations_in_memory=False` path works end-to-end with postprocess intact
- [ ] Lazy load preserves metadata access without disk read; `materialize_activation()` bit-exact
- [ ] `pytest tests/ -m "not slow"` green; ruff + mypy + smoke green
- [ ] Old pickles (v1.0.2 plain `pickle.dump(model_log)`) load with deprecation warning
- [ ] Corruption tests green (truncated blob, missing blob, corrupted pickle, tampered manifest)
- [ ] `cleanup_tmp` utility removes `.tmp.<uuid>` dirs safely

---

## 11. Change Log vs Round 1

| Finding | Severity | Resolution |
|---------|----------|-----------|
| 1 | CRITICAL | Pickle scrubbing contract (Fork A + S1). |
| 2 | CRITICAL | Eviction moves to postprocess step (Fork D); save_tensor_data no longer mutates `.activation`. |
| 3 | CRITICAL | Gradient streaming cut; post-hoc gradient bundling only; manifest records `kind="gradient"`. |
| 4 | CRITICAL | tar/tar.gz removed from core API; optional `pack_bundle` utility in S8 for archival only, eager-only. |
| 5 | CRITICAL | Opaque zero-padded blob ids in paths; human labels live only in manifest. |
| 6 | CRITICAL | Non-tensor `activation_postfunc` guard added to S5; aborts streaming with clear error. |
| 7 | HIGH | `.activation` stays tensor-or-None; `.activation_ref` holds LazyActivationRef; explicit `materialize_activation()`. |
| 8 | HIGH | No auto-materialize; validation APIs updated to raise clearly; eager opt-out with `lazy=False`. |
| 9 | HIGH | Safetensors required; no fallback; `torch.save` removed from bundle path. |
| 10 | HIGH | Manifest v1 schema with per-tensor table, checksums, backend, device_at_save, layout, bytes. |
| 11 | HIGH | Unique-uuid temp dirs; atomicity scoped to local-fs best-effort; rename semantics documented. |
| 12 | HIGH | Streaming hooks exhaustive-pass site only; fast-path sites untouched. |
| 13 | HIGH | Explicit supported-tensor matrix; skipped tensors recorded in manifest; strict mode available. |
| 14 | HIGH | Plain `pickle.dump(model_log)` stays supported with explicit contract and test. |
| 15 | HIGH | Version policy table (Fork E). |
| 16 | HIGH | Directory default kept; `pack_bundle` utility for single-file; thousands-of-blobs issue documented. |
| 17 | HIGH | Corruption test battery (S7); exact exception types in specs. |
| 18 | HIGH | `frames/` removed from bundle layout (not needed; source context handled by scrubbed FuncCallLocation). |
| 19 | MEDIUM | ModulePassLog added to pandas scope. |
| 20 | MEDIUM | LoC estimates revised up ~35% across the board; dependencies explicit in §6. |
| 21 | MEDIUM | `compression="zstd"` removed. |
| 22 | MEDIUM | `cleanup_tmp` utility owned by S5. |
| 23 | MEDIUM | Re-save of lazy bundle: fast-copy sidecars when source intact, raise clearly otherwise. |
