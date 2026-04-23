# TorchLens I/O Sprint Plan (Round 4)

**Branch**: `codex/io-sprint`
**Goal**: comprehensive, frictionless I/O for ModelLog and its children — pickle, pandas/CSV/Parquet/JSON, activation-to-disk (including streaming during forward pass), symmetric loading.
**Out of scope**: visualization (graphviz, ELK, dagua), tar archival utilities, gradient streaming during backward.
**Inputs**: `.project-context/research/io-audit-{claude,codex}.md`, `.project-context/plans/io-sprint/review_round_{1,2,3}.md`.
**This revision addresses all 9 Round-3 findings. Change log at bottom.**

---

## 1. Design Forks — Resolved

### Fork A. Archival format
**Decision**: **directory bundle only**, containing a scrubbed pickle, safetensors-only tensor blobs, and a rich manifest. `safetensors` is a **hard dependency** of the bundle path. No tar this sprint. No `pack_bundle` utility — out of scope.

Bundle layout:
```
<bundle_dir>/
  manifest.json               # authoritative index; see §2
  metadata.pkl                # scrubbed ModelLog state
  blobs/
    0000000001.safetensors
    0000000002.safetensors
    ...                       # one file per persisted tensor; filename = zero-padded opaque blob id
```

Filenames are zero-padded monotonically-assigned blob ids. Human-readable layer labels live only in `manifest.json`.

**Pickle-scrub contract** — addresses Round-2 findings #1/#2 + Round-3 finding #2.

**Authoritative source of truth**: each class in torchlens gains a `PORTABLE_STATE_SPEC: dict[str, FieldPolicy]` class attribute that explicitly lists every attribute present on instances of that class (not just fields in FIELD_ORDER) and the policy for each. The scrubber consumes this spec. A completeness lint (in S1 tests) walks every live instance attribute on a canonical ModelLog and asserts each name appears in its class's spec.

Classes requiring a `PORTABLE_STATE_SPEC`:
- `ModelLog`
- `LayerPassLog`
- `LayerLog`
- `ModuleLog`, `ModulePassLog`
- `ParamLog`
- `BufferLog`
- `FuncCallLocation`
- Any accessor class that ends up in the object graph

Policy vocabulary:
- `keep`: pickle as-is (primitives, containers of primitives, stable torch objects like `torch.dtype`).
- `blob(kind)`: tensor-valued; replace with `BlobRef(blob_id, kind=...)` and record in manifest.
- `blob_recursive(kind)`: walk container recursively; tensors blobified; non-tensor leaves passed through.
- `drop`: set to `None` (live callables, `_param_ref`, transient build state).
- `stringify`: replace with `"<scrubbed:TypeName>"` (fall-through for user-supplied non-portable objects).
- `weakref_strip`: existing `__getstate__` weakref-nulling behavior, applied automatically.

Policy by class (abbreviated — full matrix in `torchlens/_io/scrub.py`):

| Class | Field | Policy |
|-------|-------|--------|
| LayerPassLog | `activation` | `blob("activation")` (scalar 0-d recorded inline in manifest) |
| LayerPassLog | `gradient` | `blob("gradient")` |
| LayerPassLog | `captured_args`, `captured_kwargs` | `blob_recursive("captured_arg")` if `include_captured_args=True`, else `drop` |
| LayerPassLog | `children_tensor_versions` | `blob_recursive("child_version")` if `include_captured_args=True`, else `drop` |
| LayerPassLog | `func_rng_states` | `blob_recursive("rng_state")` if `include_rng_states=True`, else `drop` |
| LayerPassLog | `func_applied` | `drop` |
| LayerPassLog | `func_config` | `blob_recursive("func_config")` with `stringify` fallback for unknown non-tensor objects |
| LayerPassLog | `parent_layer_log` | `drop` (rebuilt on load from manifest + layer_dict_all_keys graph) |
| LayerPassLog | all other FIELD_ORDER columns | `keep` |
| LayerPassLog | `_source_model_log_ref` (weakref) | `weakref_strip` |
| LayerLog | `func_applied`, `activation_postfunc` | `drop` (copied from first pass; recoverable from LayerPassLog records or repr string) |
| LayerLog | `_source_model_log_ref` | `weakref_strip` |
| LayerLog | all other columns | `keep` |
| ModuleLog | `extra_attributes` | `blob_recursive("module_meta")` with `stringify` fallback |
| ModuleLog | `_source_model_log_ref` | `weakref_strip` |
| ModulePassLog | `forward_args`, `forward_kwargs` | `blob_recursive("module_arg")` if `include_captured_args=True`, else `drop`. Summary fields (`forward_args_summary`, `forward_kwargs_summary`) always `keep`. |
| ParamLog | `_param_ref` | `drop` (pins `nn.Parameter`) |
| ParamLog | all other columns | `keep` |
| BufferLog | Inherits LayerPassLog policy (is a subclass) | — |
| FuncCallLocation | `_frame_func_obj` | `drop` |
| FuncCallLocation | `_frame`, `_linecache_entry` if present | `drop` |
| FuncCallLocation | `code_context`, `func_signature`, `func_docstring`, `call_line`, etc. | `keep` |
| ModelLog | `activation_postfunc` | `drop`; record `activation_postfunc_repr: str` if present |
| ModelLog | `_module_logs`, `_buffer_accessor`, `_module_build_data` | `drop` (rebuilt on load — existing behavior) |
| ModelLog | `_optimizer` if present | `drop` |
| ModelLog | `_saved_gradients_set`, `_mod_*`, `_module_metadata`, `_module_forward_args` | `drop` (transient build state) |
| ModelLog | all other FIELD_ORDER columns and configuration fields | `keep` |
| Any | Attribute not in spec | Lint failure during S1 tests (`tests/test_io_scrub_policy.py` in S1). |

Adding a new attribute to any of these classes in a future PR requires updating the class's `PORTABLE_STATE_SPEC` — enforced by the S1 lint.

**Completeness lint placement**: lives in S1 (`tests/test_io_scrub_policy.py`), not S7. This means the guarantee is active from the moment scrubber lands. Fixes Round-3 finding #8.

**Rehydration contract on load** — addresses Round-2 finding #1 + Round-3 finding #1.

All blob materialization happens in a dedicated loader (`torchlens/_io/rehydrate.py::rehydrate_model_log(scrubbed_state, manifest, bundle_path, *, lazy, map_location, materialize_nested)`), NOT in `__setstate__`. `__setstate__` runs only default-fill + weakref restoration, never touches disk.

This split is critical: `__setstate__` has no way to receive `map_location` from the caller, and it also runs on plain `pickle.load()` of a scrubbed state dict where the bundle root may not be accessible at all. The loader wraps `__setstate__` and then applies the rehydration pass.

**Default changes** — addresses Round-3 finding #1 on RAM blow-up:
- `torchlens.save(..., include_captured_args=False, include_rng_states=False)` are now the **defaults**. Captured args and RNG states balloon bundle size and their tensors are rarely interesting for post-hoc analysis. Users opt in explicitly: `include_captured_args=True` if they want validation-ready bundles.
- New load-time flag: `torchlens.load(path, lazy=False, map_location="cpu", materialize_nested=True)`. When `materialize_nested=False` with `lazy=True`, nested tensor refs stay as `BlobRef`-like objects — power-user mode for cases where capture-arg bundles would not fit in RAM.

Rehydration behavior by mode:
- `lazy=False` (default): `.activation`/`.gradient` materialized; nested tensors materialized with `map_location`.
- `lazy=True, materialize_nested=True` (default for lazy): `.activation`/`.gradient` remain refs; nested tensors materialized eagerly. Common case: user wants metadata access without activation RAM cost but needs real tensors for validation paths.
- `lazy=True, materialize_nested=False` (expert): everything lazy. User explicitly accepts that `.captured_args[0]` may still be a `BlobRef` and calls `rehydrate_nested(model_log)` later when needed.

`map_location` is honored in all three modes. Safetensors accepts it directly.

**Plain-pickle backward compat**:
`pickle.dump(model_log, f)` / `pickle.load(f)` continue to work via the existing `__getstate__` hooks (weakref strip only — no scrub). Limitations of current plain pickle (callables fragility, `_param_ref` pinning) are unchanged. Portable representation is `torchlens.save()`.

**Compat contract resolution** — addresses Round-2 finding #14:
Plain `pickle.dump/load` is supported for this release; a `DeprecationWarning` fires on `pickle.load` of any ModelLog with `io_format_version` field missing (old pickles from before this sprint). No deprecation warning on `pickle.dump` of a fresh ModelLog (users still allowed to produce plain pickles; they just aren't the portable format). Tested in `tests/test_io_plain_pickle.py`.

### Fork B. Load granularity
**Decision**: `lazy=False` is **the default**. Lazy is opt-in via `torchlens.load(path, lazy=True)`.

Rationale (addresses Round-2 finding #5):
- Existing TorchLens internal methods assume `.activation` is a tensor when present.
- A full audit of every public and semi-public method is out of scope for this sprint.
- Making lazy opt-in means users who want it accept the "call `materialize_activation()` before doing tensor ops" contract explicitly.
- A future sprint can flip the default after a full compat audit.

`.activation` stays `tensor | None` throughout. `.activation_ref` is populated whenever a blob exists on disk (lazy or eager load). After `lazy=True` load, `.activation` is `None` and user must call `materialize_activation()` to get a tensor. After `lazy=False` load, both `.activation` and `.activation_ref` are populated.

### Fork C. Pandas-export scope
**Decision**: DataFrames at six log levels: layer-pass, aggregate layer, module-pass, module, param, buffer.

`ModulePassLog.forward_args_summary` and `ModulePassLog.forward_kwargs_summary` new fields — addresses Round-2 finding #9 + Round-3 finding #5.
- During postprocess step `_build_module_logs`, **before** the existing GC step that clears `forward_args`/`forward_kwargs`, two summary strings are computed using a recursive formatter:
  - Tensor → `"Tensor(shape=(1,3,224,224), dtype=float32)"`.
  - Primitive (bool/int/float/str/None) → `repr(value)`.
  - List/Tuple → `"[" + ", ".join(format(v) for v in seq) + "]"`.
  - Dict → `"{" + ", ".join(f"{k}: {format(v)}" for k,v in d.items()) + "}"`.
  - Other object → `f"<{type(value).__name__}>"`.
- Stored as `forward_args_summary: str` and `forward_kwargs_summary: str` on ModulePassLog.
- `ModulePassLog.to_pandas()` includes both columns. kwargs-heavy modules and dict/list-nested inputs are preserved.
- Existing GC behavior unchanged; `forward_args` / `forward_kwargs` still cleared after the summary pass.

### Fork D. Streaming-save architecture
**Decision**: during forward pass, write the sidecar at `save_tensor_data()` completion; activation remains in memory until the postprocess evict step runs. Selective streaming hooks the exhaustive pass only. **Bundle finalization is split from activation eviction** into two separate postprocess steps (addresses Round-3 finding #3).

Concrete pipeline positions — addresses Round-2 finding #3 and Round-3 finding #3.

Current 18-step postprocess pipeline (imperative code in `postprocess/__init__.py`): this sprint adds two steps at the end:
```
Step 19: _finalize_streamed_bundle       (NEW, always runs when writer present)
Step 20: _evict_streamed_activations     (NEW, conditional on keep_activations_in_memory=False)
```

Step 19 gate: `ModelLog._activation_writer is not None`. Always writes manifest, writes scrubbed metadata.pkl, renames temp dir to final path. Unconditional for all streaming captures.

Step 20 gate: same as step 19 AND `model_log._keep_activations_in_memory is False`. Nulls `.activation`, sets `.activation_ref` to a `LazyActivationRef` pointing at the final (post-rename) bundle path.

Sequence for `log_forward_pass(..., save_activations_to=path, keep_activations_in_memory=False)`:
1. Writer opens `<path>.tmp.<uuid4>/blobs/`.
2. During forward: each `save_tensor_data()` call writes `NNNNNNNNNN.safetensors` and records `blob_id` on the LayerPassLog (private field `_pending_blob_id`). `.activation` unchanged at this point.
3. Postprocess steps 1-18 run normally with tensors in memory.
4. Step 19 `_finalize_streamed_bundle` runs: writer computes sha256 per blob, writes `manifest.json`, writes scrubbed `metadata.pkl` (tensors already on disk → `BlobRef` stubs), atomic rename `<path>.tmp.<uuid>/` → `<path>/`. Always.
5. Step 20 `_evict_streamed_activations` runs (only if `keep_activations_in_memory=False`): iterates `_pending_blob_id` entries, nulls `.activation`, builds `LazyActivationRef` with **final path** (post-rename).
6. ModelLog returned.

If `keep_activations_in_memory=True` (default), step 19 still runs and the bundle is finalized; step 20 is skipped. The returned ModelLog has activations in memory AND disk-backed.

**Streaming strictness** — addresses Round-3 finding #6:
Streaming is **always strict**. If `tensor_policy` returns an error on an activation mid-stream (sparse, meta, nested, DTensor, etc.), the writer aborts: the `.tmp.<uuid>/` dir is kept with `PARTIAL` + `REASON` files, `TorchLensIOError` is raised with the offending blob_id + layer label. No lenient streaming path. Rationale: streaming is a live capture pipeline; silently skipping a layer mid-pass is too confusing. Users who need lenient behavior should capture normally, then call `torchlens.save(..., strict=False)` post-hoc.

**Non-tensor `activation_postfunc` guard** — addresses Round-2 finding #4.
Single rule applied to both streaming and post-hoc save: if `activation_postfunc is not None` AND `save_activations_to is not None` (streaming) OR `torchlens.save()` is called with `include_activations=True` (post-hoc), validate that `activation_postfunc` output type is `torch.Tensor`. Check is performed:
- **Streaming**: at the first `save_tensor_data()` call where the writer would be invoked. If the post-postfunc value is not a `torch.Tensor`, the writer is aborted immediately, `.tmp.<uuid>/` dir gets a `PARTIAL` sentinel + `REASON` file describing the issue, `TorchLensIOError` is raised, forward pass terminates. No partially-written bundle masquerades as valid.
- **Post-hoc save**: `torchlens.save()` inspects the first LayerPassLog with a saved activation; if the activation's type differs from `torch.Tensor` (the postfunc output was retained), raise `TorchLensIOError` before any blob is written.

**Atomicity scope** (unchanged from v2): random uuid temp dir; local-same-fs best-effort; non-local FS documented as unsupported.

**Temp-dir cleanup**: `torchlens.cleanup_tmp(path, force=False)` finds sibling `<path>.tmp.*` directories with a `PARTIAL` sentinel and removes them. Without `PARTIAL`, only removes after user confirmation.

**Gradient streaming**: out of scope. Post-hoc `torchlens.save(model_log, path, include_gradients=True)` bundles any populated `.gradient` fields.

### Fork E. Unsupported-tensor policy
**Decision**: **`strict=True` is the default for `torchlens.save()`**. Lenient skip is opt-in.

Rationale — addresses Round-2 finding #6:
- Silent partial bundles are an unsafe default for research/CI artifacts.
- Users hitting a sparse/meta/quantized tensor deserve a loud error that tells them to deal with it.
- `strict=False` remains available for users who knowingly want best-effort archival.

Supported tensor matrix (unchanged from v2 in scope):

| Tensor kind | strict=True | strict=False |
|-------------|-------------|--------------|
| Dense CPU/CUDA standard dtypes | Save | Save |
| Non-contiguous | `.contiguous()` then save | Same |
| Sparse COO/CSR, quantized, nested, tensor subclass, DTensor/FSDP shard, complex32 | `TorchLensIOError` with field name + reason | Skip, record in `manifest.unsupported_tensors`, warn once |
| Meta tensors | `TorchLensIOError` (no data) | Skip, record |

### Fork F. Version & integrity policy
**Decision**: PEP 440 parsing where applicable; sha256 over safetensors file bytes on disk.

Addresses Round-2 finding #7.

Version parsing:
- `torchlens_version`, `torch_version`, `python_version` parsed with `packaging.version.Version`. Nightly/pre-release identifiers compared per PEP 440 semantics.
- On parse failure (non-conforming string): fall back to string equality; emit `UserWarning` naming the unparseable version.
- Version policy table (Fork E in v2 plan, unchanged structure but explicit parse rules):

| Scenario | Load behavior |
|----------|---------------|
| `io_format_version > current` | Raise `TorchLensIOError` before opening `metadata.pkl` |
| `io_format_version < current` | Load with `DeprecationWarning` listing missing fields |
| `io_format_version == current` | Load normally |
| `torch_version` major mismatch | Raise `TorchLensIOError` (pre-open) |
| `torch_version` minor mismatch | Load with `UserWarning` |
| `torchlens_version` newer | Load with `UserWarning` |
| `torchlens_version` older | Load with info log only |
| `python_version` major mismatch | Attempt load; wrap any pickle error in `TorchLensIOError` with version hint |
| Safetensors backend missing | Raise `TorchLensIOError` with install hint |
| Manifest unparseable | Raise `TorchLensIOError` |
| Manifest blob_id without file | Raise `TorchLensIOError` (lists missing blobs) |
| Blob checksum mismatch | Raise `TorchLensIOError` (names blob_id) |
| Unknown extra files in `blobs/` | `UserWarning`, continue |

Checksum semantics:
- `sha256` field = hex of `hashlib.sha256(open(blob_path, "rb").read()).hexdigest()` — literal file bytes on disk after safetensors write.
- Checksum verified lazily on `materialize()` for `lazy=True` loads, eagerly on load for `lazy=False`.

### Fork G. Exception contract
**Decision**: all bundle-layer errors wrap in `TorchLensIOError`. Addresses Round-2 finding #12.

- `torchlens._io.TorchLensIOError` is the single exception type for bundle failures.
- Implementation: bundle save/load code catches `OSError`, `pickle.UnpicklingError`, `safetensors`-raised errors, JSON parse errors, key errors on manifest structure, and re-raises as `TorchLensIOError` with `__cause__` chained.
- Plain `pickle.load()` of a ModelLog via user-written code does not go through the wrap — callers of plain pickle accept Python's native pickle errors.
- Tested in `tests/test_io_bundle.py`.

### Fork H. Lazy-resave drift
**Decision**: record source-bundle manifest sha256 at load + verify each source blob's sha256 before fast-copy. Addresses Round-2 finding #8 + Round-3 finding #4.

Two layers of defense:
1. **Manifest-level**: at load time, capture `source_manifest_sha256 = sha256(manifest.json)` into ModelLog private `_source_bundle_manifest_sha256`. On resave, re-hash the source manifest and compare. If differs → `TorchLensIOError("source bundle manifest has changed since load; materialize refs and retry")`.
2. **Blob-level**: for each `LazyActivationRef` about to be fast-copied, re-hash the source blob file on disk and compare against the sha256 recorded in that manifest entry. If differs → `TorchLensIOError("blob at <path> tampered since load; materialize and retry")`.

Rationale: manifest-only check misses an attacker/corruption that replaces blob bytes but leaves the manifest intact. Per-blob verify catches it at the cost of reading the bytes once — still cheaper than decoding safetensors + re-encoding (no torch involvement).

If either check fails, the user must call `.materialize_activation()` on every ref (which itself does a sha256 check during materialize) and then resave — at which point the new blobs are written fresh from in-memory tensors.

### Fork I. Operational shape
**Decision**: directory-of-one-blob-per-tensor is the intentional format for this sprint. Finding #11 is **deferred**, not resolved.

Documentation note (`.project-context/knowledge/io_architecture.md` and README):
> "For models with >10,000 saved layers or deployment on network filesystems, bundle save creates many small files. A future release may add chunked blob formats. This sprint prioritizes random-access lazy loads over file-count optimization."

Not claiming resolution we haven't earned.

### Fork J. Filesystem safety
**Decision**: reject symlinks in bundle directories and temp dirs. Addresses Round-3 finding #7.

- `torchlens.save()` fails with `TorchLensIOError` if any path inside `<path>/` (after write) is a symlink. Save writes real files only.
- `torchlens.load()` rejects a bundle if `manifest.json`, `metadata.pkl`, the `blobs/` directory, or any file under `blobs/` is a symlink. `TorchLensIOError`.
- `torchlens.cleanup_tmp(path)` rejects `.tmp.*` entries that are symlinks (not removed, reported instead). Only real directories directly under the expected parent are cleaned.
- Rationale: symlinks can escape the bundle root, hide tampered content outside the bundle, and make `cleanup_tmp` delete unrelated paths.

### Fork K. Thread-safety and reader lifecycle
**Decision**: no long-lived shared safetensors readers. `LazyActivationRef.materialize()` opens, reads, verifies sha256, and closes per call. Addresses Round-3 finding #9.

- Rationale: torchlens is already single-threaded (see existing `warn_parallel` in user_funcs). Shared mmap + refcounting would add complexity for zero benefit.
- Trade-off: materializing the same ref twice re-reads and re-hashes. Acceptable because users typically materialize once and cache via `.activation = <tensor>`.
- `cleanup()` becomes trivial: no open handles to close (documented).

---

## 2. Manifest Schema v1 (unchanged from v2)

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
  "n_auxiliary_blobs": 304,
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
    {"label": "sparse_layer:1", "kind": "activation", "reason": "sparse_coo layout not supported this release"}
  ]
}
```

`kind` values: `activation`, `gradient`, `captured_arg`, `child_version`, `rng_state`, `module_arg`, `func_config`.

---

## 3. Public API Surface

### Top-level `torchlens`
```python
torchlens.save(
    model_log: ModelLog,
    path: str | Path,
    *,
    include_activations: bool = True,
    include_gradients: bool = True,
    include_captured_args: bool = False,   # CHANGED v4: default False (bundle size + RAM)
    include_rng_states: bool = False,      # CHANGED v4: default False
    strict: bool = True,
    overwrite: bool = False,
) -> None

torchlens.load(
    path: str | Path,
    *,
    lazy: bool = False,
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = True,       # NEW v4: controls nested-tensor eager load
) -> ModelLog

torchlens.cleanup_tmp(
    path: str | Path,
    *,
    force: bool = False,
) -> list[Path]

torchlens.rehydrate_nested(             # NEW v4: power-user full materialization
    model_log: ModelLog,
    *,
    map_location: str | torch.device = "cpu",
) -> None
```

### On `ModelLog`
```python
model_log.save(path, **kwargs)
ModelLog.load(path, **kwargs)             # classmethod

model_log.to_pandas() -> pd.DataFrame
model_log.to_csv(path, **kwargs)
model_log.to_parquet(path, **kwargs)
model_log.to_json(path, *, orient="records")
```

### Accessors
```python
# existing:
model_log.layers.to_pandas() / .to_csv / .to_parquet / .to_json
model_log.modules.to_pandas() / ...

# NEW:
model_log.params.to_pandas() / .to_csv / .to_parquet / .to_json
model_log.buffers.to_pandas() / ...

# Per-log:
module_log.to_pandas() / ...
module_pass_log.to_pandas() / ...         # NEW, reads forward_args_summary
```

### Streaming at capture time
```python
log_forward_pass(
    ...,
    save_activations_to: str | Path | None = None,
    keep_activations_in_memory: bool = True,
    activation_sink: Callable[[str, torch.Tensor], None] | None = None,
)
```

### LazyActivationRef
```python
class LazyActivationRef:
    blob_id: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device_at_save: str
    source_bundle_path: Path
    kind: Literal["activation", "gradient"]

    def materialize(self, *, map_location="cpu") -> torch.Tensor: ...
```

### Resave-lazy semantics
```python
model_log = torchlens.load("a/", lazy=True)
model_log.save("b/")
# If source manifest sha256 matches the one captured at load: shutil.copy blobs from a/ to b/.
# If source manifest differs from recorded digest OR source deleted: raise TorchLensIOError
#   "source bundle at a/ has changed or is gone; call .materialize_activation() on all layers and retry."
# No silent drift.
```

---

## 4. Spec Decomposition

All specs must pass: `ruff check . --fix && mypy torchlens/ && pytest tests/ -m smoke -x --tb=short`.

### Group 1 — Foundations (parallel-safe)

**IO-S1. Pickle hardening: scrub policy + rehydrate loader + PORTABLE_STATE_SPEC lint + version tagging + accessor reconstruction**
- `torchlens/_io/__init__.py`: `IO_FORMAT_VERSION=1`, `TorchLensIOError`, `BlobRef` NamedTuple, `FieldPolicy` enum (keep/blob/blob_recursive/drop/stringify/weakref_strip).
- Add `PORTABLE_STATE_SPEC: dict[str, FieldPolicy]` class attr on ModelLog, LayerPassLog, LayerLog, ModuleLog, ModulePassLog, ParamLog, BufferLog, FuncCallLocation per Fork A table.
- `torchlens/_io/scrub.py`: `scrub_for_save(model_log, **include_flags) -> (scrubbed_state, list[BlobSpec])`. Driven purely by `PORTABLE_STATE_SPEC`.
- `torchlens/_io/rehydrate.py`: `rehydrate_model_log(scrubbed_state, manifest, bundle_path, *, lazy, map_location, materialize_nested)`. Single entry point for all blob materialization. `__setstate__` only does default-fill + weakref restore; never touches disk.
- `torchlens/_io/accessor_rebuild.py`: shared helper extracted from `postprocess/finalization.py:671-681`.
- Expand `__setstate__` on ModelLog, LayerPassLog, LayerLog, ParamLog, BufferLog, ModuleLog, ModulePassLog with version-aware default-fill.
- Files: `torchlens/_io/__init__.py`, `scrub.py`, `rehydrate.py`, `accessor_rebuild.py` (all NEW); `torchlens/data_classes/{model_log,layer_pass_log,layer_log,param_log,buffer_log,module_log,func_call_location}.py`; `torchlens/postprocess/finalization.py` (extract helper).
- Tests:
  - `tests/test_io_pickle.py` (NEW) — scrub round-trip; accessor rebuild; default-fill from forged older pickle; version hard-fail on newer; plain-pickle unchanged.
  - `tests/test_io_scrub_policy.py` (NEW) — **PORTABLE_STATE_SPEC completeness lint**: for each class, instantiate a canonical live example, walk `vars(instance)`, assert every attribute name appears in `PORTABLE_STATE_SPEC`. Fails CI if any new attribute added without a policy entry. Moved from S7 to S1 per Round-3 finding #8.
- Size: **~440 LoC** (up from 400 due to PORTABLE_STATE_SPEC surfaces + rehydrate loader separation).

**IO-S2. ParamAccessor / BufferAccessor / ModulePassLog.to_pandas() + forward_args_summary + forward_kwargs_summary**
- New FIELD_ORDER tuples in `constants.py`.
- Recursive summary formatter in `torchlens/data_classes/_summary.py` (new) producing strings per Fork C rules.
- `ModulePassLog.forward_args_summary` and `forward_kwargs_summary` fields populated in `_build_module_logs` BEFORE the existing GC step.
- New `to_pandas()` methods on ParamAccessor, BufferAccessor, ModulePassLog.
- Files: `torchlens/data_classes/{param_log,buffer_log,module_log,_summary}.py`, `torchlens/constants.py`, `torchlens/postprocess/finalization.py` (add summary capture before GC).
- Tests: `tests/test_io_pandas.py` (NEW) — schema shape; empty accessor; summary correctness with mixed tensor/primitive/list/dict/kwargs args; summary survives GC; both `_args_summary` and `_kwargs_summary` present.
- Size: **~280 LoC** (up from 240; adds kwargs summary + recursive formatter).

### Group 2 — Exports + bundle (serial: S3 after S2; S4 after S1)

**IO-S3. Export wrappers: to_csv / to_parquet / to_json on all seven DataFrame producers**
- Wrappers on ModelLog, LayerAccessor, ModuleAccessor, ModuleLog, ModulePassLog, ParamAccessor, BufferAccessor.
- `pyarrow` optional via `pip install torchlens[io]`; clear `ImportError` if missing.
- Files: `torchlens/data_classes/{interface,layer_log,module_log,param_log,buffer_log}.py`, `pyproject.toml`.
- Tests: `tests/test_io_export.py` — round-trip + schema stability + ImportError on parquet.
- Size: **~220 LoC**.

**IO-S4. Bundle save/load (eager post-hoc) + manifest v1 + integrity + exception contract**
- `torchlens/_io/bundle.py`: `save` (post-hoc) and `load`. Integrates scrub (S1), tensor_policy (below), and manifest.
- `torchlens/_io/manifest.py`: schema class, sha256 hashing, PEP 440 version parsing, policy enforcement.
- `torchlens/_io/tensor_policy.py`: implements supported-tensor matrix; returns `(Ok | SkipReason | Error)` per tensor; respects `strict` flag.
- All bundle errors wrap in `TorchLensIOError` (Fork G).
- ModelLog gets `.save()` / `.load()` sugar; `torchlens` exports `save`, `load`, `cleanup_tmp`.
- Files: `torchlens/_io/{bundle,manifest,tensor_policy}.py` (NEW); `torchlens/data_classes/model_log.py`; `torchlens/__init__.py`; `torchlens/user_funcs.py`.
- Tests: `tests/test_io_bundle.py` (NEW) — eager round-trip; strict default raises on sparse; `strict=False` skips + records; all 12 version-policy rows; corrupt manifest / missing blob / truncated safetensors / checksum tamper all raise `TorchLensIOError`; overwrite semantics; non-tensor postfunc rejected at save; include flags combos.
- Size: **~480 LoC**.

### Group 3 — Streaming + lazy (serial)

**IO-S5. Streaming-save during forward pass + postprocess steps 19+20 + streaming-strict + cleanup_tmp + symlink rejection**
- `torchlens/_io/streaming.py`: `BundleStreamWriter`. Open `<path>.tmp.<uuid>/blobs/` (reject if path exists as symlink), receive `(blob_id, tensor)` synchronously, consult `tensor_policy` per tensor (streaming is **always strict** — Round-3 finding #6), write safetensors. On `finalize()`: compute sha256 per blob, write manifest, write scrubbed `metadata.pkl`, rename temp dir → final path. Returns set of blob ids + final path for ref-patching.
- Instrument `LayerPassLog.save_tensor_data()` with writer callout; gated to exhaustive-pass call site only (`capture/output_tensors.py:866-900`).
- **Step 19 `_finalize_streamed_bundle`** registered in `postprocess/__init__.py`: always runs if writer present. Handles manifest write, scrubbed metadata.pkl write, atomic rename.
- **Step 20 `_evict_streamed_activations`** registered after step 19: runs only if writer present AND `keep_activations_in_memory=False`. Patches `LazyActivationRef` with final (post-rename) path.
- Non-tensor postfunc guard: writer verifies first received tensor is `torch.Tensor`; if not, abort, mark `.tmp.<uuid>/PARTIAL`, raise `TorchLensIOError` (Fork D rule).
- Symlink rejection: writer refuses to open if `<path>` exists as a symlink; load side already covered by S4 + Fork J.
- `torchlens.cleanup_tmp(path, force=False)` utility (rejects symlink temp dirs — Round-3 finding #7).
- Files: `torchlens/_io/streaming.py` (NEW), `torchlens/_io/bundle.py`, `torchlens/data_classes/{layer_pass_log,model_log}.py`, `torchlens/user_funcs.py`, `torchlens/postprocess/{__init__,finalization}.py`.
- Tests: `tests/test_io_streaming.py` (NEW) — blobs written during pass (verify timestamps); step 19 always runs and bundle is finalized on keep_activations_in_memory=True; step 20 gated correctly; mid-pass exception leaves PARTIAL; non-tensor postfunc aborts; sparse/meta/nested tensor mid-stream raises `TorchLensIOError` with PARTIAL preserved; lazy refs point to final path not temp; cleanup_tmp ignores symlinks; `activation_sink` callable path.
- Size: **~440 LoC** (up from 400 due to step split + strictness + symlink rejection).

**IO-S6. Lazy activation references + two-level drift guard + lazy-resave fast-copy + rehydrate_nested + no-shared-handles**
- Flesh out `torchlens/_io/lazy.py`: `LazyActivationRef(blob_id, shape, dtype, device_at_save, source_bundle_path, kind, expected_sha256)`; `materialize(map_location="cpu")` opens safetensors fresh, reads, verifies sha256 against `expected_sha256`, closes, returns tensor (Fork K: no shared readers).
- `LayerPassLog.materialize_activation(map_location="cpu") -> Tensor` (and `.materialize_gradient()`).
- `torchlens.rehydrate_nested(model_log, map_location="cpu")`: power-user function for when user did `lazy=True, materialize_nested=False`. Walks nested containers, replaces any remaining `BlobRef` with real tensors.
- ModelLog records `_source_bundle_manifest_sha256` at load time.
- Lazy-resave fast-copy path in `torchlens/_io/bundle.py::save` (Fork H two-layer):
  1. Verify `sha256(source/manifest.json) == _source_bundle_manifest_sha256`; else raise.
  2. For each `LazyActivationRef` about to be copied: verify `sha256(source/blobs/<blob_id>.safetensors) == ref.expected_sha256`; else raise.
  3. Only then `shutil.copy` to new bundle.
- Cleanup: `ModelLog.cleanup()` documents "no long-lived handles to close" (Fork K).
- Files: `torchlens/_io/lazy.py` (EXPAND), `torchlens/data_classes/{layer_pass_log,cleanup}.py`, `torchlens/_io/bundle.py`, `torchlens/_io/rehydrate.py`, `torchlens/validation/invariants.py` (clear-error path for lazy-only logs).
- Tests: `tests/test_io_lazy.py` (NEW) — shape/dtype accessible without disk read; materialize bit-exact vs eager; map_location honored; manifest-level drift (mutate manifest) raises; **blob-level drift (mutate a single blob file, leave manifest intact) also raises** (Round-3 finding #4); fast-copy on clean source; validation API raises clearly on lazy-only; `rehydrate_nested` populates nested BlobRefs; no open file handles leak after materialize.
- Size: **~400 LoC** (up from 360 due to per-blob drift + rehydrate_nested + expected_sha256 field).

### Group 4 — Integration tests + docs (parallel)

**IO-S7. Integration + corruption regression suite**
- End-to-end: capture → stream-save → lazy-load → materialize → to_parquet → re-load.
- Corruption battery: truncated safetensors, missing blob, corrupt pickle, tampered manifest, stale counts, unknown extra files, checksum mismatch. Each must raise `TorchLensIOError` with specific message.
- Forged older-format pickle loads with `DeprecationWarning`.
- DataParallel/DDP sanity skip (torchlens already `warn_parallel`; streaming inherits).
- Scrub-policy lint test: iterate all `FIELD_ORDER` columns, assert each has explicit policy entry.
- Files: `tests/test_io_integration.py` (NEW), `tests/test_io_plain_pickle.py` (NEW), minor updates to `test_save_new_activations.py`.
- Size: **~380 LoC**.

**IO-S8. Docs**
- Docstrings on every new public API with copy-pasteable examples.
- README "Save and Load" section.
- `.project-context/knowledge/io_architecture.md` (NEW): bundle layout, manifest schema, tensor matrix, version policy, operational-shape note (Fork I).
- Files: `torchlens/_io/__init__.py`, `torchlens/__init__.py`, `torchlens/data_classes/model_log.py`, `README.md`, `.project-context/knowledge/io_architecture.md` (NEW).
- Size: **~160 LoC**.

Total: ~2640 LoC (up from 2460 in v2).

---

## 5. Spec Dependency Graph

```
S1 ─┬─> S4 ─┬─> S5 ──> S6 ──> S7
    │       │
S2 ─┴─> S3 ─┘                S8 (docs, parallel with S7)
```

---

## 6. Risk Register

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Unsupported tensor layouts | strict=True default raises; strict=False opt-in records in manifest. |
| R2 | Scrub misses a new field added in a future release | S7 lint test enumerates FIELD_ORDER; CI fails if new field lacks policy entry. |
| R3 | Disk-full / crash mid-stream | `.tmp.<uuid>/PARTIAL` sentinel; `cleanup_tmp` utility. |
| R4 | Lazy refs inside nested containers confuse validation | Nested containers always eager-materialized; lazy only `.activation`/`.gradient`. |
| R5 | Lazy cleanup vs live safetensors handles | `cleanup()` closes handles before nulling. Tested. |
| R6 | Format drift in N+1 | `io_format_version` + migration helpers (S1). |
| R7 | CUDA bundle on CPU host | `map_location="cpu"` default; safetensors handles it. |
| R8 | Streaming writer blocks | Synchronous by design; documented. |
| R9 | pyarrow install weight | Optional via extras. |
| R10 | Many small files (directory format) | Documented tradeoff (Fork I); deferred to future sprint. |
| R11 | NFS/Windows rename edge cases | Documented: local-same-fs best-effort. |
| R12 | Source bundle mutated between lazy load and resave | Manifest sha256 check on resave; raise clearly. |
| R13 | DataParallel/DDP multiple capture | Torchlens is single-process; inherit existing warning. |
| R14 | Exception handling ambiguity | All bundle errors wrap `TorchLensIOError` (Fork G). |

---

## 7. Execution Plan

| Round | Specs | Gate |
|-------|-------|------|
| 1 | S1 ‖ S2 | Pickle scrub round-trip green; new pandas schemas green. |
| 2 | S3 (after S2) ‖ S4 (after S1) | All export formats green; bundle round-trip green; version policy table tested. |
| 3 | S5 (after S4) | Streaming + step 19 green; postprocess sees tensors; lazy refs point at final path. |
| 4 | S6 (after S5) | Lazy load + materialize bit-exact; source drift detected; resave fast-copy. |
| 5 | S7 ‖ S8 | Integration corruption battery green; docs complete. |
| 6 | Phase 5 dual review (me + `/codex:review`). |

---

## 8. Stop-and-Ask Triggers

- End of this plan revision — **NOW**
- Any public API signature change beyond additions
- Any bump of `io_format_version` in the future
- End of Phase 5 (which review findings to fix)

---

## 9. Success Criteria (binary)

- [ ] `torchlens.save(model_log, path)` + `torchlens.load(path)` round-trip everything on the supported-tensor matrix
- [ ] Six log-level DataFrames + `.to_csv/.to_parquet/.to_json` on all seven producers
- [ ] `log_forward_pass(..., save_activations_to=path)` streams during pass; postprocess unaffected
- [ ] `keep_activations_in_memory=False` evicts post-pass; lazy refs point to final (post-rename) path
- [ ] Lazy load preserves metadata access without disk read; `materialize_activation()` bit-exact
- [ ] Source-drift detection works
- [ ] All 14 version-policy rows tested
- [ ] All bundle errors raise `TorchLensIOError`
- [ ] Scrub-policy lint passes
- [ ] Plain pickle still works with deprecation warning on pre-sprint pickles only
- [ ] `pytest tests/ -m "not slow"` green
- [ ] `ruff check . && mypy torchlens/` green

---

## 10. Change Log vs Round 3 (addresses all 9 findings)

| Finding | Severity | Resolution |
|---------|----------|-----------|
| R3-1 Nested tensor rehydration (map_location + RAM) | HIGH | Rehydration moved from `__setstate__` to a dedicated loader entrypoint that accepts `map_location`. Added `materialize_nested: bool = True` load flag + `torchlens.rehydrate_nested()` power-user function. Flipped `include_captured_args` and `include_rng_states` defaults to `False` to reduce bundle size and RAM by default. (Fork A rehydration contract + Fork B + API surface.) |
| R3-2 Scrub surface not exhaustive | HIGH | Replaced `FIELD_ORDER` lint with per-class `PORTABLE_STATE_SPEC` covering every live attribute (including `_optimizer`, `parent_layer_log`, `LayerLog.func_applied`, `LayerLog.activation_postfunc`, etc.). Completeness lint walks `vars(instance)` against the spec. (Fork A scrub policy + S1 lint.) |
| R3-3 Step 19 gate contradictory | HIGH | Split into Step 19 `_finalize_streamed_bundle` (always runs when writer present) and Step 20 `_evict_streamed_activations` (conditional on `keep_activations_in_memory=False`). (Fork D + S5.) |
| R3-4 Manifest-only drift misses blob tampering | HIGH | Added per-blob sha256 verify before fast-copy. Each `LazyActivationRef` records `expected_sha256`; source blob file is re-hashed before any copy. (Fork H + S6.) |
| R3-5 forward_args_summary misses kwargs | MEDIUM | Added `forward_kwargs_summary`. Recursive formatter handles list/tuple/dict/primitives. (Fork C + S2.) |
| R3-6 Streaming unsupported-tensor policy | MEDIUM | Streaming is **always strict**; unsupported tensor mid-pass raises `TorchLensIOError` with PARTIAL sentinel. No lenient streaming. (Fork D + S5.) |
| R3-7 Symlink handling undefined | MEDIUM | Fork J: all bundle paths and `.tmp.*` entries must be real; symlinks rejected in save/load/cleanup_tmp. (S4 + S5.) |
| R3-8 Scrub lint lands too late | MEDIUM | Moved scrub-policy completeness lint from S7 to S1 tests. Guarantees active from the moment the scrubber lands. (S1.) |
| R3-9 Reader thread-safety model | MEDIUM | Fork K: no long-lived shared readers. `materialize()` opens/reads/closes per call. Torchlens remains single-threaded; no locks needed. (Fork K + S6.) |
