# TorchLens I/O Sprint Plan (Round 3)

**Branch**: `codex/io-sprint`
**Goal**: comprehensive, frictionless I/O for ModelLog and its children — pickle, pandas/CSV/Parquet/JSON, activation-to-disk (including streaming during forward pass), symmetric loading.
**Out of scope**: visualization (graphviz, ELK, dagua), tar archival utilities, gradient streaming during backward.
**Inputs**: `.project-context/research/io-audit-{claude,codex}.md`, `.project-context/plans/io-sprint/review_round_{1,2}.md`.
**This revision addresses all 12 Round-2 findings. Change log at bottom.**

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

**Pickle-scrub contract (complete field policy)** — addresses Round-2 finding #1, #2.

Before `metadata.pkl` is written, the scrubber walks the ModelLog state and applies this policy exhaustively (deny-list first, then allow-list for each class):

| Field | Class | Policy |
|-------|-------|--------|
| `activation` | LayerPassLog | Replace tensor with `BlobRef(blob_id, kind="activation")`. Scalars (0-d) recorded inline in manifest. |
| `gradient` | LayerPassLog | Replace tensor with `BlobRef(blob_id, kind="gradient")`. |
| `captured_args`, `captured_kwargs` | LayerPassLog | If `include_captured_args=True` (default): walk recursively, each tensor → `BlobRef(kind="captured_arg")`. Non-tensor leaves preserved. If `include_captured_args=False`: set to `None`. |
| `children_tensor_versions` | LayerPassLog | If `include_captured_args=True`: each tensor → `BlobRef(kind="child_version")`. Else `None`. |
| `func_rng_states` | LayerPassLog | If `include_rng_states=True` (default): each tensor → `BlobRef(kind="rng_state")`. Else `None`. |
| `forward_args`, `forward_kwargs` | ModulePassLog | If `include_captured_args=True`: walk, tensors → `BlobRef(kind="module_arg")`. Else `None`. |
| `func_applied` | LayerPassLog | Drop unconditionally (recorded by `func_name` string). |
| `activation_postfunc` | ModelLog | Drop unconditionally; record `activation_postfunc_repr: str` if present. |
| `_param_ref` | ParamLog | Drop unconditionally (pins `nn.Parameter`, not portable). |
| `_frame_func_obj` | FuncCallLocation | Clear unconditionally (already done by disabled-source path). |
| `func_config` | LayerPassLog | Recursively walk: primitives (bool/int/float/str/None) kept; tensors → `BlobRef(kind="func_config")`; anything else (user subclasses, unknown objects) → replaced with `"<scrubbed:TypeName>"` string placeholder. |
| `extra_attributes` | ModuleLog, ModelLog, any data class exposing one | Same policy as `func_config`: primitives + containers of primitives pass; tensors blobified; other objects stringified. |
| Any attribute not enumerated above | Any | Default: attempt pickle. On `PicklingError` during scrub, replace with `"<scrubbed:TypeName>"` and log at `WARNING`. |

The scrub policy is implemented once in `torchlens/_io/scrub.py` with unit tests that enumerate every known `FIELD_ORDER` column. Adding a new field requires updating the policy (guarded by a lint check in S7 tests).

**Rehydration contract on load** — addresses Round-2 finding #1 fully.

On `torchlens.load(path, lazy=False)` (default):
- `.activation` / `.gradient`: fully materialized to `torch.Tensor`; `.activation_ref` / `.gradient_ref` also populated for idempotency.
- All other `BlobRef`s in nested containers (`captured_args`, `captured_kwargs`, `children_tensor_versions`, `func_rng_states`, `forward_args`, `forward_kwargs`, `func_config`): **eagerly materialized** during `__setstate__` — replaced in place with the real tensor. Validation and replay code sees real tensors everywhere.

On `torchlens.load(path, lazy=True)` (opt-in):
- `.activation` / `.gradient`: **NOT** materialized. `.activation_ref` / `.gradient_ref` populated; `.activation` / `.gradient` are `None`.
- Nested container tensors: **still eagerly materialized** (lazy never applies to nested fields). Keeps lazy simple and keeps validation paths intact.

This means lazy mode only defers the two fields that dominate memory cost (activations + gradients). Everything else loads eagerly.

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

`ModulePassLog.forward_args_summary` new field — addresses Round-2 finding #9.
- During postprocess step `_build_module_logs`, **before** the existing GC step that clears `forward_args`/`forward_kwargs`, a short summary string is computed: `[shape=(1,3,224,224) dtype=float32]` per tensor arg, primitives rendered directly. Stored as `forward_args_summary: str` on ModulePassLog.
- `ModulePassLog.to_pandas()` reads from the summary field, not from live `forward_args`.
- Existing GC behavior unchanged.

### Fork D. Streaming-save architecture
**Decision**: during forward pass, write the sidecar at `save_tensor_data()` completion; activation remains in memory until a new postprocess step `_evict_streamed_activations` runs. Selective streaming hooks the exhaustive pass only.

Concrete pipeline position — addresses Round-2 finding #3.

Current 18-step postprocess pipeline (from `postprocess/__init__.py`): numbered steps running in order. This sprint adds step 19 at the end:
```
Step 19: _evict_streamed_activations   (NEW)
```
Gate: runs only if `ModelLog._activation_writer is not None` AND `model_log._keep_activations_in_memory is False`.

Sequence for `log_forward_pass(..., save_activations_to=path, keep_activations_in_memory=False)`:
1. Writer opens `<path>.tmp.<uuid4>/blobs/`.
2. During forward: each `save_tensor_data()` call writes `NNNNNNNNNN.safetensors` and records `blob_id` on the LayerPassLog (new private field `_pending_blob_id`). `.activation` unchanged.
3. Postprocess steps 1-18 run normally with tensors in memory.
4. Step 19 runs: writer finalizes `manifest.json` with full blob table + checksums, writes `metadata.pkl` (scrubbed), renames `<path>.tmp.<uuid>/` → `<path>/`. On successful rename, iterates all `_pending_blob_id` entries and replaces `LayerPassLog.activation` with `None`, sets `.activation_ref = LazyActivationRef(blob_id=..., source_bundle_path=<final_path>)`. Ref holds the FINAL path, never the temp path.
5. ModelLog is returned with lazy refs pointing at the stable `<path>`.

If `keep_activations_in_memory=True` (default), Step 19 still runs manifest + metadata.pkl + rename, but **does not** null out `.activation`. The returned ModelLog has activations in memory AND disk-backed.

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
**Decision**: record source-bundle manifest sha256 at load time; verify at resave. Addresses Round-2 finding #8.

- `LazyActivationRef.source_bundle_path` records bundle dir.
- Loader captures `source_manifest_sha256 = sha256(manifest.json)` at load time into ModelLog private `_source_bundle_manifest_sha256`.
- On `model_log.save(new_path)` fast-copy path: re-hash source manifest; if it doesn't match recorded digest, raise `TorchLensIOError` telling user to materialize all refs and retry.
- Prevents silent corruption from concurrent source-bundle mutation.

### Fork I. Operational shape
**Decision**: directory-of-one-blob-per-tensor is the intentional format for this sprint. Finding #11 is **deferred**, not resolved.

Documentation note (`.project-context/knowledge/io_architecture.md` and README):
> "For models with >10,000 saved layers or deployment on network filesystems, bundle save creates many small files. A future release may add chunked blob formats. This sprint prioritizes random-access lazy loads over file-count optimization."

Not claiming resolution we haven't earned.

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
    include_captured_args: bool = True,
    include_rng_states: bool = True,
    strict: bool = True,              # CHANGED from v2: strict is now default
    overwrite: bool = False,
) -> None

torchlens.load(
    path: str | Path,
    *,
    lazy: bool = False,               # CHANGED from v2: lazy is now opt-in
    map_location: str | torch.device = "cpu",
) -> ModelLog

torchlens.cleanup_tmp(
    path: str | Path,
    *,
    force: bool = False,
) -> list[Path]
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

**IO-S1. Pickle hardening: complete scrub policy + version tagging + accessor reconstruction**
- `torchlens/_io/__init__.py`: `IO_FORMAT_VERSION=1`, `TorchLensIOError`, `BlobRef` NamedTuple.
- `torchlens/_io/scrub.py`: full policy table from Fork A. One public entry `scrub_for_save(model_log, **include_flags) -> (scrubbed_state, list[BlobSpec])`. Walks every class field; tested by enumerating every FIELD_ORDER column and asserting the policy is applied correctly.
- `torchlens/_io/rehydrate.py`: inverse — reads scrubbed state + loads tensors from disk, reassembles nested structures, returns live ModelLog.
- Expand `__setstate__` on ModelLog, LayerPassLog, LayerLog, ParamLog, BufferLog, ModuleLog, ModulePassLog with version-aware default-fill.
- Accessor reconstruction helper extracted from `postprocess/finalization.py:671-681`.
- Files: `torchlens/_io/__init__.py`, `scrub.py`, `rehydrate.py`, `accessor_rebuild.py` (all NEW); `torchlens/data_classes/{model_log,layer_pass_log,layer_log,param_log,buffer_log,module_log}.py`; `torchlens/postprocess/finalization.py`.
- Tests: `tests/test_io_pickle.py` (NEW) — scrub policy coverage lint, round-trip scrubbed state, accessor rebuild, default-fill from forged older pickle, version hard-fail on newer, plain-pickle compat unchanged.
- Size: **~400 LoC**.

**IO-S2. ParamAccessor / BufferAccessor / ModulePassLog.to_pandas() + forward_args_summary field**
- New FIELD_ORDER tuples in `constants.py`.
- `ModulePassLog.forward_args_summary` field populated in `_build_module_logs` before existing GC. Short format: `"(shape=(1,3,224,224) dtype=float32, shape=(), dtype=int64)"`.
- New `to_pandas()` methods on ParamAccessor, BufferAccessor, ModulePassLog.
- Files: `torchlens/data_classes/{param_log,buffer_log,module_log}.py`, `torchlens/constants.py`, `torchlens/postprocess/finalization.py` (add summary capture).
- Tests: `tests/test_io_pandas.py` (NEW) — schema shape, empty accessor, summary correctness with mixed tensor/primitive args, summary survives GC.
- Size: **~240 LoC**.

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

**IO-S5. Streaming-save during forward pass + postprocess step 19 + cleanup_tmp**
- `torchlens/_io/streaming.py`: `BundleStreamWriter`. Open `<path>.tmp.<uuid>/blobs/`, receive `(blob_id, tensor)` synchronously, write safetensors. On `finalize()`: write manifest, write scrubbed `metadata.pkl`, rename to `<path>`, return set of blob ids for ref-patching.
- Instrument `LayerPassLog.save_tensor_data()` with writer callout; gated to exhaustive-pass call site only (`capture/output_tensors.py:866-900`).
- New postprocess step `_evict_streamed_activations` registered as step 19 in `postprocess/__init__.py`. Runs only when writer active AND `keep_activations_in_memory=False`. Patches LazyActivationRef with final (post-rename) path.
- Non-tensor postfunc guard: writer verifies first received tensor is `torch.Tensor`; if not, abort, mark `.tmp.<uuid>/PARTIAL`, raise `TorchLensIOError` (Fork D rule).
- `torchlens.cleanup_tmp(path, force=False)` utility.
- Files: `torchlens/_io/streaming.py` (NEW), `torchlens/_io/bundle.py`, `torchlens/data_classes/{layer_pass_log,model_log}.py`, `torchlens/user_funcs.py`, `torchlens/postprocess/{__init__,finalization}.py`.
- Tests: `tests/test_io_streaming.py` (NEW) — blobs written during pass (verify timestamps); mid-pass exception leaves PARTIAL; non-tensor postfunc aborts before writing blob; `keep_activations_in_memory=False` evicts after postprocess (step 19 ordering); lazy refs point to final path not temp; cleanup_tmp finds PARTIAL dirs; `activation_sink` callable path.
- Size: **~400 LoC**.

**IO-S6. Lazy activation references + source-drift guard + lazy-resave fast-copy**
- Flesh out `torchlens/_io/lazy.py`: `LazyActivationRef(blob_id, shape, dtype, device_at_save, source_bundle_path, kind)`; `materialize(map_location="cpu")` reads safetensors, verifies sha256 against manifest at first read, caches result.
- `LayerPassLog.materialize_activation(map_location="cpu") -> Tensor` (and `.materialize_gradient()`).
- ModelLog records `_source_bundle_manifest_sha256` at load time for drift detection.
- Lazy-resave fast-copy path in `torchlens/_io/bundle.py::save`: for each lazy-only layer, if source manifest digest matches recorded value, `shutil.copy` blob file from source bundle to new bundle; otherwise raise `TorchLensIOError` with remediation hint (Fork H).
- Cleanup: `ModelLog.cleanup()` closes any open safetensors handles before nulling refs.
- Files: `torchlens/_io/lazy.py` (EXPAND), `torchlens/data_classes/{layer_pass_log,cleanup}.py`, `torchlens/_io/bundle.py`, `torchlens/validation/invariants.py` (clear-error path for lazy-only logs).
- Tests: `tests/test_io_lazy.py` (NEW) — shape/dtype accessible without disk read; materialize bit-exact vs eager; map_location; source-drift detection (mutate source manifest between load and resave → raises); fast-copy on clean source; validation API raises clearly on lazy-only.
- Size: **~360 LoC**.

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

## 10. Change Log vs Round 2 (addresses all 12 findings)

| Finding | Resolution |
|---------|-----------|
| R2-1 Nested tensor rehydration | Nested tensor fields always eagerly materialized on load (Fork A rehydration contract). Lazy applies only to `.activation`/`.gradient`. |
| R2-2 Scrub allow-list holes | Fork A policy table covers `func_config`, `extra_attributes`, and default fallback for unknown fields. S7 has a scrub-policy lint test. |
| R2-3 Eviction pipeline position | Explicit: step 19 in `postprocess/__init__.py`; refs store post-rename final path (Fork D). |
| R2-4 Non-tensor postfunc inconsistency | Single rule in Fork D applied to both streaming and post-hoc save. |
| R2-5 Lazy default too aggressive | Flipped: `lazy=False` is default; lazy is opt-in (Fork B). |
| R2-6 Silent partial bundles | Flipped: `strict=True` is default for save (Fork E). |
| R2-7 Version parsing + checksum scope | PEP 440 parsing; sha256 over safetensors file bytes on disk (Fork F). |
| R2-8 Lazy-resave drift | Manifest sha256 captured at load, verified before fast-copy; raise on drift (Fork H). |
| R2-9 ModulePassLog summary capture | `forward_args_summary` field populated in `_build_module_logs` before existing GC (Fork C + S2). |
| R2-10 pack_bundle semantics | Cut from sprint. Deferred explicitly. |
| R2-11 Operational shape | Acknowledged as deferred, not claimed resolved (Fork I). |
| R2-12 Exception contract | All bundle-layer errors wrap in `TorchLensIOError` (Fork G). |
