# TorchLens I/O Sprint Plan (Round 1)

**Branch**: `codex/io-sprint`
**Goal**: comprehensive, frictionless I/O for ModelLog and its children — pickle, pandas/CSV/Parquet/JSON, activation-to-disk (including on-the-fly streaming), symmetric loading.
**Out of scope**: visualization (graphviz, ELK, dagua).
**Inputs**: `.project-context/research/io-audit-claude.md`, `.project-context/research/io-audit-codex.md`.

---

## 1. Design Forks — Resolved

### Fork A. Primary save format
**Decision**: multi-format bundle — **pickle (nested state) + safetensors (tensor blobs) + manifest.json (version / shape index)**.

Why (resolving audit disagreement):
- Claude proposed parquet-primary. Parquet is fundamentally tabular; it cannot round-trip `func_rng_states`, `func_autocast_state`, `captured_args` (nested dicts with torch objects), or `func_applied` callables without lossy flattening.
- Codex proposed a bundle; this matches how the data shape actually is. Pickle handles nested Python, safetensors handles tensors safely and version-stably, a small manifest.json carries version metadata and enables partial/lazy reads.
- Parquet/CSV/JSON remain first-class **export** targets for tabular views (layers/passes/modules/params/buffers DataFrames), not the primary archival format.

Bundle layout:
```
<bundle>/
  manifest.json            # io_format_version, torchlens_version, torch_version, python_version, counts
  metadata.pkl             # ModelLog with activations stubbed to LazyActivationRef keyed by layer label
  activations/
    <layer_label_pass>.safetensors    # one file per saved tensor (activation) when present
    <layer_label_pass>.grad.safetensors  # one file per saved gradient when present
  frames/                  # optional source-context snapshots if save_source_context=True
```

Single-file alternative: `<bundle>.tlens` (.tar.gz of the directory). Both forms supported; directory is default (faster incremental access, friendlier to tools).

### Fork B. Default load granularity
**Decision**: **lazy by default** when a bundle contains activations; **eager when no activation sidecars** (nothing to defer).

Why:
- RAM pressure is the primary motivation of the sprint.
- `LazyActivationRef.materialize()` makes it one call to force-load.
- `torchlens.load(path, lazy=False)` for users who prefer the old behavior.

### Fork C. Pandas-export scope
**Decision**: DataFrames at **all five log levels**: layer-pass (ModelLog), aggregate layer, module, param, buffer.

Current gaps: ParamAccessor and BufferAccessor have no `to_pandas()`. Add them for symmetry.

### Fork D. Streaming-save architecture
**Decision**: single instrumentation point at `LayerPassLog.save_tensor_data()` (layer_pass_log.py:455-501). All three capture sites (`_make_layer_log_entry` at output_tensors.py:866, fast-path overwrite at output_tensors.py:608, source refresh at source_tensors.py:316) already converge there.

Mechanism: `log_forward_pass(..., save_activations_to: str | Path | None = None)`. When set, ModelLog holds an `_activation_writer` (protocol below) that receives `(layer_label_with_pass, tensor)` immediately after the clone/detach/device/postfunc pipeline inside `save_tensor_data`. The in-memory activation is still kept unless user also passes `keep_activations_in_memory=False`, in which case it is replaced with a `LazyActivationRef` pointing at the just-written sidecar.

Atomicity: writer writes to `<bundle>.tmp/activations/…` and renames on forward-pass success. On exception, the temp directory is kept for debugging (warning logged, user can delete or retry). No partial bundle ever exposed under the final path.

### Fork E. Backward compatibility
**Decision**: version-tagged pickle + loader-side default-fill for `N-1` minor versions.

Add:
- `IO_FORMAT_VERSION = 1` constant in `torchlens/_io/__init__.py`
- `io_format_version` field on ModelLog (set in `__init__`)
- `__setstate__` on ModelLog, LayerPassLog, LayerLog, ParamLog, BufferLog, ModuleLog: read version from state, default-fill missing fields, warn if loaded from a newer torchlens than current
- manifest.json also carries version so bundle loaders can check before opening the pickle

---

## 2. Public API Surface (proposed)

### On `torchlens` (top-level)
```python
torchlens.save(model_log, path, *,
               include_activations=True,
               include_gradients=True,
               format="bundle",      # "bundle" | "tar" (tar.gz of bundle)
               compression="zstd")   # for safetensors-compatible compression of tensors
    -> None

torchlens.load(path, *,
               lazy=True,
               map_location="cpu")   # passed to safetensors loader
    -> ModelLog
```

### On `ModelLog`
```python
# Save/load
model_log.save(path, **kwargs)        # sugar for torchlens.save(self, path, **kwargs)
ModelLog.load(path, **kwargs)         # classmethod, equivalent to torchlens.load

# Tabular exports (already exists: to_pandas)
model_log.to_pandas() -> pd.DataFrame
model_log.to_csv(path, **kwargs)      # one-row-per-layer-pass
model_log.to_parquet(path, **kwargs)
model_log.to_json(path, *, orient="records")

# Streaming — at capture time, not after the fact:
# log_forward_pass gets new kwargs:
#   save_activations_to: str | Path | None = None
#   keep_activations_in_memory: bool = True
#   activation_sink: Callable[[str, torch.Tensor], None] | None = None
```

### On accessors (new / expanded)
```python
# Existing on ModelLog namespace:
model_log.layers.to_pandas() -> pd.DataFrame       # LayerAccessor (exists)
model_log.modules.to_pandas() -> pd.DataFrame      # ModuleAccessor (exists)
# New:
model_log.params.to_pandas() -> pd.DataFrame       # ParamAccessor — ADD
model_log.buffers.to_pandas() -> pd.DataFrame      # BufferAccessor — ADD

# All accessors gain:
.to_csv(path), .to_parquet(path), .to_json(path)
```

### LazyActivationRef (new)
```python
class LazyActivationRef:
    path: Path
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device   # recorded device at save time

    def materialize(self, *, map_location=None) -> torch.Tensor: ...
    # Implicit materialization on direct access (attribute passthrough for shape/dtype is free,
    # any torch op triggers materialize). NOT a torch.Tensor subclass — keep the distinction
    # explicit; user code that needs a tensor calls .materialize() or accesses .activation which
    # materializes on read.
```

Rationale for not subclassing Tensor: subclassing breaks `isinstance(x, torch.Tensor)` semantics in subtle ways and fights `__torch_function__`. Explicit materialization is honest and cheap.

### LayerPassLog helpers
```python
layer_pass.activation         # → torch.Tensor, or LazyActivationRef if lazy and not yet materialized
layer_pass.materialize_activation(*, map_location=None) -> torch.Tensor
layer_pass.has_saved_activations: bool  # True even when lazy
```

---

## 3. Spec Decomposition

Specs are Codex-dispatchable, self-contained, each must include unit tests and pass all three quality gates (`ruff check . --fix && mypy torchlens/ && pytest tests/ -m smoke -x --tb=short`). Execution groups run serially; specs inside a group are independent and CAN be dispatched in parallel but default is one-at-a-time per branch.

### Group 1 — Foundations (dispatch in parallel)

**IO-S1. Pickle hardening + version tagging + accessor reconstruction on load**
- Add `torchlens/_io/__init__.py` with `IO_FORMAT_VERSION = 1`, `TorchLensIOError`, helpers for default-fill.
- Add `io_format_version` field to ModelLog.
- Expand `ModelLog.__setstate__` to:
  - Read version, warn on unknown-newer, default-fill missing fields.
  - Rebuild `_module_logs` via ModuleAccessor and `_buffer_accessor` via BufferAccessor using the same logic in `postprocess/finalization.py:671-681` (extract to a shared helper).
- Add `__setstate__` default-fill to LayerPassLog, LayerLog, ParamLog, BufferLog, ModuleLog.
- Files: `torchlens/_io/__init__.py` (NEW), `torchlens/data_classes/model_log.py`, `layer_pass_log.py`, `layer_log.py`, `param_log.py`, `buffer_log.py`, `module_log.py`, `postprocess/finalization.py` (extract helper only).
- Tests: `tests/test_io_pickle.py` (NEW) — round-trip preserves all FIELD_ORDER columns; accessors functional after load; default-fill from stubbed older pickle; version warning on newer.
- Size: ~240 LoC.

**IO-S2. ParamAccessor.to_pandas() + BufferAccessor.to_pandas()**
- Mirror the field-selection pattern from LayerAccessor.to_pandas (`layer_log.py:665-687`).
- Define param and buffer DataFrame schemas in FIELD_ORDER-style tuples in `torchlens/constants.py`.
- Files: `torchlens/data_classes/param_log.py`, `buffer_log.py`, `constants.py`.
- Tests: `tests/test_io_pandas.py` (NEW) — schema shape, NaN handling, empty-accessor case.
- Size: ~160 LoC.

### Group 2 — Tabular exports and bundle save/load (serial, S1 must land first)

**IO-S3. Export wrappers: to_csv / to_parquet / to_json on all DataFrames**
- Add `to_csv(path, **kwargs)`, `to_parquet(path, **kwargs)`, `to_json(path, *, orient="records", **kwargs)` to: ModelLog (via `interface.py`), LayerAccessor, ModuleAccessor, ModuleLog, ParamAccessor, BufferAccessor.
- pyarrow added as optional dep with import-time lazy check; parquet methods raise a clear `ImportError` with install hint if missing.
- Files: `torchlens/data_classes/interface.py`, `layer_log.py`, `module_log.py`, `param_log.py`, `buffer_log.py`, `pyproject.toml` (add `pyarrow` to `[extra] io`).
- Tests: `tests/test_io_export.py` (NEW) — round-trip csv/parquet/json, schema stability, optional-dep error.
- Size: ~200 LoC. DEPENDS ON: S2.

**IO-S4. Bundle save/load (pickle + safetensors + manifest)**
- Implement `torchlens.save(model_log, path, ...)` and `torchlens.load(path, ...)` in `torchlens/_io/bundle.py`.
- Add safetensors as optional dep; bundle format falls back to `torch.save` for tensors if safetensors unavailable (warn-once).
- Manifest schema v1: `{io_format_version, torchlens_version, torch_version, python_version, created_at, n_layers, n_activations, n_gradients, bundle_format}`.
- ModelLog gets `.save(path, ...)` sugar and `ModelLog.load(path)` classmethod.
- tar.gz (`format="tar"`) writes the same directory layout and streams into a single `.tlens` file.
- Files: `torchlens/_io/bundle.py` (NEW), `torchlens/_io/manifest.py` (NEW), `torchlens/_io/__init__.py`, `torchlens/data_classes/model_log.py`, `torchlens/__init__.py` (export `save`/`load`), `torchlens/user_funcs.py` (re-export).
- Tests: `tests/test_io_bundle.py` (NEW) — round-trip eager, round-trip lazy, tar form, missing safetensors fallback, manifest mismatch, disk-full mid-write (monkeypatched).
- Size: ~320 LoC. DEPENDS ON: S1.

### Group 3 — Streaming + lazy (serial, S4 must land first)

**IO-S5. Streaming-save during forward pass**
- Add `save_activations_to: str | Path | None` and `keep_activations_in_memory: bool = True` and `activation_sink: Callable[[str, torch.Tensor], None] | None = None` to `log_forward_pass` (user_funcs.py).
- When `save_activations_to` is set: write directly to `<path>.tmp/` during the pass, rename to `<path>` on success; on exception, keep `.tmp/` and emit a warning with a resume hint.
- ModelLog gains a private `_activation_writer` attribute (contextvar-friendly).
- Instrument `LayerPassLog.save_tensor_data()` (layer_pass_log.py:455-501): after the current tensor-prep pipeline (clone / detach / output_device / postfunc under pause_logging), call `model_log._activation_writer(label_with_pass, self.activation)` when present; if `keep_activations_in_memory=False`, replace `self.activation` with a `LazyActivationRef`.
- Files: `torchlens/data_classes/layer_pass_log.py`, `torchlens/data_classes/model_log.py`, `torchlens/user_funcs.py`, `torchlens/_io/streaming.py` (NEW), `torchlens/_io/lazy.py` (NEW, stub here — fleshed out in S6).
- Tests: `tests/test_io_streaming.py` (NEW) — artifacts written during pass, mid-pass exception leaves `.tmp/`, atomic rename, sink callback, `keep_activations_in_memory=False` evicts tensor, round-trip with `torchlens.load`.
- Size: ~270 LoC. DEPENDS ON: S4.

**IO-S6. Lazy activation references**
- Flesh out `LazyActivationRef` in `torchlens/_io/lazy.py`: holds `path`, `shape`, `dtype`, `device_at_save`; `materialize(map_location=None) -> Tensor` reads safetensors.
- `LayerPassLog.materialize_activation(map_location=None)` materializes and replaces `self.activation` in place.
- Bundle loader (from S4) returns ModelLog with activations populated as `LazyActivationRef` when `lazy=True`.
- Non-subclass approach: direct attribute access to `.activation` returns the ref; user must call materialize to get a tensor. `.tensor_shape` / `.tensor_dtype` / `.tensor_memory` on LayerPassLog already provide metadata without materializing.
- Files: `torchlens/_io/lazy.py` (EXPAND), `torchlens/data_classes/layer_pass_log.py`, `torchlens/_io/bundle.py` (already present from S4 but loader wiring lives here).
- Tests: `tests/test_io_lazy.py` (NEW) — lazy load preserves shape/dtype metadata without reading tensor, materialize correctness (bitwise equality with eager load), device mapping, memory footprint.
- Size: ~240 LoC. DEPENDS ON: S4, S5.

### Group 4 — Tests + docs (serial, after S6)

**IO-S7. Integration regression suite**
- End-to-end: capture → save (streaming and post-hoc) → load (lazy and eager) → to_pandas → to_parquet round-trips.
- Large-model memory-footprint test (skipped with `@pytest.mark.slow` if too heavy for CI).
- Cross-version default-fill (forge an older-format pickle by hand; assert load succeeds with warning).
- Safetensors-missing fallback path.
- Files: `tests/test_io_integration.py` (NEW), minor updates to existing `test_save_new_activations.py`.
- Size: ~280 LoC. DEPENDS ON: S1-S6.

**IO-S8. Docs + public API**
- Module-level docstrings on new APIs with copy-pasteable examples.
- Update `README.md` (short I/O section) and `torchlens/__init__.py` docstring surface.
- Add `.project-context/knowledge/io_architecture.md` describing bundle layout and format-version policy.
- No release notes in this sprint — user gates that.
- Files: `torchlens/_io/__init__.py`, `torchlens/_io/bundle.py`, `torchlens/_io/lazy.py`, `torchlens/_io/streaming.py`, `torchlens/__init__.py`, `torchlens/data_classes/model_log.py`, `README.md`, `.project-context/knowledge/io_architecture.md` (NEW).
- Size: ~120 LoC.

---

## 4. Risk Register (sprint-local)

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Safetensors can't store arbitrary dtypes (meta, sparse) | Fallback to `torch.save` for those layers, record in manifest. |
| R2 | `func_applied` callable identity drift across torch versions breaks pickle | Already present as a risk; document. Not in sprint scope to fix. |
| R3 | Disk full mid-stream leaves orphan `.tmp/` | Emit clear warning with path + `torchlens.cleanup_tmp(path)` utility. |
| R4 | Lazy refs leak through pandas export (saved as `<LazyActivationRef>` repr) | DataFrame exporters never touch `.activation` directly; only metadata fields. Add assertion in tests. |
| R5 | Circular-ref cleanup (`ModelLog.cleanup()`) after lazy load deletes refs that materialize would need | `cleanup()` becomes aware of lazy state: closes bundle handles before clearing. Unit test. |
| R6 | pickle-format drift in N+1 sprint breaks N-1 loads | `io_format_version` bump + explicit migration table. Test both directions. |
| R7 | CUDA tensor loaded on CPU-only host | `map_location="cpu"` default; `torch.load`-style semantics on safetensors loader. |
| R8 | Large tensor single-file write blocks event loop in streaming | Streaming writer is synchronous; document. Async writer is out of scope. |
| R9 | Parquet dependency (pyarrow) adds install weight | Make optional via extras; clear ImportError if used without install. |
| R10 | `include_activations=True` bundle is huge for multi-GB models; user doesn't realize | Save emits a summary: "Wrote N activations, X GB total." |

---

## 5. Execution Plan

| Round | Specs | Gate after |
|-------|-------|------------|
| 1 | S1, S2 (parallel-safe, different files) | All quality gates green on branch |
| 2 | S3, S4 (serial: S3 after S2 lands; S4 after S1 lands — can overlap) | Bundle round-trip proven |
| 3 | S5, S6 (S5 first, S6 reads S5's writer) | Streaming + lazy E2E proven |
| 4 | S7, S8 (parallel) | Full regression suite green |
| 5 | Adversarial review (Codex) + self-review — Phase 5 of sprint | User-facing findings list |

One feature branch `codex/io-sprint`. Each spec is one Codex dispatch. Quality gates mandatory before next dispatch.

---

## 6. Stop-and-Ask Triggers (from directive)

- End of Phase 2 (this document) — **NOW**
- Any public API signature change beyond additions — none planned; flag if it comes up
- New dep (`safetensors`, `pyarrow`): both optional via extras; flag if consensus is to make required
- Pickle-format break vs N-1 — none planned; flag if it becomes unavoidable
- End of Phase 5 — which review findings to fix

---

## 7. Success Criteria (binary)

- [ ] `model_log.save(path)` and `torchlens.load(path)` round-trip tensors (lazy and eager)
- [ ] `model_log.to_pandas()` + `.to_csv` + `.to_parquet` + `.to_json` at 5 log levels
- [ ] `log_forward_pass(..., save_activations_to=path)` streams to disk during pass
- [ ] Activations can be loaded back lazily without RAM spike
- [ ] `pytest tests/ -m "not slow"` green; all quality gates green
- [ ] No regression in existing smoke tests
- [ ] Every new public API has docstring with example
- [ ] Old pickles (from torchlens 1.0.2) load with a deprecation warning
