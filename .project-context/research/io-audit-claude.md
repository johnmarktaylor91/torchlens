# TorchLens I/O Audit Report

**Audit Date**: 2026-04-23
**Scope**: Save/load/export functionality, pickle behavior, data class structure
**Status**: READ-ONLY audit (no modifications made)

---

## 1. INVENTORY

### Existing I/O Methods

| Location | Signature | Behavior |
|----------|-----------|----------|
| `user_funcs.py:695` | `validate_batch_of_models_and_inputs(...) -> pd.DataFrame` | Reads/writes validation results to CSV via `pd.read_csv()` and `to_csv()` |
| `data_classes/interface.py:421` | `to_pandas(self) -> pd.DataFrame` | Exports 59 fields from all LayerPassLog entries to single-level pandas DataFrame |
| `data_classes/layer_pass_log.py:455` | `save_tensor_data(t, t_args, t_kwargs, ...)` | Stores tensor via `safe_copy()`, optionally moves device, applies `activation_postfunc` |
| `data_classes/layer_pass_log.py:503` | `log_tensor_grad(grad: torch.Tensor)` | Stores gradient via `detach().clone()` (bare clone, not deep-copy) |
| `data_classes/model_log.py:337` | `__getstate__(self) -> Dict` | Pickle serialization: strips `_module_logs`, `_buffer_accessor`, `_module_build_data` weakrefs |
| `data_classes/model_log.py:346` | `__setstate__(self, state)` | Pickle deserialization: rebuilds accessors, restores `source_model_log` refs on all LayerPassLog/LayerLog/ModuleLog objects |
| `data_classes/layer_pass_log.py:390` | `__getstate__(self) -> Dict` | Pickle: strips `_source_model_log_ref` (weakref) |
| `data_classes/layer_pass_log.py:396` | `__setstate__(self, state)` | Pickle: restores `__dict__` from state |
| `data_classes/layer_log.py:216` | `__getstate__(self) -> Dict` | Pickle: strips `_source_model_log_ref` weakref |
| `data_classes/layer_log.py:222` | `__setstate__(self, state)` | Pickle: restores `__dict__` from state |
| `data_classes/layer_pass_log.py:417` | `copy(self)` | Selective-depth copy: deep-copies most fields except tensors, funcs, RNG states (documented) |
| `capture/trace.py:55` | `save_new_activations(model_log, model, input_args, ...)` | Re-runs model in fast mode, saves only requested layers; clears prior activations in place |
| `data_classes/cleanup.py:42` | `cleanup(self)` | Full teardown: deletes all attributes from ModelLog and LayerPassLog entries, breaks circular refs, calls `torch.cuda.empty_cache()` |
| `data_classes/cleanup.py:31` | `release_param_refs(self)` | Releases nn.Parameter references from all ParamLogs to allow model GC |

### No Explicit I/O Functions Found

- **No `save()` / `load()` / `export()` methods** on ModelLog or data classes
- **No torch.save / torch.load** calls in the codebase
- **No JSON, CSV, Parquet, HDF5, or SafeTensors serialization** beyond `to_pandas()` + manual CSV
- **No streaming-to-disk** during forward pass
- **No activation-to-disk** utilities
- **No format versioning** infrastructure

---

## 2. DATA CLASS SHAPE TABLE

### ModelLog

| Attribute | Type | Tensor? | Location | Notes |
|-----------|------|---------|----------|-------|
| `model_name` | str | No | model_log.py:164 | Model class name |
| `layer_list` | List[LayerPassLog] | Cross-ref | model_log.py:197 | Ordered list of all layer passes (post-processing) |
| `layer_dict_all_keys` | OrderedDict[str, LayerPassLog] | Cross-ref | model_log.py:199 | All lookup keys → entry |
| `layer_logs` | OrderedDict[str, LayerLog] | Cross-ref | model_log.py:202 | No-pass label → aggregate |
| `param_logs` | ParamAccessor | Cross-ref | model_log.py:258 | ParamLog objects for each parameter |
| `_module_logs` | ModuleAccessor | Cross-ref | model_log.py:279 | ModuleLog objects |
| `_buffer_accessor` | BufferAccessor | Cross-ref | model_log.py:230 | BufferLog objects |
| `total_activation_memory` | int | No | model_log.py:253 | Sum of all saved tensor sizes |
| `num_tensors_saved` | int | No | model_log.py:254 | Count of layers with saved activations |

### LayerPassLog

| Attribute | Type | Tensor? | Location | Notes |
|-----------|------|---------|----------|-------|
| `activation` | torch.Tensor | **Tensor** | layer_pass_log.py:135 | Actual output tensor (or None). Device/grad controlled by `output_device`, `detach_saved_tensor`. May be postprocessed via `activation_postfunc`. |
| `gradient` | torch.Tensor | **Tensor** | layer_pass_log.py:155 | Backward gradient (bare clone, not autograd-linked) |
| `has_saved_activations` | bool | No | layer_pass_log.py:136 | Whether `activation` is not None |
| `has_gradient` | bool | No | layer_pass_log.py:157 | Whether `gradient` is not None |
| `captured_args` | List[Any] | Mixed | layer_pass_log.py:141 | May contain cloned tensors + primitives |
| `captured_kwargs` | Dict[str, Any] | Mixed | layer_pass_log.py:142 | Same as above |
| `tensor_shape` | Tuple[int,...] | No | layer_pass_log.py:143 | Metadata only |
| `tensor_dtype` | torch.dtype | No | layer_pass_log.py:144 | Metadata only |
| `tensor_memory` | int | No | layer_pass_log.py:145 | Size in bytes |
| `func_name` | str | No | layer_pass_log.py:164 | Operation name (e.g., 'conv2d') |
| `func_applied` | Callable | No | layer_pass_log.py:163 | Direct reference to torch function (not pickled by default Python rules) |
| `parent_layers` | List[str] | No | layer_pass_log.py:205 | Layer labels (cross-references) |
| `child_layers` | List[str] | No | layer_pass_log.py:208 | Layer labels (cross-references) |
| `parent_param_logs` | List[ParamLog] | Cross-ref | layer_pass_log.py:187 | References to ParamLog objects |
| `func_rng_states` | Dict[str, torch.Tensor] | Mixed | layer_pass_log.py:169 | RNG state dicts (large, not mutated) |
| `func_autocast_state` | Dict | No | layer_pass_log.py:170 | Autocast context metadata |
| `children_tensor_versions` | Dict | Mixed | layer_pass_log.py:151 | RAW (not postprocessed) child tensor values for validation |
| `func_config` | Dict | No | layer_pass_log.py:267 | Hyperparameters (stride, padding, etc.) |

### LayerLog

| Attribute | Type | Tensor? | Location | Notes |
|-----------|------|---------|----------|-------|
| `layer_label` | str | No | layer_log.py:68 | Human-readable no-pass label |
| `func_applied` | Callable | No | layer_log.py:81 | Shared reference (not deep-copied) |
| `passes` | Dict[int, LayerPassLog] | Cross-ref | layer_log.py (implied) | Per-pass details |
| `_source_model_log_ref` | weakref | Cross-ref | layer_log.py:76 | Weakref to owning ModelLog |

### ParamLog

| Attribute | Type | Tensor? | Location | Notes |
|-----------|------|---------|----------|-------|
| `address` | str | No | param_log.py:51 | Full address (e.g., 'features.0.weight') |
| `name` | str | No | param_log.py:52 | Short name (e.g., 'weight') |
| `shape` | Tuple[int,...] | No | param_log.py:53 | Parameter shape |
| `dtype` | torch.dtype | No | param_log.py:54 | Parameter dtype |
| `num_params` | int | No | param_log.py:55 | Total # of scalar values |
| `memory` | int | No | param_log.py:56 | Size in bytes |
| `trainable` | bool | No | param_log.py:57 | Requires grad? |
| `_param_ref` | Optional[nn.Parameter] | **Tensor** (indirect) | param_log.py:66 | Direct reference to actual parameter (prevents GC, used for lazy grad access) |
| `_grad_shape`, `_grad_dtype`, `_grad_memory` | Metadata | No | param_log.py:73-75 | Lazy gradient metadata |

### ModuleLog / ModulePassLog

| Attribute | Type | Tensor? | Location | Notes |
|-----------|------|---------|----------|-------|
| `address` | str | No | module_log.py:123 | Module address in hierarchy |
| `layers` / `all_layers` | List[str] | No | module_log.py (MultiPass/Aggregate) | Layer label references |
| `forward_args` | Optional[tuple] | Mixed | module_log.py:147 | Captured module forward inputs (may contain tensors) |
| `forward_kwargs` | Optional[dict] | Mixed | module_log.py:148 | Captured module forward keyword inputs |

### BufferLog

| Attribute | Type | Tensor? | Location | Notes |
|-----------|------|---------|----------|-------|
| (extends LayerPassLog) | | | buffer_log.py:22 | Inherits all LayerPassLog fields; `buffer_address` field identifies the buffer |
| `buffer_address` | str (inherited) | No | layer_pass_log.py:224 | Full address (e.g., 'features.0.running_mean') |
| `name` (computed) | str | No | buffer_log.py:36 | Last segment of buffer_address |
| `module_address` (computed) | str | No | buffer_log.py:44 | Everything before last dot in buffer_address |

---

## 3. GAPS

### a. Round-trip Pickling ModelLog

**Status**: ✅ **PARTIAL SUPPORT** (basic pickle works, but with caveats)

**What exists**:
- `ModelLog.__getstate__()` strips `_module_logs`, `_buffer_accessor`, `_module_build_data` (weakref-backed accessors)
- `ModelLog.__setstate__()` rebuilds accessors and restores `source_model_log` refs on all child entries
- `LayerPassLog.__getstate__()` strips `_source_model_log_ref` weakref
- `LayerPassLog.__setstate__()` restores `__dict__`
- `LayerLog.__getstate__()` / `__setstate__()` follow same pattern

**Limitations**:
- `func_applied` (direct reference to torch function) may or may not pickle depending on PyTorch version/build
- `func_rng_states` contains torch RNG state dicts (pickle-able but version-brittle)
- `gradient` is stored as bare clone; after unpickling, gradients are stale (no connection to live parameters)
- `_param_ref` pins model parameters (prevents GC); unpickling does NOT restore live refs to nn.Parameters
- `children_tensor_versions` stores raw tensor data for validation; on round-trip, these are disconnected from the model
- **No test coverage for pickle round-trip in visible codebase**
- **No documentation** on pickle compatibility or version stability

**Risk**: High. Pickling works for metadata but activations/parameters become stale on load. Suitable for post-mortem inspection, not for resuming training.

---

### b. Pandas DataFrame Export

**Status**: ✅ **SINGLE-LEVEL ONLY**

**What exists**:
- `model_log.to_pandas()` (interface.py:421) exports 59 fields from all LayerPassLog entries
- One row per layer pass (not per module, not per parameter)
- Fields include metadata (shape, dtype, func_name), graph edges (parent_layers, child_layers), module/param metadata
- Automatic type conversion (int, bool, etc.)

**Missing**:
- No parameter-level export (one row per ParamLog)
- No module-level export (one row per ModuleLog)
- No buffer-level export (one row per BufferLog)
- No activation tensor values in DataFrame (only metadata)
- No hierarchical/multi-level index support
- **No `to_csv()` / `to_parquet()` / `to_json()` on ModelLog** (user must call `to_pandas().to_csv()` manually)

---

### c. CSV / Parquet / JSON Export

**Status**: ❌ **NOT IMPLEMENTED (except CSV via to_pandas().to_csv())**

**What exists**:
- `validate_batch_of_models_and_inputs()` writes results CSV via `pd.to_csv()`
- User can call `model_log.to_pandas().to_csv()` manually

**Missing**:
- No native `model_log.to_csv()` method
- No `to_parquet()` support
- No `to_json()` support
- No schema versioning for exported formats
- No round-trip deserialization

**Note**: numpy arrays (parent_layers, parent_param_shapes, etc.) in DataFrame require custom encoding for JSON/Parquet.

---

### d. Activation-to-Disk

**Status**: ❌ **NOT IMPLEMENTED**

**Current behavior**:
- All saved activations live in RAM in `layer_pass_log.activation` as torch.Tensor
- No checkpointing to disk during or after forward pass
- `save_new_activations()` re-runs model to refresh activations (expensive)
- No selective/streaming save during forward pass

**Missing**:
- No eager save-to-disk hook
- No streaming activation sink
- No sparse/selective activation save (would require forward-pass hook system)
- No memory-mapped tensor access post-save
- No automatic offload to disk when RAM budget exceeded

**Risk**: Large models with many layers hit memory limits. Users must manually implement activation offloading.

---

### e. Loading / Deserialization

**Status**: ❌ **NOT IMPLEMENTED**

**Current workflow**:
- pickle: `pickle.load(f)` → ModelLog object (weakly supported, see section 3a)
- CSV: No built-in loader; user must create ModelLog from scratch
- Parquet/JSON: Not supported
- **No lazy loading** of activations from disk
- **No round-trip validation** (unpickle → verify = original)

**Missing**:
- No `ModelLog.load(path)` classmethod
- No format detection / auto-parser
- No lazy tensor loading
- No dependency injection (binding activations to new model after unpickle)
- No migration for schema changes (format versioning absent)

---

## 4. RISKS

### Circular References & GC

**Confirmed cycles**:
1. `ModelLog.layer_list[i].source_model_log → ModelLog` (broken by `__getstate__`, requires explicit cleanup)
2. `ModelLog.layer_logs[label].passes[1].source_model_log → ModelLog` (same)
3. `ModelLog._module_logs[addr]._source_model_log_ref → ModelLog` (weakref, should be safe)
4. `ParamLog._param_ref → nn.Parameter → model.parameters()` (pins model, OK for short-term use)

**Mitigation**:
- `cleanup()` method (cleanup.py:42) explicitly breaks all forward refs
- No automatic cleanup; user must call `model_log.cleanup()` to free memory
- Python's cyclic GC handles missed explicit cleanup, but slower

**Risk for I/O**: After unpickling, circular refs may not be properly re-established. Calling `cleanup()` on unpickled ModelLog may delete still-needed objects.

---

### Tensor Device & Grad State on Round-Trip

**Pickle behavior**:
- `activation` tensor device is preserved (GPU/CPU tag stays)
- Gradient tensors are stored as bare clones (no autograd links)
- `gradient` on unpickle is a stale snapshot, not live with model

**Risk**:
- Loading a ModelLog from pickle on a different device (model on GPU, saved on CPU) → `activation` device mismatch
- Trying to run `loss.backward()` using unpickled activations → no gradient flow (tensors are detached)
- No automatic device transfer on load

---

### Memory Footprint on Large Models

**Current strategy**: All activations in RAM
- For ResNet-152 (152 layers) with batch_size=32: ~2-5 GB depending on layer depth
- For Vision Transformers: even larger (many attention heads)
- No tiered memory (RAM → disk → eviction)

**Pickle overhead**:
- `func_rng_states` stores full RNG state per layer (100-200 bytes × 1000 layers = 100-200 KB)
- `captured_args` stores deep clones of input tensors to each function (can be large)
- No compression

**Risk**: Unpickling a large ModelLog may spike RAM usage (deserialize full graph + activations).

---

### Global Toggle State Interactions

**Toggle mechanism** (_state._logging_enabled):
- Single global boolean flag
- Checked in ~2000 wrapped torch functions
- `pause_logging()` context manager used during `activation_postfunc` to prevent logging postfunc ops

**Pickle risk**:
- `_logging_enabled` is not part of ModelLog (lives in module state)
- Unpickling does not restore logging state
- If user unpickles in an active logging session, cross-contamination possible (unlikely but plausible)

**Risk**: LOW (logging state is global and user-managed, not tied to ModelLog serialization).

---

### PyTorch Version Brittleness

**Observed version-sensitive code**:
- `func_rng_states` stores `torch.get_rng_state()`, `torch.cuda.get_rng_state()` (format changes between PyTorch versions)
- `torch.dtype` objects may not unpickle identically across versions
- `func_applied` (function reference) may not pickle if torch functions are refactored

**Risk**:
- ModelLog pickled in PyTorch 2.0 may not unpickle in PyTorch 2.2
- No version tag in pickle file
- No migration path for schema changes

---

### Pickle Security

**Current state**:
- Default pickle protocol used (no explicit `protocol=5` seen)
- `func_applied` stores direct references to torch callables (arbitrary code potential if attacker provides malicious torch module)
- **No signature verification**

**Risk**: Loading untrusted ModelLog pickles is unsafe (attacker can craft a pickle that calls arbitrary code via pickled function references). Best practice: only unpickle from trusted sources.

---

## 5. RECOMMENDATIONS

### 5.1 Format Strategy

**Recommended Approach: Parquet-primary + pickle fallback**

**Rationale**:
1. **Parquet** (columnar, typed, language-neutral, schema evolution):
   - Native support for numpy arrays → list-of-lists conversion
   - Handles 59-column DataFrame cleanly
   - Version-safe schema (metadata in Parquet file)
   - Can add parameter/module/buffer tables as separate files
   - Suitable for data pipelines (dbt, pandas, polars, DuckDB)

2. **Pickle** (backward compat only):
   - Current code has `__getstate__/__setstate__` but untested
   - Fine for temporary caches, not long-term archival
   - Good for resume-training (activations must be regenerated anyway)

3. **Reject JSON**:
   - Poor for tensors (no native uint8, no compression)
   - Verbose for large graphs
   - Use Parquet + CSV instead

4. **Reject HDF5/SafeTensors**:
   - Overkill for metadata-heavy logs (not primarily tensor storage)
   - Parquet is simpler, language-neutral

**Tradeoff**: Parquet requires new dependency (pyarrow); pickle is built-in but unsafe + version-brittle.

**Recommendation**:
- Export: `model_log.to_parquet(path)` (saves metadata table + optional activation tensors in separate .pt files)
- Load: `ModelLog.from_parquet(path)` (reconstructs from metadata, optionally lazy-loads activations)
- Fallback: `pickle.dumps(model_log)` still works, but warn user about instability

---

### 5.2 Streaming-to-Disk Feasibility

**Eager save during forward pass**: ✅ **Feasible, Medium Effort (2-3 sprints)**

**Approach**:
1. Add optional `activation_sink: Callable[[str, torch.Tensor], None]` parameter to `log_forward_pass()`
2. In `log_function_output_tensors()`, after `save_tensor_data()`, call:
   ```python
   if self.source_model_log.activation_sink is not None:
       activation_sink(layer_label, self.activation)
   ```
3. User provides sink (e.g., save to disk, write to shared memory, send to queue)

**Example sink**:
```python
def disk_sink(label: str, tensor: torch.Tensor):
    torch.save(tensor, f"/tmp/activations/{label}.pt")

model_log = log_forward_pass(model, input, activation_sink=disk_sink)
```

**Streaming (selective save)**: ✅ **Also feasible, same infrastructure**
- User passes `layers_to_save=['conv2d_1_1', 'conv2d_2_1']` → only those layers hit the sink
- Reduces memory from O(L) to O(k) where k = # of saved layers

**Effort**:
- Modify `save_tensor_data()`: +10 lines
- Add parameter to ModelLog.__init__: +1 line
- Tests: +3 test cases
- Docs: +1 example notebook
- **Total: ~20-40 hours (1 week, one engineer)**

**Tradeoff**: Eager sink requires user to implement; streaming isn't automatic. Consider lazy-load infrastructure (see below) as complement.

---

### 5.3 Lazy vs Eager Tensor Loading

**Recommended: Hybrid**

**Lazy loading (first-class support)**:
1. On disk: activations stored as separate torch.pt files (one per layer)
2. On unpickle: `ModelLog.from_parquet(path, lazy=True)` → metadata loaded, activations replaced with `_ActivationRef` proxy objects
3. On access: `layer_log.activation` → if proxy, load from disk on-demand, cache in RAM

**Benefits**:
- Users can explore metadata (graph, FLOPs, module structure) without loading activations
- Selective activation load: `layer_log.activation` vs `layer_log.activation_shape` (metadata only)
- Suitable for very large models (VLM, LLM)

**Implementation**:
```python
class _ActivationRef:
    def __init__(self, path: str):
        self.path = path
        self._cached = None

    def __getattr__(self, name):
        if self._cached is None:
            self._cached = torch.load(self.path)
        return getattr(self._cached, name)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        # Transparent in torch operations
        return func(self._get_tensor(), **kwargs) if kwargs else func(self._get_tensor())

    def _get_tensor(self):
        if self._cached is None:
            self._cached = torch.load(self.path)
        return self._cached
```

**Eager (status quo)**:
- `from_parquet(path, lazy=False)` loads all activations into RAM immediately
- Suitable for small-to-medium models

**Tradeoff**: Lazy adds ~200 lines, requires careful transparent tensor handling. Worth it for 1.0 release.

---

### 5.4 Concrete API Proposal

```python
# Export
model_log.to_parquet(
    path: str,
    include_activations: bool = False,
    include_gradients: bool = False,
    compression: str = "snappy",
)
# Saves metadata table(s) to {path}/metadata.parquet
# If include_activations: creates {path}/activations/ with one .pt per layer
# If include_gradients: creates {path}/gradients/ with one .pt per layer

# Load
model_log = ModelLog.from_parquet(
    path: str,
    lazy_activations: bool = True,
    lazy_gradients: bool = True,
    device: str = "cpu",
)
# Reconstructs ModelLog from parquet metadata
# If lazy: activations are _ActivationRef proxies; access triggers disk load + cache
# device: move tensors to specified device on load

# Backward compat (pickle)
model_log = pickle.load(open("model_log.pkl", "rb"))
# Still works, but deprecated. Warn user: "Pickle format is unstable across PyTorch versions. Use to_parquet() for archival."

model_log.to_pickle(path: str)
model_log = ModelLog.from_pickle(path: str)
# Explicit wrappers (clearer intent than raw pickle.dump/load)

# Parameter/Module/Buffer-level export (future, not in 1.0)
model_log.params.to_csv("params.csv")
model_log.modules.to_parquet("modules.parquet")
model_log.buffers.to_csv("buffers.csv")
```

**Per-layer detail export** (also future):
```python
for layer_label, layer_log in model_log.layer_logs.items():
    layer_log.to_dict()  # → {'layer_label': ..., 'activation': ..., ...}

# Or bulk:
model_log.to_dict_list()  # → [{'layer_label': 'conv2d_1_1', ...}, ...]
```

---

### 5.5 Backward Compat & Format Versioning

**Schema versioning**:
1. Add `_io_format_version = 1` to ModelLog
2. On export to parquet: embed version in file metadata (Parquet schema footer)
3. On import: check version, migrate if needed
   ```python
   def _migrate_v0_to_v1(state: Dict) -> Dict:
       # Example: add new fields with defaults
       state['new_field'] = None
       return state
   ```

**Backward compat strategy**:
- Keep `__getstate__/__setstate__` working (pickle support) for ≥ 2 minor versions
- Parquet format is primary going forward
- No breaking changes to pickle unless major version bump
- Document in CHANGELOG

**Migration example**:
```python
@classmethod
def from_parquet(cls, path, ...):
    schema_version = _read_parquet_metadata(path)['io_format_version']
    df = pd.read_parquet(path)
    if schema_version == 0:
        df = _migrate_v0_to_v1(df)
    return cls._from_dataframe(df)
```

---

## 6. IMPLEMENTATION OUTLINE

### Phase 1: Parquet Export (1 sprint, 1 engineer)
**Files touched**: `data_classes/interface.py`, `data_classes/model_log.py`

1. Add `to_parquet(path, include_activations, ...)` method
   - Reuse `to_pandas()` logic
   - Optionally save activations as separate `.pt` files
   - Write schema version to parquet metadata
   - Tests: round-trip to_parquet → read_parquet

**Deliverable**: Users can export metadata to parquet.

---

### Phase 2: Parquet Import + Lazy Loading (1.5 sprints, 1 engineer)
**Files touched**: `data_classes/model_log.py`, `data_classes/interface.py`, new file `data_classes/lazy_activation.py`

1. Create `_ActivationRef` proxy class (lazy_activation.py)
   - `__getattr__`, `__torch_function__` for transparency
   - Caching + device handling
2. Add `from_parquet()` classmethod to ModelLog
   - Deserialize metadata from parquet
   - Reconstruct LayerLog, LayerPassLog, ModuleLog objects
   - Optionally bind activations (eager or lazy)
3. Tests: round-trip export → import, lazy load on access, device transfer

**Deliverable**: Users can import metadata + lazy-load activations.

---

### Phase 3: Streaming Activation Sink (1 sprint, 1 engineer)
**Files touched**: `user_funcs.py`, `data_classes/model_log.py`, `capture/output_tensors.py`

1. Add `activation_sink: Optional[Callable]` parameter to `log_forward_pass()`
2. Pass sink to ModelLog.__init__
3. In `log_function_output_tensors()`, after `save_tensor_data()`:
   ```python
   if self.source_model_log.activation_sink is not None:
       self.source_model_log.activation_sink(
           self.layer_label_no_pass, self.activation
       )
   ```
4. Tests: verify sink called with correct labels/tensors

**Deliverable**: Users can stream activations to disk during forward pass.

---

### Phase 4: Pickle Hardening (0.5 sprint, 1 engineer)
**Files touched**: `data_classes/model_log.py`, `data_classes/layer_pass_log.py`, `data_classes/layer_log.py`

1. Add `_io_format_version` to ModelLog (set to 1)
2. Enhance `__getstate__()` to include version info
3. Add `from_pickle()` classmethod for clarity
4. Add warning on `__setstate__()` if unpickling stale model

**Deliverable**: Pickle still works; users aware it's not primary path.

---

### Phase 5: Per-Level Export Methods (1.5 sprints, 1 engineer)
**Files touched**: `data_classes/param_log.py`, `data_classes/module_log.py`, `data_classes/buffer_log.py`

1. Add `to_dict()` / `to_pandas()` / `to_csv()` to ParamAccessor
2. Add `to_dict()` / `to_pandas()` / `to_parquet()` to ModuleAccessor
3. Add `to_csv()` to BufferAccessor
4. Tests: round-trip export for each

**Deliverable**: Fine-grained export options (parameters, modules, buffers separately).

---

### Phase 6: Documentation + Examples (1 sprint, 1 engineer)
**Files touched**: `docs/io.md`, example notebooks

1. Write guide: "Saving & Loading ModelLogs"
   - Best practices (parquet for archival, pickle for quick debugging)
   - Device handling
   - Memory management (lazy loading for large models)
2. Example notebook: export layer graph to parquet → analyze in Pandas
3. Example notebook: resume training with saved activations (for validation)
4. Deprecation notice: pickle format instability

**Deliverable**: Users understand all export options and tradeoffs.

---

## Summary Table

| Feature | Status | Recommendation | Effort |
|---------|--------|-----------------|--------|
| Pickle round-trip | ✅ Partial | Harden + deprecation warning | 0.5 sprint |
| Pandas export | ✅ Exists | No change (stays) | — |
| Parquet export | ❌ Missing | Add in Phase 1 | 1 sprint |
| Parquet import | ❌ Missing | Add in Phase 2 | 1.5 sprints |
| Lazy loading | ❌ Missing | Add in Phase 2 | 1.5 sprints |
| Streaming sink | ❌ Missing | Add in Phase 3 | 1 sprint |
| Per-level export | ❌ Missing | Add in Phase 5 | 1.5 sprints |
| Format versioning | ❌ Missing | Add in Phase 4 | 0.5 sprint |
| **Total** | | | **7.5 sprints** (2.5 months, 1 FTE) |

---

## Conclusion

TorchLens has **no production I/O system**. Pickle hooks exist but are untested and unstable. The single export pathway is `to_pandas().to_csv()`, which covers only metadata. Activations are RAM-resident only.

**Recommended path forward**:
1. **Parquet as primary format** (safer, version-stable, interoperable)
2. **Lazy loading for large models** (metadata exploration without RAM spike)
3. **Streaming sink for eager disk save** (opt-in, user-driven)
4. **Deprecate pickle** except for debugging (too brittle for release code)
5. **Per-level export** (parameters, modules, buffers) for fine-grained analysis

**This is a **2.5-month effort** for a full solution (6–7 engineer-sprints). Prioritize Phases 1–2 (parquet import/export + lazy loading) as MVP, then add streaming and per-level exports as follow-ups.**
