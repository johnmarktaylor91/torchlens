## Key Concepts

### Toggle Architecture
- Lazy wrapping: `wrap_torch()` installs wrappers on first capture or explicit call.
- Persistent wrappers: after wrapping, calls only pay a `_state._logging_enabled` check when
  logging is off.
- `active_logging(trace)` enables logging during the forward; `pause_logging()` protects
  internal TorchLens tensor ops from recursive capture.
- Stale `from torch import cos` style references are recovered by the rescue re-run
  (`backends/torch/rescue.py`, disclosed via `trace.rescue_rerun`); the protocol-invisible
  constructors keep targeted module-attr patching via the mechanical belt (`backends/torch/belt.py`).

### Module Containment
Module containment is captured via a wrap-forward stack helper at
`backends/torch/module_stack.py`. Both fastlog and exhaustive modes share the helper. Each
captured op snapshots the stack at op-creation time; downstream postprocess only appends
the canonical module-path suffix to `equivalence_class` for loop detection. This replaces
the older tensor-entry/exit thread-replay system removed in v2.18 (sprint
module-containment-refactor).

### Data Flow
1. Decoration intercepts torch function calls.
2. Barcode nesting detection identifies bottom-level operations.
3. `capture/` builds raw `Op` entries.
4. `postprocess/` removes orphans, marks conditionals, detects loops, labels nodes, builds logs.
5. `Trace` exposes lookup, visualization, validation, save/load, intervention, and summary helpers.

### Journal Producer (single since producer unification P7)
Torch captures journal decomposed `OpRecord` rows (`ir/op_record.py`: `OpCore` + typed
facets, strict protocol with legacy flat-name properties) through the ONE commit tail
`capture/projections.py::commit_op`; step 0 ingests them via the generated scatter
(`ingest_op_records`). The op lane is genuinely append-only: post-commit knowledge rides
the typed `OpAmendment` lane (nine exact-set families, `append_amendment` the single
writer) and every amended-state read folds through `CaptureEvents.amended_op_records()`.
ONE sanctioned in-place carve-out exists: the fastlog ancestry-closure backfill
(`fastlog/types.py::_backfill_cooked_ancestry`) replaces op-lane cells on the
`copy_for_replay` projection a cook owns — never the sealed Recording stream — and nulls
`_amended_fold_cache` afterwards. Every in-place op-lane writer (this one included) is
pinned by the closed reason-bearing ledger in
`tests/producer_parity/test_op_lane_inplace_writers.py`, whose package-wide AST scan sees
subscript/slice/augmented writes, `del`, and mutating list methods; a new writer is red
until consciously ledgered.
grad-fn handles live only in the journal side index (`grad_fn_handles_by_label_raw`).
The legacy `OpEvent` torch producer and its dual-path env switch were
deleted (P7); preview backends keep emitting compat `OpEvent`s until S15 and adapt at the
one ingest boundary (`op_record_from_event`), with `PATH_TO_FLAT` as the amendment fold
guard and `_clone_op_event_for_replay` record-shape-aware, all retained-with-schedule.

### Portable Artifacts
`tl.save()` and `tl.load()` route through `_io/bundle.py`. Unified `.tlspec` directories have
`manifest.json` plus safetensors blobs; public schema validation lives in `validation/__init__.py`.
Runnable saves are sparse by default and produce `sparse_recorded_taken_path_v2` descriptors with
REQUIRED explicit execution-context records (per-call `CallExecutionContext` + capture-scoped
`AmbientExecutionContext`), restored at replay or refused typed; legacy v1 artifacts load
analysis-only. `include_weights=True` bundles the full capture-time
`state_dict` (named parameters plus persistent buffers) as a separate `state_dict_v1` blob family;
the sparse core still contains no tensor values. Load binds it through the same strict state
contract used by `Trace.load_state_dict()`, while explicit user state overrides it at run time.
Used non-persistent buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family
(declared state; not gated on either include flag; disclosed at save).
`include_activations=True` independently writes capture-time `save=`-selected
`out`/`transformed_out` values as `selected_activation_v2`, including physical
`InputAttestationFingerprint` eligibility records. Loaded values are available through
`Trace.archived_activations` for inspection and eligible byte-exact attestation only; the sparse
scheduler never consumes them. Original-input, capture-equivalent real-state runs report
`attested` or fail transactionally with `numeric_attestation_failed`; changed-input (logical or
physical), random-state, and nondeterministic-capture-context runs report `not_applicable`, and
`attested` always implies `verified`.
The frozen `ReadinessStatus`, `RunProvider`, `StateSource`, `PathFaithfulness`, `DivergencePolicy`,
`NumericAttestationStatus`, and `RunnableErrorCode` vocabularies live in `torchlens.runnable` and are
documented exhaustively in `docs/reference/runnable_tlspec_contract.md`. r37 additions:
`state_alias_topology_unsupported` (save-time state-topology refusal; tied live-identity state
stages as one alias-group allocation) and `context_field_invalid` (parse-time closed-vocabulary
context refusal); zero-tensor-leaf and instance-stateful container outputs refuse at save with
`missing_output_container_contract` via the one per-kind capability table.
Non-torch preview backends use `payload_policy="array_payloads"` when their codecs can materialize
payloads; Paddle bf16 payloads carry logical dtype metadata because NumPy transports them as
`uint16`. TensorFlow preview payloads also use `array_payloads` for dense numeric/bool forward
arrays and preserve `tf.bfloat16` logical dtype metadata.
Intervention specs can be saved at audit, executable-with-callables, or portable levels.

### Appliances
The appliance subfolders `notebook` and `neuro` are part of the 2.x package layout. Their
extras enforcement is DEFERRED, never import-time: `import torchlens.notebook` /
`torchlens.neuro` stays inert by design, and the dependency check fires on first
attribute access via PEP-562 `__getattr__` — precisely so a bare import can never run
foreign code without a trust opt-in (RCE-hardened, gated by
`tests/test_r9_appliance_import_rce.py`). Never "fix" them to import their
dependencies at module load; that is the pattern the code forbids.
