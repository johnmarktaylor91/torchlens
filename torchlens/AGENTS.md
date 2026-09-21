# torchlens architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

## What This Is

TorchLens extracts outs and metadata from backend-resolved captures. PyTorch eager capture is the
stable default; MLX, JAX, tinygrad, Paddle, and TensorFlow are technical-preview backends. `import torchlens`
exposes the public API and compatibility shims, but torch wrapping is lazy: the first torch capture
prepares the model and calls `wrap_torch()` from `backends/torch/`.

## Architecture Overview

```
import torchlens
  |- exposes 119 top-level public names in __all__
  |- eagerly imports ONLY the light spine: options, errors/_state, ir.*,
  |  captured_run, observers, quantities, _deprecations, _errors, _io,
  |  _literals, _save_budget, utils,
  |  and visualization (hard-pinned by tests/test_import_hygiene.py
  |  _EAGER_TORCHLENS_MODULES). capture, intervention, fastlog, autoroute,
  |  bridge, compat, export, report, stats, validation, and viz-rendering
  |  internals are ALL lazy (_LAZY_ATTRS) — do not add eager imports
  |
trace(model, input, save=..., intervene=..., lookback=..., storage=...)
  |- backends/registry.py      - resolve torch / MLX / JAX / tinygrad / Paddle / TensorFlow backend
  |- backends/torch/model_prep.py - ensure torch is wrapped, prepare modules/buffers/params
  |- capture/trace.py          - run forward pass with active logging
  |- backends/torch/ops.py     - build raw torch op records during wrapper calls
  |- postprocess/              - current 26-step graph cleanup/finalization pipeline
                                 (contract keys 0..20 + 5 fractional inserts)
  +- returns Trace

tl.record(model, input, save=...)
  |- uses the same wrapper hot path
  |- stores predicate-selected RecordContext/ActivationRecord values
  +- returns Recording; Recording.to_trace() materializes full structure
```

Selective `layers_to_save` uses a predicate-backed single pass when early labels are
sufficient and falls back to the two-pass strategy for final-numbering selectors:
negative indexes, integer ordinals, indexed label strings (`relu_1_2`, `relu_1` — any
`_<digit>` component; orphan removal renumbers ordinals AND type indexes after capture),
output labels, identity labels, and gradient selection (integer `save_grads` ordinals
included — deferred grad hooks install post-postprocess from the reference escrow, never
from raw-index prediction). A mixed selection with a negative tail disables the escrow
eviction window so early final-numbering components keep their payloads. String
selectors keep the legacy substring contract. Unqualified recurrent
labels save all passes; pass-qualified labels such as `"attn:2"` save one 1-based pass.
Prefer `save=tl.func(...)`, `save=tl.in_module(...)`, and composed predicates for new
single-pass selective capture. The old `keep_op=`/`keep_module=` `record()` alias
kwargs are removed; `save=` is the only predicate spelling and `default_module=`
gates module-boundary event recording (uniformly — ALL module enter/exit events;
predicate-gated module-event selection has no public spelling).

Common unified capture examples:

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
paddle_trace = tl.trace(paddle_model, paddle_x, backend="paddle")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("attn"),
    intervene=tl.when(tl.func("attn"), tl.scale(0.5)),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
trace.draw(show_containers="nodes")
```

Provisional semantic I/O surface (review-day names):

```python
log = tl.trace(model, x, output_style="classification", output_head="logits")
log.output_table(top_n=5)
log.summary(level="output")
log.to_pandas(include_decoded_output_summary=True)

input_log = tl.trace(model, raw_text, transform=text_to_tensor, save_raw_input="small")
input_log.draw(show_input_transform_summary=True)

mds_layers = tl.in_module("block1") | tl.in_module("block2")
image_log = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)
image_log.model_profile
image_log.output_table(top_n=5)
tl.repgeom.mds_evolution(image_log, save=mds_layers, min_n=8)
tl.repgeom.rdm_evolution(image_log, save=mds_layers)
tl.viz.feature_map_evolution(image_log, save=mds_layers)
tl.repgeom.scree_evolution(image_log, save=mds_layers)
image_log.draw(node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8))
image_log.draw(node_spec_fn=tl.repgeom.rdm_node_spec(max_stimuli=8))
image_log.draw(node_spec_fn=tl.viz.feature_map_node_spec())
image_log.draw(node_spec_fn=tl.repgeom.scree_node_spec())
```

Sprint B annotation/MDS names are provisional until review-day signoff. `Trace.model_profile`
is computed, not persisted. `tl.repgeom.mds_evolution(...)` requires the target batch
activations to have been saved at capture time; use a curated `save=` subset, not exhaustive
`layers_to_save="all"`,
for image batches. `Trace._annotation_blobs` is public-provisional only for render-time
annotation payloads and compatibility review.
Sprint C RDM, feature-map, and scree node visuals are PIL-only render-time images composed
from `tl.viz.render_*` primitives and are provisional until review-day signoff.

`backward_ready=True` is the public opt-in for losses built from saved outs. It keeps
floating tensors graph-connected, preserves user `requires_grad`, and rejects incompatible
detaching or disk-only out storage.
`inference_only=True` is the opt-in no-grad capture path for forward-only analysis; it is mutually
exclusive with backward-related capture because it discards the autograd graph.

## Top-Level Modules

| Path | Purpose |
|------|---------|
| `__init__.py` | Top-level API, 119-name `__all__`, deprecation shims, `peek`/`extract` helpers |
| `_state.py` | Global logging toggle, active log, decoration maps, prepared-model registry; no torchlens imports except the sanctioned `errors._base` leaf (a RUNTIME base-class import, cycle-safe; only its TYPE_CHECKING block is typing-only) |
| `_trace_state.py` | Small runtime state enum exposed through `torchlens.io` |
| `_errors.py`, `errors/` | Public and legacy exception classes |
| `_io/`, `io/` | Portable `.tlspec` save/load, manifest, lazy tensor refs, public I/O helpers |
| `options.py` | Capture, save, visualization, replay, intervention, and streaming option groups |
| `observers.py` | `tap()` and `span()` observer helpers (`record_span` is a deprecated warning alias) |
| `report/` | `report.explain(log)` and capture-time scalar logging |
| `stats/` | Streaming stats and `aggregate()` over dataloaders |
| `types.py`, `accessors/` | Moved type/accessor aliases for non-top-level public names |

## Subpackages

- `capture/` - real-time forward and backward operation logging.
- `data_classes/` - `Trace`, `Layer`, `Op`, module/param/buffer/grad logs. The declared
  record schema carries per-field `StorageBinding` axes (generated `_schema_bindings.py`,
  regenerate with `tools/generate_record_schema.py`); Trace fields have a declared
  component ownership map (`_trace_components.py`).
- `_trace_core/` - private columnar store substrate (columns, pools, edge-occurrence
  table, payload arena, overlays, `TraceCore`). The M5 Op seam is LIVE: every captured
  `Op` is a two-word `(_core, _row)` facade over the per-trace `OpRowStore`
  (`op_store.py`) held at `trace._trace_core` (declared `FieldPolicy.DROP`); rows are
  row-major lists while building and seal after the final step, key 20 (columnar transpose with
  numeric packing at >=512 rows). `Op.copy()`, pickle restore, fork shells, and
  preview backends use detached single-row stores. M6 relations are LIVE
  (`relation_views.py`): on FINISHED traces the relation accessors return IMMUTABLE
  views — `tuple` for label sequences (`parents`, `children`, `modules`,
  `module_call_stack`, conditional child lists, ...), `frozenset` for label sets
  (`input_ancestors`, `output_descendants`, `root_ancestors`,
  `internal_source_ancestors`) — an authorized public type break (JMT 2026-08-12):
  in-place mutation raises, assignment still works and normalizes to the view type,
  and equal views may be shared across records. `parents`/`children` live in the
  core's canonical dataflow edge-occurrence table (CSR by edge id) and rematerialize
  lazily; dict-shaped relation metadata (`parent_arg_positions` etc.) stays mutable.
  M7a group views are LIVE: `equivalent_ops`/`recurrent_ops` cells hold ONE shared
  `GroupRef` per membership group (`groups.py`); reads resolve to the group's cached
  immutable view (`frozenset`/`tuple`) and removal scrub rebinds the group row once.
  M7b shared-fact blocks (`fact_blocks.py`): the call-level container facts
  (`code_context`, `non_tensor_pos_args`, `non_tensor_kwargs`, `func_non_tensor_args`,
  `func_config`, `arg_names`) live ONCE per FunctionCall group and `param_shapes` once
  per distinct value (the ParamAlias block); member cells hold the `_FACT` sentinel and
  the facade hydrates the exact public container type per row on first read (fresh
  mutable copy, cached back — per-row isolation is unchanged). Sibling outputs of one
  wrapped call also share ONE journal-side `FunctionCallRef`.
  During postprocess the staging containers remain real mutable builtins; legacy
  list/set state normalizes on load.
  M8 remaining kinds: `Layer` is an AGGREGATE FACADE — the ~86 representative
  fields the dict era copied from the first pass are class-level mirror
  descriptors reading through to `ops[0]` (with the exact copy-time
  normalizations); writes shadow per-layer in `__dict__`, deletes tombstone,
  and cleanup/removal materialize mirrors before husking the backing ops.
  `Param`/`Buffer`/`FuncCallLocation`/`ModuleCall`/`Module` are row facades
  over per-trace kind tables (`record_rows.py`, `TraceCore.kind_rows`,
  adopted by the build passes and sealed with the core; preview backends and
  loads stay detached-backed). The canonical label -> op-row index binds at
  the freeze as `TraceCore.label_rows`.
  M9 backward epochs: `GradFn`/`GradFnCall`/`BackwardPass` are row facades;
  each successful backward projection binds an atomic `BackwardEpoch`
  (`TraceCore.backward_epochs`) — a full rebuild atomically replaces the
  epoch list, a clean tail fold extends the live epoch — whose per-kind row
  stores back the projected records. The lazy watermark/revision
  invalidation stays trace-side and byte-identical; loaded/preview traces
  keep detached-backed backward records.
  M10: `TraceBuildState` is GONE — its transient fields dissolved into three
  named per-phase workspaces (`ir/workspaces.py`: `RawGraphWorkspace` for
  capture ingress + steps 0-11, `ModuleCaptureWorkspace` for module
  prep/stack capture consumed at step 16, `WrapperRuntimeWorkspace` for the
  wrapper hot path), each dropped at the transient-state cleanup seam; the
  backend `finalize_forward_session` protocol takes the raw-graph workspace
  as its ownership token. Each `POSTPROCESS_STEP_CONTRACTS` entry (v2)
  declares its exact op-store COLUMN write AND read sets plus
  `placeholder_probes`, `row_effects` (creates/deletes row sanctions), and
  closed-vocabulary `trace_state` tokens; the step order is DERIVED from
  these declarations by rank-keyed Kahn (`postprocess/_executor.py`), with
  the frozen `LEGACY_STEP_RANK` and the reason-bearing `PINNED_ORDER_PAIRS`
  corpus as the two-key direction authority — a coordinated rank+registry
  reversal slips the drift checks by construction and only the corpus
  catches it. `TORCHLENS_POSTPROCESS_ASSERTIONS` arms a zero-cost-when-off
  write audit and `TORCHLENS_POSTPROCESS_READ_AUDIT=enforce` the read side
  (class-swap instrumentation in `op_store.py`), scoped by per-step
  begin/run/end/assert windows (postconditions run outside any window; no
  window survives the loop). Declared-never-observed writes and reads live
  in reason-bearing phantom-exemption ledgers, no-op writers are pinned and
  cannot discharge a read, and day-1 findings are pinned by name; widening
  any set is a reviewed contract diff.
  M11: `Trace.fork()` is COPY-ON-WRITE (`data_classes/_trace_fork.py`): the
  fork core wraps the sealed op store and every kind table in per-fork
  `OpStoreView`s (own overlay; base overlay/rows snapshot at fork;
  eager fork-time isolation of exact builtin mutable containers with
  tensor/callable identity preserved; GroupRef translation to cloned group
  tables; record/accessor translation for cell-held references), fork
  records are fresh two-word shells at the SAME rows, and only Layer shadow
  dicts, record extras, and the policy-driven trace-side remainder are
  copied. The object-graph forkcopier (typed deepcopy engine) is deleted;
  differentiable replay forks drop the deep-cone/shallow-rest split. The
  facade identity cache is weak-valued (strong side table only for
  non-weakref-able `Op`, whose weakref refusal aliases-v1 pins); the
  standalone compaction passes are folded into the core freeze seam
  (`data_classes/_compaction.py`); `TraceCore.transaction()` checkpoints
  every mutation surface atomically. Architecture of record:
  `docs/reference/trace_core_design.md`.
- `backends/torch/` - torch function wrapping, explicit wrap/unwrap, module prep.
- `fastlog/` - sparse predicate recording with RAM/disk storage and recovery.
- `postprocess/` - graph cleanup, conditionals, loop detection, labeling, finalization.
- `validation/` - forward replay, backward validation, metadata invariants, `.tlspec` schema checks.
- `visualization/` - Graphviz rendering, rank layout, NodeSpec, themes, overlays, bundle diff.
- `intervention/` - selectors, sites, hooks, helpers, Bundle, fork/replay/rerun/save.
- `intervention/_super/` - internal Bundle-level Super* aligned views and accessors.
- `intervention/_topology/` - internal bundle supergraph and topology diff support.
- `merged/` - cross-rank merging (C1): `tl.merge_ranks`/`tl.merge_report`, the
  `MergedTrace` presenter, frozen merge vocabularies, and the merged-directory
  artifact (routes 2 of the 119 `__all__` names; own AGENTS.md).
- `distributed/` - explicit-collective capture support: `tl.distributed.arm()`,
  group-lifecycle ledger, membership-lineage audit (own AGENTS.md).
- `bundle/` - the intervention `Bundle` product and its aligned Super* views.
- `ir/` - eagerly-imported capture-event/record IR shared by every backend.
- `autoroute/` - HF/entry-point auto-routing (lazy).
- `attribution/`, `receptive_field/`, `repgeom/` - influence geometry and
  representation analysis surfaces (lazy power-user submodules).
- `export/`, `report/`, `stats/`, `debug/` - export bridges, human reports,
  summary stats, power-user diagnostics (lazy; `debug` deliberately not in
  `__all__`).
- `partial/` - failed-capture recovery (`tl.partial.from_failed_capture`).
- `accessors/`, `semantic/`, `observers`, `io/` - accessor protocols, facet
  recipes, public observers, and the `torchlens.io` save/load facade.
- `bridge/`, `compat/`, `callbacks/` - optional integrations and migration facades.
- `notebook/`, `neuro/` - appliance package boundaries gated by extras.
- `examples/`, `experimental/`, `schemas/` - packaged examples, incubating
  surfaces, and the shipped tlspec manifest schemas.

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

## Files in This Directory

KEY files only, NOT exhaustive (the package holds ~48 top-level modules; notable
omissions include `runnable.py` — home of the 7 frozen runnable enums cited
below — `captured_run.py`, `hash.py`, `facets.py`, `_capture_fingerprint.py`,
and the 16-file `_runnable_*` execution seam). `ls torchlens/*.py` is the
authority.

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports, moved-name deprecation shims, `peek`, `extract`, `batched_extract`, validation aliases |
| `_state.py` | Global toggle, active log, decoration maps, prepared model registry; no torchlens imports except the sanctioned `errors._base` leaf |
| `_trace_state.py` | Runtime state enum surfaced through `torchlens.io` |
| `_errors.py`, `_robustness.py`, `_training_validation.py` | Legacy/public error and compatibility helpers |
| `_literals.py` | Shared literal types for options and modes |
| `_source_links.py` | Source-link helpers used by reports/visualization |
| `constants.py` | FIELD_ORDER tuples and decorated torch function discovery |
| `options.py` | Immutable grouped options and flat-argument merge helpers |
| `observers.py` | `tap`, `span` (canonical; `record_span` is a deprecated warning alias), and active span state |
| `types.py` | Moved public type aliases not kept in top-level `__all__` |
| `user_funcs.py` | Main capture, summary, visualization, validation, and bundle graph entry points |

## Attribute Conventions

- TorchLens metadata attached to user/model objects lives under `obj._tl`.
- Permanent module metadata uses `_tl.address` and `_tl.module_type`.
- Session tensor/parameter metadata is cleaned per capture; callable wrapper markers also live
  under `_tl`.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state.

## Public Surface

`torchlens.__all__` is intentionally small and currently has 119 names. New user-facing
objects should usually live under submodules (`torchlens.io`, `torchlens.options`,
`torchlens.bridge`, `torchlens.errors`, etc.) with moved-name shims only when compatibility
requires them.

Unified capture examples:

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
paddle_trace = tl.trace(paddle_model, paddle_x, backend="paddle")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("linear"),
    intervene=tl.when(tl.func("linear"), tl.scale(0.5)),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
live_result = relu_trace.run(inputs=x, seed=42)
loaded_result = tl.load("architecture.tlspec").run(inputs=x, seed=42)
```

The unified `inputs=` run surface returns `RunResult(output, trace, report)` without mutating its
source. Live traces use the existing fast refresh projector; loaded runnable traces execute the
resolved sparse DAG with staged, embedded capture, or N1-a state. Analysis-only loads cannot run.
Runnable descriptors are `sparse_recorded_taken_path_v2`: every call carries a REQUIRED explicit
`CallExecutionContext` and the descriptor one `AmbientExecutionContext`, both restored at replay or
refused typed; legacy v1 artifacts load analysis-only with a typed readiness refusal.
Use `tl.save(trace, path, level="runnable", include_weights=True)` to opt into the full capture-time
`state_dict` (named parameters plus persistent buffers). It is a separate `state_dict_v1` blob
family, not part of the tensor-value-free sparse core or a reconstructed model. Used non-persistent
buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family (declared state, not
gated on either include flag; disclosed at save).
Use `include_activations=True` independently to archive exactly the existing capture-time `save=`
selection as `selected_activation_v2` (with physical `InputAttestationFingerprint` eligibility
records). Inspect it through `Trace.archived_activations`; never use those blobs as DAG inputs.
Eligible original-input/capture-equivalent-state runs byte-attest raw saved slots (`attested` or
transactional `numeric_attestation_failed`), while changed-input (logical or physical),
random/non-equivalent-state, and nondeterministic-capture-context runs are `not_applicable`;
`attested` always implies `verified`.
`trace.run(inputs=..., seed=..., on_divergence="raise")` reports readiness, state source,
`verified|diverged|unverifiable` path faithfulness, and numeric attestation in its `RunResult`.
Use `return_diverged` only when a permanently poisoned diagnostic result is intended. Match failures
through `RunnableErrorCode`; the complete frozen taxonomy is in
`docs/reference/runnable_tlspec_contract.md`. r37: overlapping/unprovable distinct-object state
alias topology and zero-tensor-leaf or instance-stateful container outputs refuse at save
(`state_alias_topology_unsupported` / `missing_output_container_contract`); tied live-identity
state stages as one alias-group allocation; persisted context values validate at parse
(`context_field_invalid`); non-global host RNG/entropy/clock touches permanently ceiling replay
(r39: numpy instances via a chained `sys`/`threading.setprofile` classifier + a cheap
model-attribute state digest -- NO process-wide gc scan; unseeded-construction `randbits` entropy;
`datetime`/`localtime` clocks; an externally-held generator on a pre-existing non-hooked thread is
a documented residual, and a benign background thread never ceilings a capture); tensor->host VALUE escapes are caught by dual observer routes (aten
census + a mode-independent method/predicate belt for `_disable_current_modes` regions, plus the
`__repr__`/`__str__` print interception); loaded-sparse and live providers settle through one
finalizer (a live opaque output is `unverifiable`+poisoned, a parse-refused descriptor degrades
every payload family analysis-only, an inexecutable divergent input raises `PathDivergenceError`);
structseq trust keys on the resolution authority, never `__module__`; CUDA state stages lazily at
run preparation behind a no-allocation readiness capability gate.

Provisional semantic I/O examples (review-day names):

```python
classifier_trace = tl.trace(model, x, output_style="classification", output_head="logits")
classifier_trace.output_table(top_n=5)
classifier_trace.summary(level="output")

input_trace = tl.trace(model, raw_text, transform=text_to_tensor, save_raw_input="small")
input_trace.draw(show_input_transform_summary=True)

mds_layers = tl.in_module("block1") | tl.in_module("block2")
image_trace = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)
image_trace.model_profile
image_trace.output_table(top_n=5)
tl.repgeom.mds_evolution(image_trace, save=mds_layers, min_n=8)
tl.repgeom.rdm_evolution(image_trace, save=mds_layers)
tl.viz.feature_map_evolution(image_trace, save=mds_layers)
tl.repgeom.scree_evolution(image_trace, save=mds_layers)
image_trace.draw(node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8))
image_trace.draw(node_spec_fn=tl.repgeom.rdm_node_spec(max_stimuli=8))
image_trace.draw(node_spec_fn=tl.viz.feature_map_node_spec())
image_trace.draw(node_spec_fn=tl.repgeom.scree_node_spec())
```

Sprint B annotation/MDS names are provisional until review-day signoff. `Trace.model_profile`
is computed, not persisted. `tl.repgeom.mds_evolution(...)` requires the target batch
activations to have been saved at capture time; use a curated `save=` subset, not exhaustive
`layers_to_save="all"`,
for image batches. `Trace._annotation_blobs` is public-provisional only for render-time
annotation payloads and compatibility review.
Sprint C RDM, feature-map, and scree node visuals are PIL-only render-time images composed
from `tl.viz.render_*` primitives and are provisional until review-day signoff.

`record(keep_op=...)` and `record(keep_module=...)` are removed and raise `TypeError`.
`record(save=...)` is the only selective-capture spelling. `layers_to_save=[...]` still exists
as the deprecated flat alias for final-label selection; it is NOT two-pass-only —
`_trace_selector_helpers.py` builds a live single-pass predicate whenever early labels
suffice, falling back to two-pass resolution otherwise. An
unqualified recurrent layer label saves all passes, while `"label:2"` saves only pass 2.

Current 2.x backend surface: torch eager is the stable default; MLX, JAX, tinygrad, Paddle, and
TensorFlow are technical previews behind `BackendSpec`. Paddle M3 is dygraph/eager only, uses
`tl.backends.paddle.GradOptions` for derived-gradient previews, materializes `.tlspec` array
payloads through the Paddle codec, and does not provide true backward capture.
TensorFlow preview targets Keras 3 on TF>=2.16 with
`keras.backend.backend() == "tensorflow"`; its shipped primary path is eager live capture via
`op_callbacks` with real values/control flow/op-level records/module stacks. The graph-only
FuncGraph static path is implemented for compiled/SavedModel entries (opaque regions stay
honestly unverified). Static-label `intervene=` SHIPS for eager entries (two-level writable
layer, fail-closed site reachability), and T1 derived gradients SHIP for eager entries via
`tl.backends.tf.GradOptions` (graph-only captures refuse `grad_options` typed; `intervene=`
cannot combine with `grad_options=`). Deferred: `halt=`/`recipes=`, true backward capture,
and value-dependent predicates.

## Constants as Ordering Spec

FIELD_ORDER tuples define canonical serialized and display field sets. When adding a field,
update the class definition, the appropriate FIELD_ORDER constant, metadata tests, and any
`to_pandas()`/summary surface that should expose it.

## Critical Invariants

1. `_state.py` has no outgoing torchlens imports except the sanctioned `errors._base`
   leaf — a RUNTIME import (its classes are base classes, e.g. `ReentrantTraceError`);
   only the TYPE_CHECKING block below it is typing-only (cycle-safe by construction;
   documented in `_state.py`).
2. `_ensure_model_prepared()` is the lazy wrapping chokepoint; do not reintroduce import-time
   torch namespace mutation.
3. RNG state capture/restore must happen before `active_logging()`.
4. Internal torch ops during capture must be wrapped in `pause_logging()`.
5. Module suffixes are appended to `equivalence_class` at op creation before loop detection.
6. There is no `postprocess_fast()` orchestrator; refresh captures run the full `postprocess()`
   entry point (see `postprocess/AGENTS.md`, "Refresh Projection").
7. `backward_ready=True` must preserve user `requires_grad` and reject detach/disk conflicts.
8. Portable I/O must reject unsafe paths/symlinks and unsupported tensor variants.

## Newer 2.x Subsystems

- `_trace_core/`: private columnar store substrate (typed columns, per-trace intern
  pools, canonical edge-occurrence table + CSR, identity-preserving payload arena,
  sparse overlays, `TraceCore` with COW fork). The M5 Op seam is LIVE: `Op` is a
  `(_core, _row)` facade with one generated data descriptor per stored field; captured
  ops share the per-trace `OpRowStore` (`trace._trace_core`, `FieldPolicy.DROP`,
  sealed after the final postprocess step, key `20`), while copy/pickle/fork/preview paths use detached
  single-row stores. Architecture of record in
  `docs/reference/trace_core_design.md`. The declared
  record schema carries `StorageBinding` axes (`data_classes/_schema_bindings.py`,
  regenerated by `tools/generate_record_schema.py`), and Trace fields have a declared
  component ownership map (`data_classes/_trace_components.py`).
- `_io/` and `io/`: portable save/load, `.tlspec` manifests, lazy out refs, rehydration.
- `intervention/`: Bundle, sites/selectors, hooks, helpers, replay/rerun/fork/save.
- `fastlog/`: sparse `Recording` path, predicate normalization, RAM/disk storage.
- `bridge/`: optional adapters for external tools; bridge and autoroute surfaces stay lazy, as pinned by tests/test_import_hygiene.py.
- `compat/`: migration helpers and `compat.report(model, x)`.
- `callbacks/`: Lightning callback integration.
- `partial/`: partial log wrapper for failed captures.
- `debug/`, `report/`, `stats/`, `viz/`, `experimental/`: diagnostics, explanation,
  aggregation, convenience visuals, and unstable APIs.

## Package Layout Policy

- `io` is the public portable I/O facade; `_io` owns bundle, manifest, codec, and lazy-load internals.
- `errors` is the public exception facade; `_errors.py` is legacy internal exception plumbing to fold in later.
- `debug` is the public diagnostics toolbox; private debug helpers should stay local to their owning modules.
- `visualization` owns graph rendering and layout; `viz` owns image and plot primitives used by renderers.

## Conditional Branch Attribution

- Step 5 builds AST file indexes, classifies terminal bools, materializes dense
  `conditional_records`, runs backward flood, attributes forward arm edges, then derives
  legacy THEN/ELIF/ELSE views.
- Primary structures are `Trace.conditional_records`, `conditional_arm_entry_edges`,
  `conditional_edge_call_indices`, and `conditional_arm_children`.
- Graphviz renders IF/THEN/ELIF/ELSE labels; dagua conditional support remains more
  limited than Graphviz.

## Release Safety

Semantic-release uses `scripts/no_major_parser.py` plus commit hooks to block accidental
major bumps. For docs-only work use `docs(...)` or `chore(...)` and never add major-bump
markers to commit messages, PR text, or committed docs.
