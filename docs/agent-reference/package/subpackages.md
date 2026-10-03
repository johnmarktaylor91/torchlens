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
  `internal_source_ancestors`) — an authorized public type break (decided 2026-08-12):
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
  artifact (routes 2 of the 116 `__all__` names; own AGENTS.md).
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
