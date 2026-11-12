# backends/tf/ - Implementation Guide

TensorFlow backend. It fills the backend protocol (`maturity="preview"`, no `capture_backend`); its
`BackendSpec` in `../default_specs.py` and its protocol shell in
`../_protocols/tf.py` are the capability truth (contract: `../AGENTS.md`).

## backend.py
- `TFBackend.capture_trace()` is the standalone entry. Eager `op_callbacks` live
  capture is the shipped PRIMARY path; the static FuncGraph path handles
  compiled/SavedModel entries.
- `intervene=` or `stop_after=` combined with `grad_options=` refuses typed (the
  derived-gradient replay reruns the un-edited full forward). Both surfaces require
  eager live capture; the static path refuses each (`refuse_static_halt`).
- `stop_after=` halts live: `interventions.wire_eager_session()` builds the
  `_halt.TFHaltRuntime`, which raises `StopSignal` from the op callback after the
  frontier op is recorded (latch first, torch `evaluate_halt` parity); the session
  catches it and `finish_halt()` writes the stopped fields and settles an unfired
  site (selector refuses, callable warns). `facets=` attaches per-trace
  facet recipes on both the eager and the static capture path through the
  shared `backends/_facet_recipes.py` helper (the `facet_recipes` record).
- `_finish_trace(recurrence_detection=...)` runs the shared neutral grouper via
  `.._finalize.finalize_single_pass_trace`; the eager path passes the user's
  request, the static FuncGraph path stays ungrouped and stores the honest
  EFFECTIVE value `False`.

## op_callback_capture.py
- `TFEagerCaptureSession.run()` drives eager capture through TF `op_callbacks`:
  real values, taken-branch control flow, op-level `TFOpCapture` records.
  `output_hook` sees each callback's recorded `(label_raw, RecordContext, tensor)`
  batch (edit sites, then the halt); `code_context_fn` (a `_source_context.TFSourceContext`)
  stamps user frames, filtering TensorFlow, Keras and torchlens frames by module name.
- `warm_up_tf_callable()` pre-runs the callable outside the callback window.

## funcgraph.py
- `capture_static_funcgraph()` is the graph-only static path; opaque regions are
  emitted honestly unverified (`_capture_opaque_graph_region`).

## derived_grads.py
- `GradOptions` (re-exported from `__init__.py`) + `attach_tf_derived_grads()`:
  one GradientTape replay attaching leaf and exact-T1 intermediate gradients,
  with divergence refusal (`_tf_trees_close`).

## interventions.py
- `intervene=` for eager entries. `normalize_tf_interventions()` builds a
  `TFInterventionPlan` over two writable levels: module-boundary substitution
  (`apply_tf_module_intervention`) and the functional wrap (`tf_intervention_wrap`),
  whose targets come from `_wrap_registry.iter_wrap_targets()`: the curated
  `_CURATED_WRAP_ENTRIES` first, then every generated op wrapper in the `gen_*_ops`
  modules and the namespace bindings that are the same objects. `tf.raw_ops.*` are
  copies and stay unreachable.
- Static single-rule selectors become static op sites; callables and multi-rule
  specs become value sites (`_value_site`). Every op site is evaluated ONCE per
  recorded output in the callback (`observe_tf_op_sites`) into
  `plan.pending_decisions`; the wrap layer pops and fires them when the output
  reaches Python, then calls `plan.on_presented` (the deferred-halt release).
- Builtin helper adapters: `zero_ablate`, `scale`, `add` ONLY. Other helpers
  (including `replace_with`) raise `BackendUnsupportedError`; pass a callable
  action for custom replacements.
- `audit_tf_site_reachability()` is FAIL-CLOSED: a selector matching
  callback-captured ops the wrap layer never presented raises
  `TFInterventionSiteUnreachableError` instead of silently not firing.

## modules.py
- `discover_tf_module_tree()` + `patched_tf_module_stack()` capture Keras /
  `tf.Module` module stacks via class-`__call__` patching; `tf_param_logs()`
  builds Param records from module variables.
- `returns_hook()` extends the session's module-exit hook to record each call's returned
  tensors (after any module-boundary intervention) into `TFModuleTree.module_returns`, labels
  from the session's `producer_by_ref`; the shared `attach_object_module_logs_with_returns()`
  builds module logs from it, so `Module.out` works (contract: `../_module_returns.py`).

## validation.py
- `validate_tf_trace()` / `validate_tf_trace_detailed()` replay and perturb ops
  against the `replay_allowlist()` of per-op-type replayers.

## _tf_compat.py
- `HAS_TF_OP_CALLBACKS` probe; `mark_tf_capability_missing()` and
  `get_tf_capability_snapshot()` feed doctor/compat reporting.

## Local Invariants
- Deferred surfaces refuse typed, never silently no-op: true backward capture,
  module clauses inside value-dependent or multi-rule specs, and edits on ops that
  bypass the python wrappers (`tf.raw_ops.*`, `tf.function` bodies).
- `grad_options=` and `intervene=` never combine; static FuncGraph captures
  refuse both and keep `recurrence_detection=False`.
- `capabilities.py` only re-exports registry truth; never hardcode flags here.
