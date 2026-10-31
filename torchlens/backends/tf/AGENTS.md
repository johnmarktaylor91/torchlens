# backends/tf/ - Implementation Guide

TensorFlow preview backend. Tier-1 standalone spec (no `capture_backend`); the
`BackendSpec` in `../default_specs.py` is the only capability truth.

## backend.py
- `TFBackend.capture_trace()` is the standalone entry. Eager `op_callbacks` live
  capture is the shipped PRIMARY path; the static FuncGraph path handles
  compiled/SavedModel entries.
- `intervene=` combined with `grad_options=` refuses typed (the derived-gradient
  replay reruns the un-intervened forward). Both surfaces require eager live
  capture; the static path refuses each.
- `halt=` and `recipes=` refuse typed as deferred on TF (NOT torch-only
  elsewhere: Paddle's preview ships both `halt=` and `recipes=`), even though
  the registry `interventions` flag admits them through the shared option
  gate.
- `_finish_trace(recurrence_detection=...)` runs the shared neutral grouper via
  `.._finalize.finalize_single_pass_trace`; the eager path passes the user's
  request, the static FuncGraph path stays ungrouped and stores the honest
  EFFECTIVE value `False`.

## op_callback_capture.py
- `TFEagerCaptureSession.run()` drives eager capture through TF `op_callbacks`:
  real values, taken-branch control flow, op-level `TFOpCapture` records.
- `warm_up_tf_callable()` pre-runs the callable outside the callback window.

## funcgraph.py
- `capture_static_funcgraph()` is the graph-only static path; opaque regions are
  emitted honestly unverified (`_capture_opaque_graph_region`).

## derived_grads.py
- `GradOptions` (re-exported from `__init__.py`) + `attach_tf_derived_grads()`:
  one GradientTape replay attaching leaf and exact-T1 intermediate gradients,
  with divergence refusal (`_tf_trees_close`).

## interventions.py
- Static-label `intervene=` for eager entries. `normalize_tf_interventions()`
  builds a `TFInterventionPlan` over two writable levels: module-boundary
  substitution (`apply_tf_module_intervention`) and the curated
  `_CURATED_WRAP_ENTRIES` functional wrap (`tf_intervention_wrap`) — entries
  span FOUR namespaces: `tf.nn`, `tf.math`, `tf.linalg` (`matmul`), and
  top-level `tf` callables (`tf.matmul`, `tf.add`, `tf.concat`, ...).
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

## validation.py
- `validate_tf_trace()` / `validate_tf_trace_detailed()` replay and perturb ops
  against the `replay_allowlist()` of per-op-type replayers.

## _tf_compat.py
- `HAS_TF_OP_CALLBACKS` probe; `mark_tf_capability_missing()` and
  `get_tf_capability_snapshot()` feed doctor/compat reporting.

## Local Invariants
- Deferred surfaces refuse typed, never silently no-op: `halt=`, `recipes=`,
  true backward capture, value-dependent predicate conditions.
- `grad_options=` and `intervene=` never combine; static FuncGraph captures
  refuse both and keep `recurrence_detection=False`.
- `capabilities.py` only re-exports registry truth; never hardcode flags here.
