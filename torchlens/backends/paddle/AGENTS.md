# backends/paddle/ - Implementation Guide

Paddle backend: eager dygraph wrapper capture. It fills the backend protocol (no `capture_backend`); its
`BackendSpec` in `../default_specs.py` and its protocol shell in
`../_protocols/paddle.py` are the capability truth (contract: `../AGENTS.md`).

## backend.py
- `PaddleBackend.capture_trace()` is the standalone entry; `GradOptions`
  (re-exported from `__init__.py`) drives derived gradients including
  intermediates (`intermediate_derived_grads=True`).
- Live forward `intervene=` AND `stop_after=` both ship (the wrapper holds each
  concrete output before the caller sees it). With `grad_options=`, derived
  gradients replay the edited or halted program (`_edit_replay.py`). Halted traces set `is_stopped` / `stop_reason` /
  `halt_frontier`. A `stop_after` that never fires follows torch's policy
  (`PaddleInterventionRuntime.settle_unfired`, shared
  `torchlens/_capture/_stop_after_policy.py`): a selector refuses typed
  `stop_after_never_fired`, a plain callable warns
  `stop_after_never_fired_callable`.
- Intervention/halt predicates may be VALUE-DEPENDENT: dygraph is eager, so the
  `RecordContext` fields `tensor_requires_grad`, `is_scalar_bool`, and
  `bool_value` carry real values, not the lazy-backend deferred sentinel.
- The replay oracle uses a narrow corroborated user-intervention carve-out
  (`PaddleInterventionCapture` sidecar facts); a replacement value presented as
  captured-native FAILS validation.
- Recurrence grouping runs through the shared
  `.._finalize.finalize_single_pass_trace` neutral grouper
  (`recurrence_detection` default True).

## wrappers.py
- `wrap_paddle()` / `unwrap_paddle()` / `is_paddle_wrapped()`;
  `_PaddleWrapperRegistry` owns the wrapped surface and
  `paddle_wrap_inventory()` returns the `PaddleInventory` audit.
- `is_alias_allowed_op()` gates alias-returning ops; mutator-named candidates
  are denied (`_is_mutator_name` -> `_raise_denied`).

## interventions.py
- `PaddleInterventionRuntime` validates and evaluates the intervene/halt
  predicates per wrapped call; hooks must return a `paddle.Tensor`.
- Builtin helper adapters (`_PADDLE_SUPPORTED_HELPER_NAMES`, applied in
  `_edit_helpers.py`, which the replay adapter's `apply_helper` also uses for
  fork edits): `add`, `clamp`, `mean_ablate` (self form), `replace_with`,
  `scale`, `zero_ablate`; others refuse typed. The literal tuple stays in
  `interventions.py` (read by AST in tests).
- Records each replacing hook by wrapped-call ordinal and output path
  (`fired_edits`) and the halting call (`halt_call_id`) for the derived-grad replay.
- `tl.grad(site)` clauses are split off the live predicate at init and armed on
  the finished trace (`arm_gradient_sites`) through
  `intervention/_grad_site_neutral.py`; the caller passes
  `hooks._target_operands` so that module stays out of the import cycle.
  Gradient-edit callables are `fn(grad)`. Autograd-node selectors refuse.
- The module imports without Paddle installed.

## _edit_replay.py, _grad_sites.py
- `_edit_replay.run_derived_grad_forward` re-runs the model through a tap that
  re-applies fired hooks by call ordinal, stops at the halt call, and registers
  armed `tl.grad` edits as Paddle gradient hooks on the site outputs.
- `_grad_sites.paddle_grad_site_records` recomputes an armed pass with the
  reverse engine, refuses if its input-leaf gradients disagree with
  `paddle.grad`, and returns the intermediate records. Keep both modules free of
  imports from the selector/hook cycle (layering largest_scc ratchet).

## model_prep.py
- `discover_paddle_module_tree()` builds `PaddleModuleTree`;
  `prepare_model_session()` installs layer pre/post hooks,
  `cleanup_model_session()` removes them.
- The post-hook records each layer call's returned tensors (after any module-exit
  intervention) into `PaddleModuleTree.module_returns` through `record_layer_return()`, labels
  read at exit from the wrapper-bound backend (`wrappers.bound_paddle_backend()`); the shared
  `attach_object_module_logs_with_returns()` builds module logs from it, so `Module.out` works
  (contract: `../_module_returns.py`).

## tensor_store.py
- `PaddleTensorLabelStore` maps live tensor identity to raw labels.

## validation.py
- Replay + perturbation over rebuilt inputs (`_rebuild_inputs`,
  `_parent_perturbations_change_output`) with a `_coverage_oracle` over the
  whole trace.

## Local Invariants
- Value-dependent `save=`, streaming, fastlog, and rng_replay stay refused
  (registry flags + `.._options` policies); true backward capture is
  unsupported (`backward_capture=False`).
- Payloads serialize as `array_payloads`; bf16 payloads carry logical dtype
  metadata (`paddle_bfloat16_transport_bits`) because NumPy transports them as
  `uint16` -- the codec lives in `torchlens/_io/payload_codec.py`, not here.
- `capabilities.py` only re-exports registry truth; never hardcode flags here.
