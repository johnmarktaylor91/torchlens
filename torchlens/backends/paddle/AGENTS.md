# backends/paddle/ - Implementation Guide

Paddle preview backend: eager dygraph wrapper capture. Tier-1 standalone spec
(no `capture_backend`); the `BackendSpec` in `../default_specs.py` is the only
capability truth.

## backend.py
- `PaddleBackend.capture_trace()` is the standalone entry; `GradOptions`
  (re-exported from `__init__.py`) drives derived gradients including
  intermediates (`intermediate_derived_grads=True`).
- Live forward `intervene=` AND `halt=` both ship (the wrapper holds each
  concrete output before the caller sees it); combining `grad_options=` with
  either refuses typed. Halted traces set `halted` / `stop_reason` /
  `halt_frontier`.
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
- Builtin helper adapters (`_PADDLE_SUPPORTED_HELPER_NAMES`): `add`,
  `replace_with`, `scale`, `zero_ablate`; others refuse typed.
- Decisions are forward-only; the module imports without Paddle installed.

## model_prep.py
- `discover_paddle_module_tree()` builds `PaddleModuleTree`;
  `prepare_model_session()` installs layer pre/post hooks,
  `cleanup_model_session()` removes them.

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
