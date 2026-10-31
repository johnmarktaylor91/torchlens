# backends/mlx/ - Implementation Guide

MLX preview backend: eager wrapper capture with live per-op replay validation.
Tier-1 standalone spec (no `capture_backend`); the `BackendSpec` in
`../default_specs.py` is the only capability truth.

## backend.py
- `MLXBackend.capture_trace()` is the standalone entry; `GradOptions`
  (re-exported from `__init__.py`) drives derived gradients including
  intermediates (`_MLXIntermediateTapObserver`, `intermediate_derived_grads=True`).
- Static-label `intervene=` AND `halt=` both ship; combining either with
  `grad_options=` refuses typed (derived-grad replay would misrepresent the
  intervened forward). Halted traces set `halted` / `stop_reason` /
  `halt_frontier`; a halt selector matching zero sites warns.
- `mx.compile` / `mx.grad` / `mx.vmap` traced-transform entries refuse typed at
  capture entry (`_find_mlx_compiled_attributes`, `_mlx_traced_transform_type`);
  a compiled model attribute ceilings the capture (`capture_verified=False`).
- Recurrence grouping runs through the shared
  `.._finalize.finalize_single_pass_trace` neutral grouper
  (`recurrence_detection` default True).

## wrappers.py
- `wrap_mlx()` / `unwrap_mlx()` / `is_mlx_wrapped()` install and remove the
  eager wrappers; `_MLXWrapperRegistry` owns the wrapped surface and
  `mlx_tap_observer()` scopes an observer over a capture. With an observer
  active the wrapper passes `module_address=` (the discovered module's primary
  address, else `None`) so the intermediate-gradient replay observers mirror
  capture's `"address:call_index"` module frames; their signatures must match
  `op.modules` or module-shaped models get no intermediate gradients.

## interventions.py
- `resolve_mlx_intervention_plan()` builds `MLXInterventionPlan` /
  `MLXHookApplier`; `_reject_non_static_selector()` keeps the surface
  static-label only (value-dependent predicate fields stay deferred).
- Curated MLX-native helper appliers (`_MLX_SUPPORTED_HELPER_NAMES`):
  `zero_ablate`, `scale`, `add`, `clamp`, `mean_ablate`, `replace_with`;
  anything else refuses typed rather than half-working through torch hooks.

## model_prep.py
- `discover_mlx_module_tree()` builds `MLXModuleTree`; `prepare_model_once()` /
  `prepare_model_session()` / `cleanup_model_session()` own session lifecycle.

## tensor_store.py
- `MLXTensorLabelStore` maps live array identity to raw labels during capture.

## validation.py
- `validate_mlx_captures()` replays `MLXOpCapture` templates
  (`build_capture_template()`) with perturbation evidence and declared-parent
  consistency checks; coverage failures count against the trace.

## Local Invariants
- True backward capture is unsupported: MLX traces always report
  `Trace.autograd.has_backward_pass = False`; RNG replay snapshots are `None`.
- fastlog, streaming, and rng_replay stay refused (registry flags); unsupported
  trace options refuse typed through `.._options` policies.
- `capabilities.py` only re-exports registry truth; never hardcode flags here.
