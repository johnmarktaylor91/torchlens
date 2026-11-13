# backends/mlx/ - Implementation Guide

MLX backend: eager wrapper capture with live per-op replay validation.
It fills the backend protocol (no `capture_backend`); its
`BackendSpec` in `../default_specs.py` and its protocol shell in
`../_protocols/mlx.py` are the capability truth (contract: `../AGENTS.md`).

## backend.py
- `MLXBackend.capture_trace()` is the standalone entry; `GradOptions`
  (re-exported from `__init__.py`) drives derived gradients including
  intermediates (`_MLXIntermediateTapObserver`, `intermediate_derived_grads=True`).
- Static-label `intervene=` AND `stop_after=` both ship; combining either with
  `grad_options=` refuses typed (derived-grad replay would misrepresent the
  intervened forward). Halted traces set `is_stopped` / `stop_reason` /
  `halt_frontier`. A `stop_after` selector that never fires refuses typed with
  torch's `stop_after_never_fired` (the shared policy in
  `torchlens/_capture/_stop_after_policy.py`); an intervene selector matching
  zero sites warns. Those capture-time disclosures (zero-match selectors,
  compiled attributes) live in `_capture_disclosures.py`.
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
  `MLXHookApplier`; `_reject_non_static_selector()` admits the static-label
  kinds plus `predicate` (`tl.where` and bare callables, wrapped by
  `_as_selector()`); module selectors mixed with anything else still refuse.
- `_match_live_sites()` matches each wrapped call's output leaves against the
  edit plan and stop selector. When one of them holds a `predicate`,
  `_value_record_context()` sets `is_scalar_bool` / `bool_value` to `None`,
  matching torch's live capture contexts. Never evaluate an array in this path:
  every capture stays realize-once. `tensor_requires_grad` stays deferred (MLX
  has no per-array flag). If torch's live contexts start carrying values, change
  both backends together.

## _source_frames.py
- `mlx_code_context()` fills `Op.code_context` like the torch walk: TorchLens
  frames skipped, call site first, the op's frame last. The model boundary is
  the `MLXBackend.capture_trace` frame, matched by code name and the sibling
  `backend.py` path (`_is_anchor`; rename either and update it), not a
  `forward` name. Records are cached in
  `trace._code_context_cache` and adopted into the trace core at finalize
  (`adopt_code_context_records`).
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
