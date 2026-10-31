# backends/tinygrad/ - Implementation Guide

tinygrad preview backend: UOp-snapshot capture of an eager forward. Tier-1
standalone spec (no `capture_backend`); the `BackendSpec` in
`../default_specs.py` is the only capability truth. This is the smallest
preview: `backend.py` plus registry-backed `capabilities.py` only.

## backend.py
- `TinygradBackend.capture_trace()` is the standalone entry: it runs the
  callable under `_observe_tensor_ops` (Tensor-API observation building
  `TinygradUOpCapture` records from pre-realization UOp lineage) and
  `_reject_mid_capture_execution` (typed refusal of mid-capture realization
  that would truncate lazy UOp lineage).
- `GradOptions` (re-exported from `__init__.py`) drives derived gradients
  including intermediates (`TinygradIntermediateCandidate`,
  `intermediate_derived_grads=True`).
- Module identity: `discover_tinygrad_module_tree()` builds
  `TinygradModuleTree` / `TinygradModuleFrame`; `scoped_tinygrad_module_calls`
  scopes call frames, `_module_stack_for_uop` attributes UOps to modules, and
  `tinygrad_param_logs()` builds Param records from module tensor attrs.
- Recurrence grouping runs through the shared
  `.._finalize.finalize_single_pass_trace` neutral grouper
  (`recurrence_detection` default True).
- Restricted option surface (each documented in the `capture_trace` docstring):
  `layers_to_save` must be `"all"` (full-save only for live traces),
  `output_device` must be `"same"`, `save_arg_values` / `save_code_context` /
  `save_rng_states` must be false, and `save_grads` / `backward_ready` /
  `transform` / `module_filter` / `activation_transform` are unsupported.

## capabilities.py
- Registry-backed re-exports only. Per the registry: `interventions=False`,
  `intermediate_derived_grads=True`, backward capture / fastlog / rng replay /
  streaming all False, `validation_replay=True`,
  `payload_policy="array_payloads"`.

## Local Invariants
- `intervene=` / `halt=` are NOT supported on tinygrad (unlike the tf/mlx/
  paddle previews); unsupported trace options refuse typed through the
  `.._options` TINYGRAD policies, never silently no-op.
- Capture correctness depends on unrealized UOp lineage: anything that forces
  realization mid-capture must stay a typed refusal.
- `capabilities.py` only re-exports registry truth; never hardcode flags here.
