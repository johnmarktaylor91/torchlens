# backends/tinygrad/ - Implementation Guide

tinygrad preview backend: UOp-snapshot capture of an eager forward. It fills the backend protocol (`maturity="preview"`, no `capture_backend`); its
`BackendSpec` in `../default_specs.py` and its protocol shell in
`../_protocols/tinygrad.py` are the capability truth (contract: `../AGENTS.md`). This is the smallest
preview: `backend.py` plus registry-backed `capabilities.py` only.

## _uop_graph.py
- The UOp walk helpers `backend.py` uses (file cap). `_data_flow_uops` decides
  which UOps become ops: only values that reach an output as data, walked from
  the outputs through materializable UOps and stopped at the capture's input
  UOps. A movement op's scalar shape `CONST`s and an input's storage `BUFFER`
  are not data of the forward and are never recorded (an unconsumed constant
  failed `c1_constant_parents_recorded`). `_is_materializable_uop` is
  structural (shape, device, dtype; a view over a host NPY buffer is not
  materializable) and never realizes. Regenerate the conformance
  inventories (`tests/conformance/_rosters/_tinygrad_inventories.py` and the
  probe rows in `tests/conformance/_preview_adapters.py`) when this rule moves.

## _realize_once.py
- Realize-once capture. After the observed forward, `plan_capture` decides the
  recorded UOps on the untouched graph (no realization), then
  `realize_recorded_values` realizes inputs, recorded values and outputs in one
  `TinygradProtocol.realize_once` call. Each value is listed as
  `CONTIGUOUS(uop)`: it gets its own buffer while consumers and outputs still
  fuse exactly as in eager (listing the bare UOp makes it a buffer boundary
  and changes downstream rounding, which fails exact parity). A buffer the
  capture created is the payload as is; a shared buffer is copied. Read every
  UOp before the call: tinygrad repoints live tensors at their buffers when it
  realizes.
- Op names: the observer sees tinygrad's internal helpers (`_mop`, `_rop`,
  lambdas); `_op_func_name` falls back to the UOp op name for those.
- tinygrad gotcha: realizing the RNG counters (reading
  `Tensor._device_rng_counters`) before a lazy random weight is realized
  changes the weight's values. Read eager outputs before RNG state.
- The host-only NPY device cannot copy data in; host tensors there are built
  from numpy (`_host_tensor`).
- RNG boundary: a Threefry-bearing UOp that a live tensor held before the
  forward (`live_tensor_uops`, from `tinygrad.tensor.all_tensors`) is a
  pre-forward random value (a default-init weight). Its generator subgraph is
  not walked; it is recorded as the buffer ops its `CONTIGUOUS` realized into
  (`recorded_values`), the same ops a pre-realized weight records. RNG drawn
  inside the forward refuses `rng_state_in_graph` before any realization.
  Never realize a recorded UOp a second time (it would rerun the counter
  store and draw different values).

## backend.py
- `TinygradBackend.capture_trace()` is the standalone entry: it runs the
  callable under `_observe_tensor_ops` (Tensor-API observation building
  `TinygradUOpCapture` records from pre-realization UOp lineage) and
  `_reject_mid_capture_execution` (in `_execution_guard.py`; typed refusal of
  mid-capture realization that would truncate lazy UOp lineage).
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
  `output_device` must be `"same"`, `save_args` / `save_source_context` /
  `save_rng` must be false, and `save_grads` / `backward_ready` /
  `transform` / `activation_transform` are unsupported.

## capabilities.py
- Registry-backed re-exports only. Per the registry: `can_intervene=False`,
  `intermediate_derived_grads=True`, backward capture / fastlog / rng replay /
  streaming all False, `validation_replay=True`,
  `payload_policy="array_payloads"`.

## Local Invariants
- `intervene=` / `stop_after=` are NOT supported on tinygrad (unlike the tf/mlx/
  paddle previews); unsupported trace options refuse typed through the
  `.._options` TINYGRAD policies, never silently no-op.
- Capture correctness depends on unrealized UOp lineage: anything that forces
  realization mid-capture must stay a typed refusal.
- `capabilities.py` only re-exports registry truth; never hardcode flags here.
