# backends/jax/ - Implementation Guide

Jaxpr-first JAX backend. It fills the backend protocol (no `capture_backend`); its
`BackendSpec` in `../default_specs.py` and its protocol shell in
`../_protocols/jax.py` are the capability truth (contract: `../AGENTS.md`).

## backend.py
- `JAXBackend.capture_trace()` derives a closed jaxpr from the callable
  (expected `fn(params, *inputs)` convention) and interprets it into a `Trace`.
- `GradOptions` (re-exported from `__init__.py`) drives derived gradients for
  input/param leaves and intermediates (`intermediate_derived_grads=True`);
  oracles include `_finite_difference_directional_check` and the private
  `_experimental_per_op_boundary_vjp_oracle`.
- JAX-specific kwargs: `jax_static_argnums`, `jax_control_flow`
  (`"reject"` / `"unroll"` / `"region"`, default `"unroll"`), and
  `jax_max_control_flow_unroll`.
- Typed entry refusals: `_reject_transformed_callable` (jit/grad/vmap-wrapped
  models), `_reject_tracer_inputs`, `_reject_closed_over_host_state`, and
  `_raise_nnx_trace_context_error`; `_readable_closure_consts` (in
  `_closure_consts.py`) refuses a function-root closure constant with no
  readable buffer (`constant_unrealizable`).
- Keyword inputs (`_input_kwargs.py`): array-valued keywords become one trailing
  positional dict of jaxpr leaves; other keyword values bind statically. The
  wrapper is a plain function so `_reject_closed_over_host_state` (now in
  `_host_state.py`) can still inspect the user callable behind it.
- Derived gradients over module roots and closure constants
  (`_converted_grads.py`) evaluate the captured closed jaxpr with its constants as
  explicit arguments; `derive_closed_jaxpr` is passed in so the module stays out of
  the import cycle.
- Recurrence grouping calls `group_recurrent_nodes` directly (the same neutral
  grouper torch uses), gated by `recurrence_detection` (default True).

## jaxpr.py
- Closed-jaxpr derivation and interpretation. The DEFAULT control-flow mode is
  `"unroll"`: `cond`/`scan`/`while_loop` (and forward `custom_vjp_call` regions)
  are CAPTURED by mode-gated handlers that run before the refusal backstop;
  `REJECTED_NESTED_PRIMITIVES` refusals apply under the explicit reject mode.
  `EFFECT_PRIMITIVES` and `PURE_JIT_CALL_PRIMITIVES` classify callback and
  jit-call equations; they and the nested-call inlining predicates live in
  `_jaxpr_inlining.py`, and `jaxpr.py` re-exports the two sets. Inside an attributed Equinox/NNX module the enclosing
  equation's `tlm_`/`tlc_` scope is threaded into nested jit, taken cond
  branches, scan and while bodies; synthetic control captures (`scan_read`,
  `scan_stack`, `cond_decision`, `while_decision`) get the equation's stacks.
  `checkpoint`/`remat2`/`custom_vjp_call`/`shard_map` inside an attributed
  module refuse through the existing `_entry_refusals` codes with an
  "inside attributed module" note; never a bare `ValueError`.
- `_interp_hooks.py`: `InterpreterHooks(session, halt, bind_combiners)`. Retention
  placeholders (`jax/_retention.py` contract: `keep(capture_index)`), the halt
  check at capture append (`InterpreterHalt` unwinds every nested frame; the
  result carries `halt_frontier`, no suffix primitive binds) and the pullback
  entry `interpret_pullback_jaxpr` (binds combiner primitives such as
  `scatter-add`).
- `_module_stacks.py`: the `tlm_`/`tlc_` codec and relative-stack composition
  (nested name stacks carry only their own scope); `CallNumbering` gives a module
  called inside a loop iteration or a jit bind one runtime call per run, as on
  PyTorch (`module_call_renumbering` on the result moves later trace-time calls;
  `_module_return_probe.split_outputs` applies it to forward args and the ledger).
  Leaf module. A module capture with an imported region (scan or while over the
  unroll cap, or `jax_control_flow="region"`) refuses typed in `split_outputs`:
  `backend._finish_trace` pairs module params with `result.equations`, which has no
  region captures, so a region would misalign them; passing `result.captures`
  there lifts it.
- `_jit_consts.py`: `derive_closed_jaxpr` hoists the closure constants of top-level
  jit calls (a module's weights read inside `jax.jit`) to top-level constants, so
  their reads are recorded parents (module params, or `constant` sources outside a
  module). A jit inside a cond,
  scan or while body that closes over constants still refuses (purity gate).
- `_halt.py`: `JaxStop`, the `stop_after=` site for `capture_trace` (selector or
  callable, matched on the equation context at hook time; halted traces get
  torch's `is_stopped`/`stop_reason`/`_graph_stop_frontier`/`stopped` slot).
  Uses the session's private `_reserve`/`_context`.

## modules.py
- Module helpers for Equinox and Flax NNX roots — `EquinoxModuleTree` for
  Equinox and the separate `NnxModuleTree` for NNX;
  `decode_module_scope()` / `decode_module_call_scope()` decode the `tlm_` /
  `tlc_` scope-label prefixes that thread module identity through the jaxpr.

## _return_ledger.py, _module_return_probe.py
- Recorded module returns (the shared seam in `../_module_returns.py`). Module calls
  run under `jax.make_jaxpr`, so returns are tracers: `extend` traces the model plus
  every top-level tracer leaf a module call returned as extra jaxpr outputs, `strip`
  gives the backend the same jaxpr without them (stamped; same equations, no op
  added), `split_outputs` moves the extra outputs off the interpreter result, and
  `note_return_labels` / `build_module_logs_with_returns` resolve them to op labels.
  A call under `jax.vmap` (inner-trace tracers) stays unrecorded, never guessed;
  a call inside a scan, cond or while body (body-trace tracers) is likewise left
  unrecorded and disclosed with a `CaptureGapWarning`; a halted run releases the
  ledger. The probe
  must not import import-cycle members (`jaxpr.py`, `_postprocess.finalization`):
  backend.py passes them in, which keeps `largest_scc` at its baseline.

## _pytree_param_barcodes.py
- Stamps param barcodes on raw pytree ops before recurrence grouping (the full
  attachment runs after it), so distinct same-shape modules stay separate layers.

## capabilities.py
- Registry-backed re-exports only. Per the registry: `can_intervene=False` (the
  backend implements edits; the registry flip belongs to the registry owner),
  `intermediate_derived_grads=True`, backward capture / fastlog / rng replay /
  streaming all False, `payload_policy="array_payloads"`.

## _edits.py, _edit_helpers.py, _source_lines.py
- `_edits.py` holds the per-capture equation session: the interpreter's
  `equation_hook` simulates labels with the same counters `CaptureEvents` uses
  (inputs, then closure constants, then one per capture; region captures count as
  `jax_region` / `jax_region_out`), so the label a predicate sees is the label the
  event gets. Any change to emission order must change the simulation with it.
- An edited capture is a `JaxEditedEquationCapture` carrying its appliers; the
  validator replays the primitive, re-applies them and compares with the saved
  output, and checks parent sensitivity on the raw replay. Never skip the replay
  for an edited op, and never compare against the raw output.
- Replacements are copied (`checked_replacement`) because parent edges resolve
  by value identity; shape and dtype changes refuse (`intervention_result_invalid`).
- `_edits.py` is the one module that joins the import cycle; `_edit_helpers.py` and
  `_source_lines.py` must not import cycle members.
- `_source_lines.py` maps `eqn.source_info.traceback` to `FuncCallLocation`
  frames: drop jax, jaxlib and TorchLens frames; the traced region ends at the first
  TorchLens frame. Equation tracebacks stop at the traced function, so the call site
  is read from the live stack when the session opens.
- Every refusal raise passes `code=` and has a row in
  `tests/composition_expectations/data/refusal_site_classification.tsv`. Refusal
  factories need package-unique names: the census keys factories by bare name.
- Derived-gradient closures must not close over host scalars or arrays (the
  host-state check rejects them); select behaviour with a function, not a flag.

## Local Invariants
- `intervene=` edits equation outputs at capture time (`_edits.py`); `stop_after=`
  halts at capture time through `_halt.py` once `backend.py` wires `JaxStop` and
  the options policy stops refusing it; until then it is refused as a runtime option.
- The capture is static (jaxpr interpretation), not eager wrapper capture;
  taken-branch honesty depends on the control-flow mode, and rejected nested
  primitives must stay typed refusals rather than silent boundary nodes.
- Module scope labels are the only module-identity channel; the `tlm_`/`tlc_`
  codec lives in `_module_stacks.py` and the writer in `modules.py`: keep them
  in lockstep.
- Perturbation exemptions are value proofs only. `_zero_annihilator.py` (torch
  parity, reason `multiplicative_zero_annihilator`) credits a `mul`/`dot_general`
  edge no candidate moved only when the parent fills one slot and the saved
  co-operand and saved output are exactly zero everywhere; it runs after the
  replay matched and never replaces it. Pinned by `tests/test_jax_zero_annihilator.py`.
