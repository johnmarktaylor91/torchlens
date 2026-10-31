# backends/jax/ - Implementation Guide

Jaxpr-first JAX backend preview. Tier-1 standalone spec (no `capture_backend`);
the `BackendSpec` in `../default_specs.py` is the only capability truth.

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
  `_raise_nnx_trace_context_error`.
- Recurrence grouping calls `group_recurrent_nodes` directly (the same neutral
  grouper torch uses), gated by `recurrence_detection` (default True).

## jaxpr.py
- Closed-jaxpr derivation and interpretation. `SAFE_JIT_NAMES` allowlists
  inlineable jit sub-jaxprs. The DEFAULT control-flow mode is `"unroll"`:
  `cond`/`scan`/`while_loop` (and forward `custom_vjp_call` regions) are
  CAPTURED by mode-gated handlers that run before the refusal backstop —
  only `remat2` reaches the raise on the default path (and today as a bare
  `ValueError`, not the `BackendUnsupportedError` the backends contract
  asks for — known gap). `REJECTED_NESTED_PRIMITIVES` refusals apply under
  the explicit reject mode, not the default; `EFFECT_PRIMITIVES` and
  `PURE_JIT_CALL_PRIMITIVES` classify callback and jit-call equations.

## modules.py
- Module helpers for Equinox and Flax NNX roots — `EquinoxModuleTree` for
  Equinox and the separate `NnxModuleTree` for NNX;
  `decode_module_scope()` / `decode_module_call_scope()` decode the `tlm_` /
  `tlc_` scope-label prefixes that thread module identity through the jaxpr.

## capabilities.py
- Registry-backed re-exports only. Per the registry: `interventions=False`,
  `intermediate_derived_grads=True`, backward capture / fastlog / rng replay /
  streaming all False, `payload_policy="array_payloads"`.

## Local Invariants
- `intervene=` / `halt=` are NOT supported on jax (unlike the tf/mlx/paddle
  previews); do not claim intervention parity.
- The capture is static (jaxpr interpretation), not eager wrapper capture;
  taken-branch honesty depends on the control-flow mode, and rejected nested
  primitives must stay typed refusals rather than silent boundary nodes.
- Module scope labels are the only module-identity channel; keep the
  `tlm_`/`tlc_` codecs in `modules.py` and `jaxpr.py` in lockstep.
