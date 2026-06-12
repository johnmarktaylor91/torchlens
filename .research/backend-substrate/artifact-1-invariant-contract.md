# M0.1a Artifact 1: Backend Invariant Contract

Date: 2026-06-12
Plan source: `.research/jax-tinygrad-sprint_PLAN.md` v13, M0.1a artifact 1.
Scope: design artifact only. No code, tests, ruff, pytest, or benchmark work.

## Purpose

This contract defines which TorchLens trace invariants must remain byte-stable for torch,
which checks become backend-neutral, and which checks are backend-mode-specific when JAX
jaxpr-first capture and later tinygrad/MLX substrate work are added.

Current code has two validation layers:

- behavioral replay validation: output equality, per-op replay, parent perturbation, and
  completeness in `torchlens/validation/core.py:1`, `torchlens/validation/core.py:159`,
  `torchlens/validation/core.py:251`, and `torchlens/validation/core.py:475`;
- structural metadata validation: grouped checks A through T plus func-call-id consistency in
  `torchlens/validation/invariants.py:1`, called in dependency order at
  `torchlens/validation/invariants.py:92`.

The implementation contract is: `Trace.validate_forward_pass()` must dispatch by
`trace.backend` and return a real `bool` for every supported backend. It must not silently
skip validation. Current `Trace.validate_forward_pass()` hard-calls
`torchlens.validation.core.validate_saved_outs` (`torchlens/data_classes/trace.py:5333`), so
M0.2 must introduce backend dispatch before JAX validation lands.

## Backend Modes

The invariant suite is evaluated under an explicit trace module-identity mode:

| Mode | First user-facing use | Meaning | Required module root |
|---|---|---|---|
| `torch_module` | Existing torch | `nn.Module` address tree and call tree are authoritative. | `self` is the root module. |
| `function_root` | JAX M1 raw functions | The declared callable is exposed as a call root, not as a framework module. | `self`, `kind="function_root"`. |
| `pytree_module` | Future Equinox adapter | Pytree paths define module/param identity. | `self` plus pytree-derived records. |

Current `Trace.backend` is typed as `Literal["torch", "mlx"]`
(`torchlens/data_classes/trace.py:1007`) and initialized to `"torch"`
(`torchlens/data_classes/trace.py:1351`). M0.2 must widen the backend name type and add a
stored `module_identity_mode` before mode-specific invariants are executable.

## Invariant Inventory

### V0. Validation Entry Contract

Current behavior:

- `validate_forward_pass(model, ...)` runs a direct model forward, then traces with
  `layers_to_save="all"`, `save_arg_values=True`, `detach_saved_activations=False`, and
  `save_rng_states=True` before calling `Trace.validate_forward_pass()`
  (`torchlens/user_funcs.py:2923`, `torchlens/user_funcs.py:2983`,
  `torchlens/user_funcs.py:3020`, `torchlens/user_funcs.py:3033`).
- `Trace.validate_forward_pass()` delegates to torch replay validation
  (`torchlens/data_classes/trace.py:5333`).
- Portable bundles without executable callables are rejected before replay
  (`torchlens/validation/core.py:75`).

Contract:

- Torch keeps the existing path unchanged.
- JAX M1 must supply a backend validation implementation based on jaxpr interpreter replay:
  output equality, per-equation replay, parent perturbation, and invariant checks.
- MLX/tinygrad must either provide full payload validation or raise a documented unsupported
  error. Returning `True` from metadata-only validation is forbidden.
- Validation captures force full-save prerequisites for the backend. Save-shaping kwargs that
  would make replay impossible are rejected at API entry, not skipped later.

### V1. Ground-Truth Output Equality

Current behavior:

- Non-halted traces compare each saved output node to direct model output before graph replay
  (`torchlens/validation/core.py:191`).
- Floating outputs get tight output-only tolerance; shape, dtype, NaN, and inf structure are
  checked (`torchlens/validation/core.py:108`).
- Output nodes are synthetic children of the real output parent, with `parent_arg_positions`
  linking arg slot 0 to the parent (`torchlens/postprocess/graph_traversal.py:180`,
  `torchlens/postprocess/graph_traversal.py:294`).

Contract:

- Backend-neutral: every full-save trace must have explicit output nodes or backend-equivalent
  terminal records that can be compared positionally to declared outputs.
- JAX M1 must normalize pytree output leaves into stable `container_path` records, then compare
  concrete arrays after async barriers.
- If a backend cannot produce concrete output payloads, `validate_forward_pass()` raises
  unsupported.

### V2. Replay and Parent-Perturbation Tripwire

Current behavior:

- Replay walks backward from output/internal-sink seeds and validates parent edges before
  enqueueing parent nodes (`torchlens/validation/core.py:210`,
  `torchlens/validation/core.py:328`).
- Each child op checks that parent outs appear exactly in saved args/kwargs
  (`torchlens/validation/core.py:475`).
- Each replayable op re-executes from saved parents, then perturbs each representative parent
  and expects output change (`torchlens/validation/core.py:300`,
  `torchlens/validation/core.py:310`).
- In-place parent versions are saved in `out_versions_by_child` when a child saw a different
  parent value (`torchlens/backends/torch/ops.py:2188`).

Contract:

- Backend-neutral: full validation must prove not only that final output matches, but that every
  declared parent edge is used by the child computation.
- Torch keeps saved-arg replay and `out_versions_by_child`.
- JAX M1 replaces function-call replay with equation replay from the closed jaxpr. Parent
  perturbation applies to jaxpr variable inputs for each equation, not Python args.
- The implementation round must include an adversarial fixture where final output still matches
  after a dropped parent/site attribution; validation must fail.

### A. Trace Self-Consistency

Current check: label uniqueness, op count, param totals, output presence, timing, and tensor
count relationships (`torchlens/validation/invariants.py:710`).

Contract:

- Backend-neutral: label uniqueness, output presence, non-negative capture timing, saved count
  consistency.
- Mode-specific: param totals depend on `param_source`.
  `native-module` uses current module-derived params; `pytree-derived` uses declared JAX param
  leaves; `none` requires zero param totals.
- Torch behavior remains unchanged.

### T. Backward Graph Invariants

Current check: grad-fn label grammar, root ids, call ordinals, backward event flow, forward
op-to-grad-fn backpointers, saved grad accessors, and backward pass density
(`torchlens/validation/invariants.py:201`, `torchlens/validation/invariants.py:550`).
Current public methods reject MLX backward capture by hard-coded backend branch
(`torchlens/data_classes/trace.py:5658`, `torchlens/data_classes/trace.py:5697`).

Contract:

- Torch keeps the current check unchanged.
- JAX M1 has no true backward graph. All grad-fn/backward-pass structures must be inert-empty:
  `has_backward_pass=False`, no `grad_fn_logs`, no `backward_pass_logs`, no root grad-fn ids.
- JAX derived gradients are not a backward graph and must live in `trace.derived_grads` plus
  mirrored param-leaf grad slots where unambiguous. They must not satisfy or fake T.
- Backend dispatch must replace hard-coded MLX branches with backend capability errors.

### B/C/P/M/O. Graph Shape, Edges, Ordering, and Reachability

Current checks:

- special layer lists match per-op flags (`torchlens/validation/invariants.py:814`);
- parent-child edges are bidirectional and `out_versions_by_child` keys are child labels
  (`torchlens/validation/invariants.py:859`);
- non-source computational nodes have parents and orphans are pruned
  (`torchlens/validation/invariants.py:2671`);
- raw indices are unique/monotone, step indices are unique, parents precede children, raw labels
  do not survive (`torchlens/validation/invariants.py:2316`);
- distance metadata is internally consistent when enabled
  (`torchlens/validation/invariants.py:2569`).

Current construction:

- torch capture records source tensors before forward under active logging
  (`torchlens/capture/trace.py:663`);
- postprocess materializes event streams, adds output nodes, removes orphans, maps raw labels to
  final labels, builds lookup keys, and marks tracing finished
  (`torchlens/postprocess/__init__.py:210`, `torchlens/postprocess/__init__.py:240`,
  `torchlens/postprocess/__init__.py:291`, `torchlens/postprocess/__init__.py:331`).

Contract:

- Backend-neutral: a trace graph is a finite DAG in postprocessed order, with explicit input,
  output, internal-source, internal-sink, and buffer/pseudo-buffer roles.
- JAX M1 jaxpr equations are already topologically ordered; raw_index follows equation order
  after input/const nodes and before output nodes.
- JAX hidden closed-over consts must become explicit source records or rejected declared-leaf
  errors. Ambient state is not allowed to appear as parentless compute.
- `step_index` uniqueness remains required for compute equations. Synthetic source/output nodes
  may keep bookkeeping step semantics equivalent to torch.

### D/S. Op Field and Function-Call Identity

Current checks:

- saved tensor shape/dtype match payload; pass labels and module depth are consistent; real
  compute ops need callable `func` and non-empty `func_name`
  (`torchlens/validation/invariants.py:956`);
- intervention-ready logs require `func_call_id` groups with consistent metadata and unique
  `container_path` (`torchlens/validation/invariants.py:136`).

Contract:

- Backend-neutral: shape, dtype, label grammar, container_path uniqueness, output index, and
  call/equation group consistency.
- Torch keeps callable `func` replay requirement.
- JAX compute ops do not have Python callables for replay. The replacement required fields are
  `primitive_name`, versioned primitive table id, jaxpr equation index, params digest, input avals,
  output avals, effects classification, and optional provenance breadcrumb.
- Invariant S generalizes from `func_call_id` to `operation_group_id`; torch maps it to current
  `func_call_id`, JAX maps multi-output equations to one equation group.

### E/N/L. Recurrence, Equivalence, and Loop Detection

Current checks:

- `is_recurrent`, `max_layer_op_count`, `layer_num_calls`, and `Layer.ops` are mutually
  consistent (`torchlens/validation/invariants.py:1069`);
- same-layer recurrent groups require self-inclusion, symmetry, shared label/equivalence/func,
  dense pass numbering, and param-sharing consistency
  (`torchlens/validation/invariants.py:2388`);
- equivalence-class maps reference valid op labels (`torchlens/validation/invariants.py:2270`).

Current construction:

- postprocess either runs full loop detection or parameter-only grouping
  (`torchlens/postprocess/__init__.py:266`);
- loop detection states the same function plus same params rule
  (`torchlens/postprocess/loop_detection.py:825`).

Contract:

- Torch keeps current recurrence/loop semantics.
- JAX M1 rejects nested-jaxpr control-flow equations (`scan`, `cond`, `while`, `remat`,
  `pjit`, `shard_map`, `custom_jvp`, `custom_vjp`) before trace materialization, so recurrence is
  expected false unless repeated identical top-level equations are later deliberately grouped.
- JAX equivalence class format is backend-owned but must be stable, include primitive identity,
  output index, aval shape/dtype, and module/function-root suffix where applicable.
- No backend may collapse distinct output leaves from a multi-output op into one logical layer.

### F/F2. Branching and Conditional Metadata

Current checks:

- `is_branching` reflects fan-out (`torchlens/validation/invariants.py:1130`);
- conditional metadata has explicit arm-entry edge, projected child-view, valid child-label,
  branch-stack prefix, and legacy-view consistency checks
  (`torchlens/validation/invariants.py:1361`).

Contract:

- Fan-out branching remains backend-neutral.
- Torch keeps Python conditional metadata.
- JAX M1 rejects nested-jaxpr control flow, so conditional-specific structures must be inert-empty
  and `has_conditional_branching` false. A rejected nested-jaxpr fixture is mandatory.
- Future boundary-op support must add conditional/loop invariants before exposing those ops as
  successful captures.

### G/R. Layer/Op Cross-References and Lookup Keys

Current checks:

- aggregate `Layer` keys, `Layer.ops`, pass indices, and child `Op.layer_label` are consistent
  (`torchlens/validation/invariants.py:1908`);
- lookup forward/reverse maps and raw/final label maps are bidirectional
  (`torchlens/validation/invariants.py:2822`).

Contract:

- Backend-neutral: accessors must resolve by final op label, aggregate layer label, index, and
  pass notation where the backend has multi-pass entries.
- Torch accessors remain byte-stable.
- JAX `function_root` may have no recurrent multi-pass layers in M1, but the maps must still be
  populated so `trace.ops`, `trace.layers`, `trace.modules`, and `trace.params` do not raise for
  supported access patterns.

### H/I/Q. Module Identity and Containment

Current checks:

- module layer containment is bidirectional (`torchlens/validation/invariants.py:1954`);
- module hierarchy requires root `self`, address parent/child consistency, dense call indices,
  and valid call parents/children (`torchlens/validation/invariants.py:2050`);
- module address chains are acyclic; address depth and per-layer module stacks are valid
  (`torchlens/validation/invariants.py:2718`).

Current construction:

- module logs are built late in postprocess (`torchlens/postprocess/__init__.py:323`);
- MLX smoke finishing sets layer/module lookup structures directly and marks tracing finished
  (`torchlens/backends/mlx/backend.py:820`).

Contract:

- `torch_module`: current checks unchanged.
- `function_root`: build a real root Module record at `self` with `kind="function_root"`;
  address tree contains only root unless optional provenance creates documented pseudo-scopes.
  Module containment checks reduce to root existence, acyclicity, and every compute op belonging
  to root or no module according to a single documented rule.
- `pytree_module`: address hierarchy follows pytree paths; the implementation must define
  escaping for dict keys and non-string path components before enabling this mode.

### J/K. Params and Buffers

Current checks:

- Param used-by op/layer labels are valid; `uses_params=True` implies `_param_logs`
  (`torchlens/validation/invariants.py:2156`);
- buffer labels and buffer address ancestry are valid (`torchlens/validation/invariants.py:2216`).

Current defaults and accessors:

- `Trace.params` returns `param_logs` as an accessor (`torchlens/data_classes/trace.py:3894`);
- default trace fields include param hashes and backend metadata
  (`torchlens/data_classes/trace.py:138`, `torchlens/data_classes/trace.py:1396`).

Contract:

- `Trace.params` remains an accessor and must never become a raising property.
- JAX with declared `params=` builds pytree-derived Param records:
  `address_kind="pytree_path"`, neutral dtype/device refs, used-by links to consuming equations,
  and `trace.param_source="pytree-derived"`.
- JAX without params sets `trace.param_source="none"` and exposes an empty accessor.
- Buffers are torch-module specific in M1 JAX and must be inert-empty unless a future backend
  defines state leaves.

### Serialization and Loaded-Trace Invariants

Current behavior:

- Trace default `tlspec_version` is `_io.TLSPEC_VERSION` through the model-log default fill
  (`torchlens/data_classes/trace.py:221`).
- Portable state keeps `backend` and `tlspec_version`
  (`torchlens/data_classes/trace.py:1028`).
- Bundle validation rejects unresolved executable callables for replay
  (`torchlens/validation/core.py:75`).

Contract:

- Loaded traces must run metadata invariants that do not require executable payload replay.
- Replay validation after load is backend-conditional:
  torch requires resolved callables, JAX requires versioned primitive interpreter support and
  concrete payloads, MLX/tinygrad require their backend policy to opt in.
- Artifact 3 owns the three version axes; this artifact only requires invariants to name which
  axis they depend on when implemented.

## Per-Item Decisions

1. `validate_forward_pass()` becomes backend-dispatched before JAX M1; torch behavior is the
   compatibility baseline.
2. The invariant contract covers both replay/perturbation and metadata checks.
3. Module identity is explicit on every trace via `module_identity_mode`.
4. JAX M1 ships `function_root` only; `pytree_module` is deferred to Equinox.
5. JAX true backward graph invariants are inert-empty; derived gradients are separate data.
6. `Trace.params` stays accessor-shaped for all backends.
7. JAX compute replay uses jaxpr equation interpreter data, not Python `func` callables.
8. Nested-jaxpr control flow is rejected in M1, not represented as partial conditional metadata.
9. A backend without concrete payload replay must raise unsupported validation, never return a
   metadata-only `True`.

## Open Questions for the Impact Gate

1. Exact field names for JAX op records: should they extend `Op` directly or live under a
   `backend_semantics`/`backend_payload` subrecord?
2. Exact `module_identity_mode` spelling and whether it is public or an internal enum surfaced
   through docs only.
3. Whether `operation_group_id` replaces `func_call_id` or is added in parallel with a torch
   alias during migration.
4. How strict JAX numeric replay tolerance should be for primitive replay across devices and x64
   settings.
5. Whether loaded JAX `.tlspec` traces may validate replay if primitive interpreter tables match,
   or whether v1 non-torch payloads remain audit-only until a later schema.
6. How to represent pytree path components in Param/Module addresses without colliding with torch
   dotted module addresses.

## Fixture/Test Inventory for Implementation Round

1. Torch invariant parity: current smoke traces still pass every A-T/S check unchanged.
2. Validation dispatch: fake backend trace proves `Trace.validate_forward_pass()` routes by
   `trace.backend`.
3. JAX output equality: pytree multi-output function maps leaves to stable output nodes.
4. JAX equation replay: a simple MLP jaxpr validates by per-equation replay.
5. JAX parent perturbation: corrupt one equation parent while preserving final output shape;
   validation fails.
6. Dropped-site adversary: remove a parent/site attribution while final output still matches;
   validation fails.
7. JAX hidden const: closed-over const becomes explicit declared source or capture rejects.
8. JAX nested-jaxpr rejection: `scan`, `cond`, `while`, and `custom_vjp` fixtures fail with
   actionable errors before trace materialization.
9. JAX effects rejection: callback, Ref, FFI, donation, and custom-derivative effect fixtures.
10. JAX function-root module mode: root `self` exists, accessors work, no torch-module-only
    hierarchy assertion fires.
11. JAX params declared: pytree param leaves populate `Trace.params`, used-by links, counts, and
    optional derived grad slots.
12. JAX params omitted: empty accessor, `param_source="none"`, zero param totals.
13. Backward inert-empty: JAX trace has no grad-fn/backward-pass records and backward accessors are
    documented empty/unsupported as appropriate.
14. MLX/tinygrad unsupported validation path: metadata-only backend cannot return `True`.
15. Loaded-trace replay policy: portable bundle without replay capability raises the backend's
    documented unsupported error.
