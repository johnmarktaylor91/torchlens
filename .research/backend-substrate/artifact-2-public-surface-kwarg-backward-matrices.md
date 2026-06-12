# M0.1a Artifact 2: Public-Surface, Kwarg, and Backward-Surface Matrices

Date: 2026-06-12
Plan source: `.research/jax-tinygrad-sprint_PLAN.md` v13, M0.1a artifact 2.
Scope: design artifact only. No code, tests, ruff, pytest, or benchmark work.

## Purpose

This artifact defines the executable matrices that must govern public backend dispatch for
`trace()`, validation, accessors, and backward-adjacent APIs before JAX/tinygrad substrate work
lands. It is grounded in the current torch/MLX code paths:

- `trace()` has no public `backend=` kwarg today; it autoroutes first, branches explicitly for
  MLX module instances, then requires `nn.Module` for torch capture
  (`torchlens/user_funcs.py:1383`, `torchlens/user_funcs.py:1602`,
  `torchlens/user_funcs.py:1664`, `torchlens/user_funcs.py:1731`).
- Existing `Trace.backend` is typed and initialized as `Literal["torch", "mlx"]`
  (`torchlens/data_classes/trace.py:1007`, `torchlens/data_classes/trace.py:1351`).
- Public option names are centralized in grouped option maps and dataclasses
  (`torchlens/options.py:33`, `torchlens/options.py:69`, `torchlens/options.py:127`,
  `torchlens/options.py:553`, `torchlens/options.py:858`, `torchlens/options.py:1440`).
- Backward graph methods currently dispatch only by a hard-coded MLX rejection before importing
  torch backward capture (`torchlens/data_classes/trace.py:5658`,
  `torchlens/data_classes/trace.py:5697`).

Implementation contract: M0.2 must add public `backend=` resolution through `BackendSpec` while
preserving torch behavior byte-for-byte when `backend` is omitted or `"torch"` and the input is a
torch `nn.Module`.

## Public Surface Matrix

| Surface | Current code | Torch contract | JAX M1 contract | tinygrad/MLX contract |
|---|---|---|---|---|
| `tl.trace(model, input_args, ..., backend=None)` | No `backend=` kwarg; signature starts at `model`, `input_args`, `input_kwargs` (`torchlens/user_funcs.py:1383`). | Add kwarg defaulting to `None`; `None` preserves autoroute plus torch fallback. Explicit `"torch"` requires torch-compatible model/input. | Explicit `"jax"` routes to jaxpr-first backend and accepts function-root callables, not only `nn.Module`. `None` may autoroute to JAX only after a documented detector is registered. | Existing MLX implicit branch becomes a `BackendSpec` route. tinygrad explicit route added only after S0.G decision. |
| `tl.record()` | One-shot fastlog API is torch `nn.Module` only (`torchlens/fastlog/_record_one_shot.py:51`, `torchlens/fastlog/_record_one_shot.py:127`). | Unchanged. | Rejected: plan v13 says `tl.record()` torch-only. Error must say use `tl.trace(..., backend="jax")`. | tinygrad rejected unless later plan explicitly funds fastlog support. |
| `tl.validate(..., scope=...)` | Consolidated validator requires `scope` and dispatches to forward/backward/intervention functions without backend kwarg (`torchlens/validation/consolidated.py:203`, `torchlens/validation/consolidated.py:274`, `torchlens/validation/consolidated.py:290`). | Add `backend=None` and keep torch results unchanged. | `scope="forward"` and `"saved"` call JAX backend validation and return real bools. `scope="backward"` raises unsupported; derived gradients are not backward validation. | MLX/tinygrad forward validation raises unsupported unless full payload replay exists. |
| `tl.validate_forward_pass()` / deprecated top-level wrapper | User function hard-runs torch model, full-save trace, then `Trace.validate_forward_pass()` (`torchlens/user_funcs.py:2923`, `torchlens/user_funcs.py:3020`, `torchlens/user_funcs.py:3033`). `torchlens.__init__` wrapper delegates to consolidated validate (`torchlens/__init__.py:630`). | Add `backend=None`; existing torch code remains default implementation. | Ground-truth and trace both run through JAX backend; force full-save equivalents. | Must route through backend validation or raise unsupported, never return metadata-only success. |
| `Trace.validate_forward_pass()` | Directly imports torch replay validator (`torchlens/data_classes/trace.py:5333`, `torchlens/data_classes/trace.py:5352`). | Dispatch via `BackendSpec.validate` when `trace.backend == "torch"` to same implementation. | Dispatch to jaxpr equation replay + perturbation, returning bool. | Raise capability error unless backend supports payload replay. |
| `show_model_graph()` / `Trace.draw()` | `show_model_graph()` requires torch-like model/input and calls `_run_model_and_save_specified_outs` (`torchlens/user_funcs.py:2255`, `torchlens/user_funcs.py:2388`). | Unchanged. | Add explicit backend route only after JAX trace construction supports forward graph draw. Raw JAX M1 may render from `Trace.draw()` if a trace already exists. | Existing MLX graph route moves through registry if currently supported; tinygrad blocked until source-of-truth decision. |
| `draw_backward()` / `Trace.draw_backward()` | Top-level wrapper expects existing trace (`torchlens/user_funcs.py:2416`); backward trace methods exist on `Trace`. | Unchanged for torch backward traces. | Unsupported because JAX M1 has no true backward graph. | Unsupported unless backend has true backward graph capture. |
| `Trace.params` | Always returns `ParamAccessor` over `param_logs` (`torchlens/data_classes/trace.py:3894`); `Trace.__init__` initializes empty accessor and param counters (`torchlens/data_classes/trace.py:1535`). | Existing native-module params unchanged. | Never raise. If `params=` is declared, populate pytree-derived param-leaf records and set `param_source="pytree-derived"`; otherwise empty accessor and `param_source="none"`. | Backend-specific accessor population through `BackendSpec`; no trap properties. |
| `Trace.modules`, `module_calls`, `root_module` | Return module accessors/calls and root `"self"` today (`torchlens/data_classes/trace.py:3936`, `torchlens/data_classes/trace.py:3957`, `torchlens/data_classes/trace.py:4034`). | Existing torch module tree unchanged. | Function-root mode must expose a real `"self"` root module/call record with `kind="function_root"`, but no fake framework submodules. | MLX/tinygrad module identity must declare mode and either build records or expose function-root. |
| `Trace.ops`, `layers`, `saved_ops` | Accessors derive from `layer_list`/`layer_logs` and saved flags (`torchlens/data_classes/trace.py:3899`, `torchlens/data_classes/trace.py:3923`, `torchlens/data_classes/trace.py:4265`). | Unchanged labels/accessors. | Populated from jaxpr source/input/equation/output records with stable labels and container paths. | Same backend-neutral accessor contract once traces exist. |
| `Trace.grad_fns`, `grad_fn_calls`, `backward_passes` | Accessors lazily sync torch backward projection (`torchlens/data_classes/trace.py:4210`, `torchlens/data_classes/trace.py:4215`, `torchlens/data_classes/trace.py:4227`). | Existing torch behavior unchanged. | Inert-empty accessors; `has_backward_pass=False`, counts zero, no lazy torch projection. | Empty or unsupported based on backend capability; no torch import side effects. |
| `Trace.save()` | Exists as trace method (`torchlens/data_classes/trace.py:2970`) and portable state keeps backend (`torchlens/data_classes/trace.py:1028`). | Existing save behavior unchanged. | Allowed subject to artifact 3 serialization contract; non-torch payloads audit-only until codec exists. | Same as backend serialization policy. |

## Trace Kwarg Matrix

The implementation must generate tests from this table for every public `trace()` kwarg in the
current signature (`torchlens/user_funcs.py:1383`). "Accept" means the JAX M1 backend honors the
option. "Reject" means explicit non-default values raise an actionable backend unsupported error
before capture. "Ignore-default" means the default is accepted but has no behavior in function-root
JAX M1.

| Kwarg/group | Current source | JAX M1 disposition | Contract detail |
|---|---|---|---|
| `backend` | New kwarg; absent today (`torchlens/user_funcs.py:1383`). | Accept. | `None`, `"torch"`, `"mlx"`, `"jax"` literals. Explicit mismatch errors must name detected backend and requested backend. |
| `model`, `input_args`, `input_kwargs` | Torch signature requires `nn.Module` and input coercion (`torchlens/user_funcs.py:1383`, `torchlens/user_funcs.py:1731`, `torchlens/user_funcs.py:1788`). | Accept with JAX callable contract. | JAX accepts declared callable plus positional/keyword leaves/statics per JAX backend options. Hidden closed-over consts must be inventoried/rejected per artifact 1. |
| `capture` / `CaptureOptions` | Group fields listed in `_CAPTURE_FIELDS` (`torchlens/options.py:33`). | Partially accept. | JAX backend must inspect explicit fields through `is_field_explicit()` (`torchlens/options.py:841`) and apply this matrix. |
| `layers_to_save` | Default `"all"` (`torchlens/options.py:640`); old selective save drives torch two-pass capture (`torchlens/user_funcs.py:1455`). | Reject non-full save. | Plan v13 says save semantics v1 is full save only. Accept omitted/`"all"`/equivalent full-save options. Reject `"none"`, lists, predicate-shaped selective capture. |
| `save` predicate / selector / `SaveOptions` | Split into grouped options vs predicate (`torchlens/user_funcs.py:1360`, `torchlens/user_funcs.py:1791`). | Reject predicate/selector save; accept only transform-free full-save options. | `SaveOptions` carrying transforms also follows transform rows below. |
| `transform` | Input preprocessor callable (`torchlens/options.py:641`, `torchlens/user_funcs.py:1777`). | Reject for M1 unless S0.J proves exact static/input normalization. | Conservative: jaxpr declared-call contract should receive already-normalized inputs; arbitrary Python preprocessing changes call identity. |
| `save_raw_input`, `batch_render`, `output_transform`, `save_raw_output` | Capture fields/defaults (`torchlens/options.py:642`, `torchlens/options.py:643`, `torchlens/options.py:644`, `torchlens/options.py:645`). | Accept metadata-only policies; reject callable `output_transform`. | Raw input/output storage is serialization/docs scope; `output_transform` callable is non-portable and not part of jaxpr graph. |
| `keep_orphans` | Default false and postprocess behavior (`torchlens/options.py:648`). | Ignore-default; reject `True`. | JAX jaxpr graph is complete by construction; orphan retention is torch wrapper debris semantics. |
| `output_device` | Validated as `"same"`, `"cpu"`, `"cuda"` (`torchlens/user_funcs.py:1878`). | Reject non-default. | JAX device transfer policy belongs to JAX backend options after S0.J; do not reuse torch device literals. |
| `activation_transform`, `save_raw_activations`, `save_mode` | Save options and save-mode validation (`torchlens/options.py:886`, `torchlens/options.py:889`, `torchlens/user_funcs.py:1839`). | Reject non-default. | Full-save JAX v1 captures concrete equation outputs; transformed/drop/raw save-shaping is explicitly out of v1. |
| `save_arg_values` | Default false, needed for torch replay (`torchlens/options.py:650`, `torchlens/user_funcs.py:1510`). | Accept but force true-equivalent internally. | Validation captures force saved equation inputs/params/avals. Explicit `False` cannot drop data needed for validation; either ignored with recorded normalized setting or rejected if it would shape capture. |
| `save_rng_states`, `random_seed` | Operation RNG capture and random seed fields (`torchlens/options.py:654`, `torchlens/options.py:655`). | Replace with explicit-key contract. | JAX stochasticity is allowed only when keys are explicit inputs/statics. `random_seed` and ambient RNG capture are rejected for JAX M1. |
| `save_code_context`, `source_context_lines`, `num_context_lines` | Source identity/context fields and alias handling (`torchlens/user_funcs.py:1515`, `torchlens/options.py:728`). | Accept as optional provenance enrichment only. | Absence of Python breadcrumbs never affects correctness. `num_context_lines` remains deprecated alias. |
| `reconstruction_ready` | Forces `save_arg_values` and RNG states (`torchlens/user_funcs.py:1734`). | Reject. | Torch semantic reconstruction is not JAX M1. |
| `save_grads`, `grad_transform`, `save_raw_gradients`, `capture_tensor_grad_hooks`, `backward_ready`, `optimizer` | Gradient/backward capture options (`torchlens/options.py:651`, `torchlens/options.py:652`, `torchlens/options.py:657`, `torchlens/options.py:665`; training validation in `torchlens/user_funcs.py:1882`). | Reject for backward capture; use JAX `GradOptions` for derived gradients. | `save_grads` must not silently mean `derived_grads`. Derived gradients are second-execution AD configured under `tl.backends.jax.GradOptions`. |
| `intervention_ready`, `hooks`, `intervene`, `halt`, `stop_after` | Intervention/halt options (`torchlens/options.py:661`, `torchlens/options.py:662`, `torchlens/user_funcs.py:1792`, `torchlens/user_funcs.py:1863`). | Reject non-default. | JAX M1 capture is jaxpr-first full trace, no live predicate-time mutation or early stop. |
| `lookback`, `lookback_payload_policy` | Predicate-window validation (`torchlens/user_funcs.py:1796`, `torchlens/user_funcs.py:1798`). | Reject non-default. | Predicate capture/save-shaping is follow-up; default `lookback=0`, `"metadata_only"` accepted. |
| `storage`, `streaming`, `save_outs_to`, `keep_outs_in_memory`, `out_sink` | Streaming options and aliases (`torchlens/options.py:127`, `torchlens/options.py:1440`, `torchlens/options.py:1476`, `torchlens/user_funcs.py:1816`). | Reject for M1. | Non-torch payload codec/offload is artifact 3; no disk streaming until backend serialization policy says yes. |
| `unwrap_when_done` | Torch wrapper cleanup option (`torchlens/options.py:663`). | Ignore-default; reject `True`. | JAX jaxpr-first capture installs no torch wrappers. |
| `verbose`, `name`, `profile` | Metadata/progress fields (`torchlens/options.py:664`, `torchlens/options.py:666`, `torchlens/user_funcs.py:1572`, `torchlens/user_funcs.py:1581`). | Accept. | These do not shape graph semantics. |
| `cache`, `cache_dir` | Capture cache fields (`torchlens/options.py:667`, `torchlens/options.py:668`). | Reject non-default. | JAX jaxpr/config/device fingerprints need separate cache key design. |
| `module_filter` | Skips out saving for torch ops (`torchlens/options.py:669`, `torchlens/user_funcs.py:1578`). | Reject. | Full save only; module identity in function-root mode is not a torch module filter. |
| `compute_input_output_distances`, `mark_layer_depths` | Default true and deprecated alias (`torchlens/options.py:658`, `torchlens/options.py:720`). | Accept. | Backend-neutral graph distance metadata can be computed post-materialization. Alias behavior unchanged. |
| `recurrence_detection` | Default true (`torchlens/options.py:660`). | Accept but inert/no-op for M1. | Nested-jaxpr control flow is rejected; repeated top-level equations are not loop-expanded in M1. |
| `raise_on_nan` | Capture field (`torchlens/options.py:672`). | Accept if checked on concrete equation outputs. | Must run after async barrier. |
| `recipes` | Trace-time facet recipes in signature (`torchlens/user_funcs.py:1437`, `torchlens/user_funcs.py:1583`). | Reject non-default. | Existing recipes assume torch semantic facets. |

## Backward Surface Matrix

The plan's amendment is that JAX M1 has "derived-gradient preview" only. It must not present
grad-fn logs, backward passes, or saved op gradients as true backward capture.

| Surface | Current code | JAX M1 behavior | Generated test expectation |
|---|---|---|---|
| `Trace.log_backward(loss, **kwargs)` | Torch implementation after MLX hard rejection (`torchlens/data_classes/trace.py:5658`). | Raise backend unsupported. | Calling on JAX trace raises; no mutation of backward fields. |
| `Trace.backward(...)` | Alias to `log_backward` (`torchlens/data_classes/trace.py:5679`). | Raise same unsupported error. | Same as `log_backward`. |
| `Trace.recording_backward()` | Torch context after MLX hard rejection (`torchlens/data_classes/trace.py:5697`). | Raise backend unsupported. | No context manager returned. |
| `tl.validate(..., scope="backward")` / `validate_backward_pass()` | Consolidated dispatch to backward validator (`torchlens/validation/consolidated.py:274`); legacy wrapper exists (`torchlens/__init__.py:659`). | Raise unsupported for JAX M1. | Error explains use derived-gradient validation, not backward capture. |
| `draw_backward()` / `Trace.draw_backward()` | Backward graph visualization surfaces (`torchlens/user_funcs.py:2416`, `torchlens/data_classes/trace.py:4854`). | Raise unsupported when `trace.backend == "jax"` or when `has_backward_pass=False`. | No empty backward image generated. |
| `has_backward_pass`, `num_backward_passes`, `backward_pass_logs`, `backward_root_grad_fn_object_ids` | Initialized false/empty/zero (`torchlens/data_classes/trace.py:1557`, `torchlens/data_classes/trace.py:1560`, `torchlens/data_classes/trace.py:1563`, `torchlens/data_classes/trace.py:1565`). | Inert-empty. | Values are false, zero, empty accessors. |
| `grad_fns`, `grad_fn_calls`, `backward_passes`, `last_backward_pass` | Accessors sync torch projection (`torchlens/data_classes/trace.py:4210`, `torchlens/data_classes/trace.py:4215`, `torchlens/data_classes/trace.py:4227`, `torchlens/data_classes/trace.py:4234`). | Inert-empty, no torch projection import/effect. | Length zero; `last_backward_pass is None`. |
| `num_backward_edges` | Returns `None` when no backward pass (`torchlens/data_classes/trace.py:4108`). | Inert `None`. | Exactly `None`, not zero. |
| `saved_grad_ops`, `saved_grad_layers`, `saved_grad_module_calls`, `saved_grad_modules`, `saved_grad_fns`, counts | Filter saved-gradient flags/logs (`torchlens/data_classes/trace.py:4150`, `torchlens/data_classes/trace.py:4273`, `torchlens/data_classes/trace.py:4297`, `torchlens/data_classes/trace.py:4343`, `torchlens/data_classes/trace.py:4353`, `torchlens/data_classes/trace.py:4379`, `torchlens/data_classes/trace.py:4391`). | Inert-empty. | Accessors empty; counts zero. |
| `Op.grad`, `Op.has_grad`, `Op.grads`, `Op.grad_for()` | Grad fields persisted in `Op.PORTABLE_STATE_SPEC` (`torchlens/data_classes/op.py:835`) and per-pass gradient accessor exists (`torchlens/data_classes/op.py:1461`). | `grad=None`, `has_grad=False`, empty records; `grad_for` raises current missing-payload style. | No derived gradient is placed here. |
| `Layer.grad`, `Layer.has_grad` | Single-pass delegation to op grad fields (`torchlens/data_classes/layer.py:740`, `torchlens/data_classes/layer.py:757`). | Mirrors inert op fields. | `has_grad=False`; `grad is None` or existing single-pass empty behavior. |
| `Param.grad`, `Param.grads` | Live torch parameter grad and accumulating gradient records (`torchlens/data_classes/param.py:265`, `torchlens/data_classes/param.py:291`). | If `params=` and derived grads requested, mirror unambiguous derived leaf gradient into param record's documented derived-grad slot or `grad` only if the implementation chooses that compatibility path and labels `param_source`. | Test distinguishes true backward records from derived gradients. |
| `trace.derived_grads` | New field. | Populated only by JAX `GradOptions`; includes params and selected input leaves plus fingerprint metadata. | Exists on JAX derived-gradient traces; absent/empty does not imply backward pass. |

## Per-Item Decisions

1. Public `backend=` is additive and default-`None`; torch default behavior remains unchanged.
2. JAX M1 supports `tl.trace(..., backend="jax")` and backend validation, not `tl.record()`.
3. JAX M1 is full-save only. Every save-shaping kwarg that drops/selects/transforms payloads is
   rejected when explicit.
4. JAX stochastic capture uses explicit PRNG keys only; torch-style `random_seed` and
   `save_rng_states` are rejected for JAX M1.
5. Function-root traces must expose real accessors (`ops`, `layers`, `params`, `modules`) instead
   of raising backend-specific traps.
6. JAX derived gradients are separate from true backward capture; all grad-fn/backward graph
   surfaces stay unsupported or inert-empty.

## Open Questions For The Impact Gate

1. Should `backend=None` autoroute to JAX from callable/input shape in M1, or require explicit
   `backend="jax"` until after docs and compatibility soak?
2. Is `transform=` worth supporting by normalizing the declared JAX callable contract in S0.J, or
   should callers always pass already-normalized JAX arguments?
3. Should JAX derived param gradients mirror into existing `Param.grad`, or only into a new
   derived-gradient field with summaries displaying it?
4. Which JAX backend options object owns `params=`, `loss_fn=`, statics, and
   `input_grad_argnums`, and does it live only at `tl.backends.jax.GradOptions` or also as a
   `trace(..., grad=...)` convenience?
5. Should `save_code_context=True` enrichment ship in M1 or slip without blocking correctness?

## Fixture/Test Inventory For Implementation

- Generated public API signature test: `trace()` accepts `backend=` and all existing kwargs remain
  present with torch defaults unchanged.
- Backend resolution tests: explicit `"torch"`/`"jax"`/`"mlx"` mismatch errors; `None` preserves
  existing torch and MLX routing.
- Generated JAX kwarg matrix tests: each row above exercises omitted/default, accepted explicit,
  and rejected explicit forms.
- Full-save refusal tests: `layers_to_save="none"`, list save, predicate `save=`, `module_filter`,
  streaming, transforms, and lookback payload policies all fail before capture.
- JAX validation bool tests: output equality, per-equation replay, parent perturbation, and
  final-output-matches-despite-dropped-parent adversarial fixture.
- Backward unsupported tests: `log_backward`, `backward`, `recording_backward`,
  `validate(scope="backward")`, and `draw_backward` raise without mutating trace state.
- Inert-empty backward accessor tests: grad-fn, backward-pass, saved-grad, and count surfaces match
  this matrix on a JAX trace.
- Derived-gradient separation tests: `trace.derived_grads` can be populated while
  `has_backward_pass=False`, `saved_grad_ops` empty, and `num_backward_edges is None`.
- Function-root accessor tests: JAX trace has root module `"self"` with function-root kind,
  populated ops/layers, and non-raising params accessor with `param_source`.
