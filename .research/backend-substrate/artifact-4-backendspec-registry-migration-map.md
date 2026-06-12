# M0.1a Artifact 4: BackendSpec Registry Contract As Executable Migration Map

Date: 2026-06-12
Plan source: `.research/jax-tinygrad-sprint_PLAN.md` v13, M0.1a artifact 4.
Scope: design artifact only. No code, tests, ruff, pytest, or benchmark work.

## Purpose

This artifact defines the `BackendSpec` registry contract and the migration map that M0.2 must
turn into executable tests. The plan's requirement is strict: every backend literal, MLX branch,
validation dispatch, save/load path, and public accessor either routes through `BackendSpec` or has
a documented temporary exception.

Current code has a useful but incomplete backend substrate:

- `CaptureBackend` is a capture-only protocol with methods for wrapper/session/tensor-event work
  (`torchlens/backends/_protocol.py:15`, `torchlens/backends/_protocol.py:18`,
  `torchlens/backends/_protocol.py:157`, `torchlens/backends/_protocol.py:168`).
- Torch and MLX adapters already implement capture-level names and capabilities
  (`torchlens/backends/torch/backend.py:33`, `torchlens/backends/torch/backend.py:36`,
  `torchlens/backends/torch/backend.py:37`, `torchlens/backends/mlx/backend.py:36`,
  `torchlens/backends/mlx/backend.py:39`, `torchlens/backends/mlx/backend.py:40`).
- Public routing still bypasses any registry: `trace()` imports autoroute, then hard-branches on
  MLX, then falls through to torch `nn.Module` capture (`torchlens/user_funcs.py:1602`,
  `torchlens/user_funcs.py:1658`, `torchlens/user_funcs.py:1664`,
  `torchlens/user_funcs.py:1731`).

The contract is: `CaptureBackend` stays as the lower-level execution adapter, but the public
backend boundary is a new registry of `BackendSpec` objects that owns resolution, capability
errors, validation dispatch, serialization policy, and public-surface routing.

## BackendSpec Shape

`BackendSpec` must be a typed object, not a dictionary, with these fields:

| Field | Type | Required behavior |
|---|---|---|
| `name` | `BackendName` | Literal-compatible string. Initial names: `"torch"`, `"mlx"`, `"jax"`, later `"tinygrad"`. |
| `can_handle` | callable | Detector for `backend=None` only. Must be side-effect-light and must not import optional heavy runtimes unless needed for the detector. |
| `capture_trace` | callable | Public `tl.trace()` implementation for the backend. May delegate to `CaptureBackend` for wrapper-based backends or to a jaxpr builder for JAX. |
| `validate_entry` | callable | Public model/input validation entry. Returns real bools or raises typed unsupported; no metadata-only success. |
| `validate_trace` | callable | `Trace.validate_forward_pass()` implementation for already-built traces. |
| `backward` | object/callables | Capability-backed `log_backward`, `recording_backward`, `draw_backward`, and backward validation policy. |
| `serialization_policy` | object/callables | Save/load schema policy, payload codec/audit-only policy, runtime compatibility check, and access-after-load behavior. |
| `capabilities` | dataclass | Single source for booleans and modes: backward capture, validation replay, fastlog, interventions, RNG replay, payload materialization, streaming, module identity modes, save levels. |
| `canonical_errors` | object/callables | Creates stable, testable exceptions for mismatch and unsupported surfaces. |

The existing MLX capability constants are the first migration input, not the final shape:
`supports_backward_capture`, `supports_fastlog`, `supports_intervention`, `supports_rng_replay`,
and `supports_compile_capture` live in `torchlens/backends/mlx/capabilities.py:5` through
`torchlens/backends/mlx/capabilities.py:9`. M0.2 must consolidate those into
`BackendSpec.capabilities` and stop reading backend support from scattered module globals.

## Registry Resolution Contract

Resolution order:

1. Explicit `backend=` wins. Unknown names fail before capture. A detected/requested mismatch
   fails before capture unless the backend spec marks the input as coercible.
2. `backend=None` preserves current behavior: autoroute detectors run first, MLX module instances
   can still route to MLX, and torch `nn.Module` remains the default fallback.
3. JAX M1 should require explicit `backend="jax"` until the impact gate decides whether callable
   autorouting is acceptable.
4. Collision tests are mandatory: two `can_handle()` hits under `backend=None` fail with a
   deterministic ambiguity error unless one spec has an explicit priority rule.

The existing `trace()` autoroute loop (`torchlens/user_funcs.py:1602`,
`torchlens/user_funcs.py:1658`) becomes either a `BackendSpec` detector phase or a documented
temporary pre-registry exception. The MLX detector `_is_mlx_module_instance()` currently imports
`mlx.nn` lazily and checks `mlx.nn.Module` (`torchlens/user_funcs.py:132`,
`torchlens/user_funcs.py:146`, `torchlens/user_funcs.py:150`); that detector must move to
`MLXSpec.can_handle`.

## Executable Migration Map

| Current site | Current behavior | Required route | Temporary exception |
|---|---|---|---|
| `torchlens/user_funcs.py:1383` | `trace()` has no `backend=` kwarg and is typed as `nn.Module` input. | Add `backend: BackendName | None = None`; call `resolve_backend_spec()` before backend-specific option normalization. | None. This is the primary public API change. |
| `torchlens/user_funcs.py:1602` | Transform-omitted path invokes `torchlens.autoroute` before MLX/torch decisions. | Either wrap autoroute as a `BackendSpec` detector stage or run it inside the `torch` spec if it is truly torch-only. | Allowed only until M0.2 lands; tests must pin unchanged torch behavior. |
| `torchlens/user_funcs.py:1664` | Inline MLX module branch. | Replace with `spec = resolve_backend_spec(...); return spec.capture_trace(...)`. | None. |
| `torchlens/user_funcs.py:1665` through `torchlens/user_funcs.py:1684` | Inline MLX rejections for save predicates, interventions, and halt. | Move to `MLXSpec.validate_trace_options()` using canonical unsupported errors. | None. |
| `torchlens/user_funcs.py:1692` through `torchlens/user_funcs.py:1727` | Inline call to `_trace_mlx_model()`. | `MLXSpec.capture_trace()` owns this call or replaces `_trace_mlx_model()` entirely. | `_trace_mlx_model()` may remain private during the cutover but may not be called from public routing directly. |
| `torchlens/user_funcs.py:1728` through `torchlens/user_funcs.py:1732` | Torch fallback warns/unwraps and rejects non-`nn.Module`. | Move into `TorchSpec.capture_trace()` after resolution. | None for new path; exact torch error text should be golden-tested. |
| `torchlens/backends/_protocol.py:15` | Capture-only `CaptureBackend` protocol. | Keep as lower-level execution adapter referenced by `BackendSpec.capture_backend` for torch/MLX. | Explicitly documented as not the public registry. |
| `torchlens/backends/__init__.py:5` | Exports only `CaptureBackend`. | Export `BackendSpec`, `BackendName`, `get_backend_spec`, `resolve_backend_spec`, and installed specs. | None. |
| `torchlens/backends/mlx/backend.py:351` | MLX adapter exposes `capture_trace()` directly. | `MLXSpec.capture_trace()` validates public kwargs, then delegates to this method. | Direct import by tests may remain only in backend-private tests. |
| `torchlens/backends/mlx/backend.py:385` through `torchlens/backends/mlx/backend.py:388` | MLX capture raises ad hoc backward/output-device errors. | Convert to canonical capability errors via spec before capture. | Backend-internal defensive checks may remain but should be unreachable in public matrix tests. |
| `torchlens/backends/mlx/backend.py:420` | MLX writes `trace.backend` via a cast to `Literal["torch", "mlx"]`. | Use widened `BackendName`; spec sets `trace.backend = spec.name`. | None. |
| `torchlens/data_classes/trace.py:1007` | `Trace.backend` type is `Literal["torch", "mlx"]`. | Replace with `BackendName`; include `"jax"` and later `"tinygrad"`. | None. |
| `torchlens/data_classes/trace.py:1351` | New traces default `backend` to `"torch"`. | Keep torch default for direct constructor, but backend specs must set the concrete value. | Direct `Trace()` construction remains torch by default for compatibility. |
| `torchlens/validation/consolidated.py:203` | `validate()` has no `backend=` and takes `nn.Module`. | Add `backend=None`; resolve spec before scope dispatch. | None. |
| `torchlens/validation/consolidated.py:274` through `torchlens/validation/consolidated.py:300` | Backward calls torch backward validator; forward/saved imports user torch validator. | `spec.validate_entry(scope=...)` owns dispatch. Torch spec delegates to current implementations. | None. |
| `torchlens/user_funcs.py:2923` | `validate_forward_pass()` is torch-specific and has no backend kwarg. | Torch implementation moves behind `TorchSpec.validate_entry`; public wrapper resolves backend. | The body may remain as a private torch helper. |
| `torchlens/data_classes/trace.py:5333` through `torchlens/data_classes/trace.py:5359` | `Trace.validate_forward_pass()` imports torch replay validator directly. | Route through `get_backend_spec(self.backend).validate_trace(...)`. | None. |
| `torchlens/data_classes/trace.py:5658` through `torchlens/data_classes/trace.py:5677` | `log_backward()` hard-rejects only MLX, then imports torch implementation. | Route through `spec.backward.log_backward()` or spec capability error. | None. |
| `torchlens/data_classes/trace.py:5697` through `torchlens/data_classes/trace.py:5709` | `recording_backward()` hard-rejects only MLX, then imports torch implementation. | Route through `spec.backward.recording_backward()`. | None. |
| `torchlens/data_classes/trace.py:4243` through `torchlens/data_classes/trace.py:4256` | Lazy backward projection imports torch code whenever `backward_events` exists. | Gate on `spec.capabilities.backward_capture`; non-torch specs must never import torch backward projection. | None. |
| `torchlens/user_funcs.py:2255` through `torchlens/user_funcs.py:2388` | `show_model_graph()` runs torch capture directly. | Add `backend=None` or explicitly document as torch-only until JAX trace draw is funded. | Temporary exception allowed for M1 if `Trace.draw()` works on an existing JAX trace and docs say top-level `show_model_graph()` is torch-only. |
| `torchlens/user_funcs.py:2416` | Top-level `draw_backward()` accepts a trace. | Check trace backend via spec backward capability before drawing. | None. |
| `torchlens/fastlog/_record_one_shot.py:51` through `torchlens/fastlog/_record_one_shot.py:81` | `record()` is torch `nn.Module` fastlog only. | Keep as explicit `TorchSpec.capabilities.fastlog=True`; other specs return canonical unsupported. | Permanent plan exception: `tl.record()` is torch-only in v13. |
| `torchlens/data_classes/trace.py:2970` through `torchlens/data_classes/trace.py:2981` | `Trace.save()` calls bundle save directly. | `save()` consults `get_backend_spec(trace.backend).serialization_policy` before scrubbing/writing. | None. |
| `torchlens/_io/bundle.py:142` through `torchlens/_io/bundle.py:153` | `save()` accepts any `Trace` but assumes torch/safetensors payloads. | Resolve serialization policy from `trace.backend`; apply level/payload rules before blob loop. | None. |
| `torchlens/_io/bundle.py:258` through `torchlens/_io/bundle.py:270` | Blob loop calls torch tensor policy and writes safetensors. | Route payload handling through backend serialization policy; torch keeps current path. | None for schema v2; schema v1 torch loads remain legacy path. |
| `torchlens/_io/bundle.py:543` through `torchlens/_io/bundle.py:587` | Unified trace load calls `Manifest.from_dict()` before backend dispatch. | Read raw `backend` and `schema_version`, resolve spec, then call spec loader policy. | None for schema v2; v1 no-backend manifests infer torch. |
| `torchlens/_io/tlspec.py:280` through `torchlens/_io/tlspec.py:298` | Unified manifest always writes `torch_version` and `body_format="safetensors"`. | Manifest writer calls `spec.serialization_policy.manifest_fields(...)`. | None. |
| `torchlens/data_classes/trace.py:3895` through `torchlens/data_classes/trace.py:4037` | Accessors assume populated torch-style params/modules/root `"self"`. | Backend builders must populate accessor-compatible records; accessors stay backend-neutral and non-raising. | None. |
| `torchlens/data_classes/trace.py:4210` through `torchlens/data_classes/trace.py:4232` | Backward accessors call lazy torch projection. | Only torch spec may project; JAX/MLX/tinygrad return inert-empty state. | None. |
| `torchlens/data_classes/trace.py:4265` through `torchlens/data_classes/trace.py:4404` | Saved activation/grad accessors filter generic `Op`/module/grad-fn state. | Keep generic filtering; specs define which fields may be populated. | No exception needed if state is valid. |

Every row above must become either a generated migration test or a grep-style audit in M0.2. The
audit must fail on new public branches like `if backend == "mlx"` or `if trace.backend == "jax"`
outside the registry, tests, backend-private implementation modules, or documented compatibility
normalizers.

## Canonical Error Contract

Registry errors must be stable enough for generated tests but not overfit exact prose. Required
classes or typed error codes:

| Error | Required trigger |
|---|---|
| `UnknownBackendError` | Explicit `backend=` is not registered. |
| `BackendMismatchError` | Explicit backend cannot handle the supplied callable/model/input. |
| `BackendAmbiguityError` | `backend=None` yields multiple equal-priority specs. |
| `BackendUnsupportedError` | Registered backend lacks a requested capability or explicit kwarg. |
| `BackendPayloadUnsupportedError` | Loaded/audit-only payload cannot materialize. |
| `BackendRuntimeCompatibilityError` | Serialization/runtime policy rejects a manifest. |

Existing ad hoc errors such as MLX backward rejection (`torchlens/user_funcs.py:263`,
`torchlens/backends/mlx/backend.py:385`) and `Trace.log_backward()` MLX-only rejection
(`torchlens/data_classes/trace.py:5673`) must be replaced or wrapped by canonical errors.

## Serialization Policy Contract

`BackendSpec.serialization_policy` owns:

1. schema v1 torch compatibility inference;
2. schema v2 manifest field production;
3. runtime compatibility checks;
4. payload body format decisions;
5. materialization or typed audit-only refusal after load.

This is necessary because current load and save are torch-shaped:

- save level coercion and blob inclusion happen before any backend decision
  (`torchlens/_io/bundle.py:202`, `torchlens/_io/bundle.py:208`);
- the blob loop requires tensors accepted by `is_supported_for_save()` and writes safetensors
  (`torchlens/_io/bundle.py:258`, `torchlens/_io/bundle.py:262`);
- trace load enforces torch manifest policy before unpickling and rehydrating
  (`torchlens/_io/bundle.py:505`, `torchlens/_io/bundle.py:529`);
- unified trace load parses through the torch-shaped `Manifest.from_dict()`
  (`torchlens/_io/bundle.py:580`, `torchlens/_io/bundle.py:583`);
- unified manifest write always records `torch.__version__` and `"safetensors"`
  (`torchlens/_io/tlspec.py:286`, `torchlens/_io/tlspec.py:292`).

Artifact 3's three-axis serialization contract remains authoritative. This artifact adds only the
registry route: schema v2 non-torch manifests must never pass through torch-only compatibility or
payload materialization code before backend policy dispatch.

## Public Accessor Contract

Public accessors should stay simple and backend-neutral. The backend-specific work happens while
building trace state:

- `params`, `ops`, `layers`, `modules`, `module_calls`, and `root_module` are property accessors
  over stored trace collections (`torchlens/data_classes/trace.py:3895`,
  `torchlens/data_classes/trace.py:3900`, `torchlens/data_classes/trace.py:3924`,
  `torchlens/data_classes/trace.py:3937`, `torchlens/data_classes/trace.py:3958`,
  `torchlens/data_classes/trace.py:4035`).
- Backward surfaces have lazy torch projection today and must be capability-gated
  (`torchlens/data_classes/trace.py:4210`, `torchlens/data_classes/trace.py:4219`,
  `torchlens/data_classes/trace.py:4231`, `torchlens/data_classes/trace.py:4250`).
- Saved activation and gradient accessors can remain generic filters if backend builders populate
  honest fields (`torchlens/data_classes/trace.py:4265`, `torchlens/data_classes/trace.py:4274`,
  `torchlens/data_classes/trace.py:4298`, `torchlens/data_classes/trace.py:4379`).

JAX M1 therefore must build function-root module records, pytree-derived params when `params=` is
declared, inert-empty backward collections, and `derived_grads` outside the true backward graph.
The registry may expose builder helpers, but public accessors must not branch per backend except
for backward projection capability checks.

## Per-Item Decisions

1. `BackendSpec` is the public registry unit; existing `CaptureBackend` remains a lower-level
   execution adapter for wrapper-based capture.
2. `backend=` resolution happens before backend-specific option normalization in `trace()` and
   `validate()`.
3. `backend=None` preserves current torch/autoroute/MLX behavior; JAX M1 requires explicit
   `backend="jax"` unless the impact gate changes this.
4. `tl.record()` remains torch-only as a documented plan exception, surfaced through capabilities.
5. Serialization dispatch moves to `BackendSpec.serialization_policy` before non-torch schema v2
   manifests are parsed by torch-shaped `Manifest`.
6. Public accessors remain backend-neutral; backend builders must populate valid state or inert
   empty state instead of accessor traps.
7. New hard-coded backend branches outside registry/backends/tests/compat normalizers are
   migration failures.

## Open Questions For The Impact Gate

1. Should `backend=None` ever autoroute raw Python callables to JAX in M1, or must JAX stay explicit
   until after compatibility soak?
2. Should `BackendSpec` and `CaptureBackend` live in the same module, or should public registry
   types live in `torchlens/backends/registry.py` to keep capture protocol dependencies smaller?
3. What priority should MLX autorouting have relative to legacy `torchlens.autoroute` detectors
   when both could return a trace?
4. Should canonical backend errors be new exception classes or existing exception classes with
   stable `.code` attributes?
5. Does `show_model_graph(..., backend="jax")` need to ship with JAX M1, or is drawing existing
   JAX traces enough for the first public surface?

## Fixture/Test Inventory For Implementation

- Registry unit tests: registered names, unknown backend, duplicate registration, resolution
  order, explicit mismatch, and ambiguous `backend=None`.
- Public signature tests: `trace()` and consolidated `validate()` accept `backend=None` while torch
  defaults remain unchanged.
- Generated migration-map audit: every row in the executable migration map has either a behavioral
  test or a grep/audit assertion.
- Hard-coded branch audit: fail on backend literal branches outside registry modules,
  backend-private modules, tests, and compatibility normalizers.
- Torch parity routing tests: `backend=None` and `backend="torch"` produce the same trace summaries
  and validation results as the pre-registry path.
- MLX routing tests: MLX module detection uses `MLXSpec.can_handle`; save predicate,
  intervention, halt, output device, and backward requests raise canonical unsupported errors.
- Validation dispatch tests: `Trace.validate_forward_pass()` calls the trace backend spec;
  consolidated `validate(scope=...)` dispatches through spec; JAX fake spec returns real bools in
  harness tests.
- Backward capability tests: `log_backward`, `recording_backward`, backward validation, and
  backward drawing route through spec and never import torch backward code for non-torch traces.
- Serialization dispatch tests: schema v1 no-backend manifests infer torch; schema v2 non-torch
  manifests resolve backend policy before `Manifest.from_dict()` or torch version enforcement.
- Accessor tests: non-torch traces expose `ops`, `layers`, `modules`, `params`, saved accessors,
  and inert-empty backward accessors without backend-specific traps.
