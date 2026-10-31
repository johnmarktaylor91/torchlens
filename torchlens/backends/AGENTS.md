# Backend Author Contract

TorchLens has two backend integration tiers.

## Tier 1: Registered Backend Spec

Every public backend must register a `BackendSpec` in `registry.py`/`default_specs.py`.
This is the contract for `tl.trace(..., backend=...)`, validation dispatch,
serialization policy, and public capability reporting.

A standalone backend must provide:

- `can_handle(model, input_args, input_kwargs)` for auto-resolution.
  It must reject inputs containing native tensor leaves from any other backend
  family, even when the model or another input leaf matches this backend.
- `capture_trace(*args, **kwargs)` for its public trace entry.
- `validate_entry(...)` and `validate_trace(...)`, or typed unsupported dispatchers.
- `BackendCapabilities` in the registry. This is the only source of capability truth.
- `SerializationPolicy` when payloads or manifests differ from torch defaults.

Use `BackendUnsupportedError` for unsupported public surfaces. Do not silently accept
options that the registry does not declare.

## Tier 2: Shared CaptureBackend Protocol

`CaptureBackend` (defined in `backends/_protocol.py`) is the lower-level adapter used by
the shared eager orchestrator in `_capture/trace.py`. A backend should expose `BackendSpec.capture_backend` only when it
implements the full protocol: input normalization, source logging, RNG hooks, inference
context, output extraction, intervention hooks, cleanup hooks, and producer-policy hooks.

Registration validates the protocol attributes when `capture_backend` is supplied. If a
backend is static, callback-based, or otherwise owns its full capture loop, leave
`capture_backend=None` and keep the standalone `capture_trace` entry honest.

Torch currently implements both tiers. Preview engines may implement only Tier 1 until
they can satisfy the shared orchestration protocol without torch assumptions.

## Preview backends

Five technical-preview backends live beside `torch/`, each Tier-1 standalone
(`capture_backend=None`) with per-directory implementation guides:

- `tf/` - eager op-callback live capture plus a static FuncGraph path
  (`tf/AGENTS.md`).
- `jax/` - jaxpr-first static capture with Equinox/Flax-NNX module helpers
  (`jax/AGENTS.md`).
- `mlx/` - eager wrapper capture with live per-op replay validation
  (`mlx/AGENTS.md`).
- `paddle/` - eager dygraph wrapper capture with live intervene=/halt=
  (`paddle/AGENTS.md`).
- `tinygrad/` - UOp-snapshot capture of an eager forward (`tinygrad/AGENTS.md`).

Shared preview plumbing: `_finalize.py` / `_recurrence.py` (single-pass trace
finalization and the neutral recurrence grouper), `_options.py` (per-backend
kwarg and trace-option refusal policies), and `_selective_save.py` /
`_validation_shared.py`. `_intervention_evidence.py` names the backends whose
interventions write the trace evidence `Trace.is_pristine` is derived from (torch
only; the others read `None`, not recorded).
