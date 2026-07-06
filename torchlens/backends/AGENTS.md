# Backend Author Contract

TorchLens has two backend integration tiers.

## Tier 1: Registered Backend Spec

Every public backend must register a `BackendSpec` in `registry.py`/`default_specs.py`.
This is the contract for `tl.trace(..., backend=...)`, validation dispatch,
serialization policy, and public capability reporting.

A standalone backend must provide:

- `can_handle(model, input_args, input_kwargs)` for auto-resolution.
- `capture_trace(*args, **kwargs)` for its public trace entry.
- `validate_entry(...)` and `validate_trace(...)`, or typed unsupported dispatchers.
- `BackendCapabilities` in the registry. This is the only source of capability truth.
- `SerializationPolicy` when payloads or manifests differ from torch defaults.

Use `BackendUnsupportedError` for unsupported public surfaces. Do not silently accept
options that the registry does not declare.

## Tier 2: Shared CaptureBackend Protocol

`CaptureBackend` is the lower-level adapter used by the shared eager orchestrator in
`capture/trace.py`. A backend should expose `BackendSpec.capture_backend` only when it
implements the full protocol: input normalization, source logging, RNG hooks, inference
context, output extraction, intervention hooks, cleanup hooks, and producer-policy hooks.

Registration validates the protocol attributes when `capture_backend` is supplied. If a
backend is static, callback-based, or otherwise owns its full capture loop, leave
`capture_backend=None` and keep the standalone `capture_trace` entry honest.

Torch currently implements both tiers. Preview engines may implement only Tier 1 until
they can satisfy the shared orchestration protocol without torch assumptions.
