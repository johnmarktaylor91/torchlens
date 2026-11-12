# Backend Author Contract

Every backend, torch included, fills one contract: `BackendProtocol`
(`_backend_protocol.py`, exported as `tl.backends.BackendProtocol`). It is the
contract for all six backends (torch, tf, jax, mlx, paddle, tinygrad); the
conformance suite and the shared rules talk only to it.

## The backend protocol (the contract for all six)

Register a `BackendSpec` in `registry.py`/`default_specs.py` with:

- `protocol=`: a lazy factory returning the backend's protocol object. The
  objects live in the import-light package `_protocols/` (one module per
  backend, importing no backend package, so registration can validate them at
  import time). Registration checks the object against
  `BACKEND_PROTOCOL_MEMBERS`.
- No maturity label: a spec states what the backend supports through its
  capabilities and the feature table in `docs/backends.md`. The shipped
  `scorecard.json` embeds the six graduation items as evidence only;
  `tests/test_backend_maturity.py` pins that no spec, protocol or record
  carries a label.
- `capabilities=`: `BackendCapabilities`, the only source of capability truth.
  A `False` flag owes a typed refusal with the code from
  `refusal_code_for_capability(flag)`; the suite's `c1_refusal_<flag>` cells
  prove it.
- `can_handle(model, input_args, input_kwargs)`: it must reject inputs holding
  native tensor leaves from any other backend family.
- `capture_trace`, `validate_entry` and `validate_trace` (the public dispatch
  surface, kept beside the protocol until the capture-pipeline rebuild folds
  them into it), and `SerializationPolicy` when payloads differ from torch.

Protocol obligations that every backend shares:

- Refuse with `BackendUnsupportedError` and a `code=` from `REFUSAL_CODES`
  before any framework call with side effects. Never accept an option the
  registry does not declare. A new code goes in through FORKS or a Lead ruling,
  never ad hoc.
- `validate` returns a `ValidationStatus` whose decisions name a reason from
  `VALIDATION_REASONS` and whose failure names the first offending op.
- Constants follow `_constants.py` (called from `_finalize.py`): closure
  constants become parentless `constant` ops with payloads, literal producers
  keep zero parents and a payload, an unreadable constant refuses
  `constant_unrealizable`.
- `materialization="lazy"` backends realize once: one framework evaluation
  over every recorded value after the forward, payloads read only after it.
- `framework` is the backend's `FrameworkRequirement`; its range must equal the
  pyproject extra and its `pinned` the per-change CI install
  (`tests/test_backend_framework_pins.py`).
- The backend's shipped `scorecard.json` is regenerated with
  `python -m tests.conformance.scorecard --backend <name> --write` on the
  pinned version, never hand-edited; its docs page `docs/backends/<name>.md`
  keeps a tutorial and a can and cannot table equal to the record's capability
  summary (`scripts/checks/backend_docs_lockstep.py`).

## The orchestration adapter (torch-internal)

`CaptureBackend` (`_protocol.py`) is the lower-level adapter the shared eager
orchestrator in `_capture/trace.py` drives: input normalization, source logging,
RNG hooks, inference context, output extraction, intervention hooks, cleanup
hooks and producer-policy hooks. It is torch's internal capture seam, not the
backend contract. Torch fills the backend protocol through an adapter over its
public entries (`_protocols/torch.py`) and also supplies `capture_backend`.
Registration validates the adapter's attributes when `capture_backend` is
supplied. A static, callback-based or self-looping backend leaves
`capture_backend=None` and keeps its standalone `capture_trace` honest.

## Non-PyTorch backends

Five non-PyTorch backends live beside `torch/`, each filling
the backend protocol with `capture_backend=None`, with per-directory
implementation guides:

- `tf/` - eager op-callback live capture plus a static FuncGraph path
  (`tf/AGENTS.md`).
- `jax/` - jaxpr-first static capture with Equinox/Flax-NNX module helpers
  (`jax/AGENTS.md`).
- `mlx/` - eager wrapper capture with live per-op replay validation
  (`mlx/AGENTS.md`).
- `paddle/` - eager dygraph wrapper capture with live intervene=/stop_after=
  (`paddle/AGENTS.md`).
- `tinygrad/` - UOp-snapshot capture of an eager forward (`tinygrad/AGENTS.md`).

Shared plumbing for the non-PyTorch backends: `_finalize.py` / `_recurrence.py` (single-pass trace
finalization and the neutral recurrence grouper), `_options.py` (per-backend
kwarg and trace-option refusal policies), and `_selective_save.py` /
`_validation_shared.py`. `_intervention_evidence.py` names the backends whose
interventions write the trace evidence `Trace.is_pristine` is derived from (torch
only; the others read `None`, not recorded).
