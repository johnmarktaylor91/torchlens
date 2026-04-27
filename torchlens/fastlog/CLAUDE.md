# fastlog/ — Predicate-Based Activation Recording

## What This Does
`torchlens.fastlog` records a sparse, predicate-selected subset of forward-pass events.
It is a sibling path to `log_forward_pass()`: the decorated torch wrappers still fire,
but fastlog builds lightweight `RecordContext` values and stores only events selected by
`keep_op` or `keep_module` predicates.

## Architecture

Fastlog predicates fire inside the logging hot path after TorchLens has enough
operation or module-event metadata to synthesize a frozen `RecordContext`, but before
the result is appended to storage. Predicate calls are wrapped in `pause_logging()` so
internal torch operations performed by user predicates do not recursively log events.

Storage is resolved from `StreamingOptions`:

- `bundle_path=None` records to RAM only.
- `bundle_path` with `retain_in_memory=True` mirrors attached or detached RAM copies to disk.
- `bundle_path` with `retain_in_memory=False` records disk-only and rejects `keep_grad=True`.

`train_mode=True` is the public training knob. When the caller omits `default_op` or
`default_module`, fastlog promotes that omitted default to
`CaptureSpec(keep_grad=True, save_activation=True, save_metadata=True)`. Explicit
defaults still win: `default_op=False` disables that slot, while `default_op=True` or
`CaptureSpec(keep_grad=False)` conflicts with training mode and raises
`TrainingModeConfigError`. Per-predicate `CaptureSpec.keep_grad` remains the low-level
control for advanced selection, and `_storage_resolver.py` remains the only place that
decides whether a RAM payload is attached or detached. Disk mirrors are detached
inspection copies; the trainable payload is always the RAM copy.

`Recorder` is the session orchestrator. It prepares the model, owns the active
`RecordingState`, opens one logging scope per `log()` call, coordinates predicate
evaluation through the orchestrator, and finalizes RAM or disk indexes on context exit.
One-shot `record()` is a thin wrapper around the same `Recorder` path.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public `tl.fastlog` namespace exports. |
| `types.py` | Source of truth for `CaptureSpec`, `RecordContext`, `ActivationRecord`, `Recording`, and trace records. |
| `_state.py` | Mutable per-session recording state, storage backend wiring, predicate failure accumulation. |
| `_orchestrator.py` | Runs one predicate pass and appends retained operation/module records. |
| `storage_ram.py` | In-memory recording backend and indexes. |
| `storage_disk.py` | Synchronous directory-bundle writer, JSONL index, manifest integration. |
| `_record_one_shot.py` | Public `record()` implementation. |
| `_recorder.py` | Public many-rollout `Recorder` context manager and DDP/FSDP handling. |
| `dry_run.py` | Predicate trace without tensor payload retention. |
| `recover.py` | Finalized bundle load and partial-bundle recovery. |
| `_predicate.py` | Predicate normalization, error handling, and default decision resolution. |
| `_record_context.py` | Canonical `RecordContext` construction and recent-history views. |
| `_storage_resolver.py` | The only tensor copy/detach routing for fastlog payloads. |
| `options.py` | Fastlog option dataclasses and merge logic. |
| `exceptions.py` | Fastlog-specific public exception types. |
| `cleanup.py` | Partial bundle cleanup helper. |

## Conventions

- Do not use `torch.no_grad()` in the recording hot path; fastlog must remain
  training-compatible.
- Do not call `.detach()` directly. Route tensor payload copies through
  `_storage_resolver._resolve_storage()` and `safe_copy()`.
- Keep frozen record dataclasses on `slots=True`.
- Wrap predicate calls in `pause_logging()`.
- Keep `RecordContext` fields centralized in `types.py` and `_record_context.py`.
- Preserve rank-local DDP semantics: unwrap to `.module`, but do not claim true
  distributed forward semantics.

## Future Work

- Async disk drain after the storage dependency and benchmarks support it.
- True DDP forward semantics rather than rank-local unwrapped capture.
- FSDP support if an unsharded event view becomes practical.
- A narrow `to_model_log`-style conversion only if sparse captures can be represented
  without violating `ModelLog` invariants.
