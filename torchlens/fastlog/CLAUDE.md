# fastlog/ - Predicate-Based Sparse Recording

## What This Does
`torchlens.fastlog` records a predicate-selected subset of forward-pass events. It is a
sibling path to `trace()`: the same decorated wrappers fire, but fastlog builds
lightweight `RecordContext` values and stores only records selected by `keep_op` or
`keep_module` predicates.

## Public Surface
- `record()` - one-shot sparse recording.
- `Recorder` - context manager for repeated rollouts.
- `Recording` / `ActivationRecord` / `RecordingTrace` - returned data structures.
- `CaptureSpec` - per-predicate keep/save/metadata decision.
- `RecordingOptions` - grouped fastlog options.
- `dry_run()` - predicate trace without payload retention.
- `preview()` - visualization overlay from `visualization.fastlog_preview`.
- `load()`, `recover()`, `cleanup_partial()` - disk bundle management.

## Files

| File | Purpose |
|------|---------|
| `types.py` | `CaptureSpec`, `RecordContext`, `ActivationRecord`, `Recording`, trace records |
| `options.py` | Fastlog-specific recording options |
| `_record_one_shot.py` | Public `record()` implementation |
| `_recorder.py` | Public `Recorder` context manager and repeated rollout orchestration |
| `_state.py` | Mutable active recording state and predicate failure accumulation |
| `_orchestrator.py` | Predicate evaluation and retained-record append logic |
| `_predicate.py` | Predicate/default normalization and error handling |
| `_record_context.py` | Canonical `RecordContext` construction and recent-history views |
| `_storage_resolver.py` | Tensor copy/detach/RAM/disk routing |
| `_indexes.py` | Recording index helpers |
| `_validation.py` | Option validation |
| `storage_ram.py` | In-memory backend |
| `storage_disk.py` | Directory bundle writer and JSONL index |
| `dry_run.py` | Predicate dry-run support |
| `recover.py` | Finalized and partial bundle load/recovery |
| `cleanup.py` | Partial bundle cleanup |
| `exceptions.py` | Fastlog exception classes |

## Architecture
Predicates fire inside the logging hot path after TorchLens has enough metadata to build a
frozen `RecordContext`. Predicate calls are wrapped in `pause_logging()` so user predicate
torch operations do not recursively log events.

Storage is resolved from streaming/recording options:
- RAM-only when no bundle path is supplied.
- RAM plus disk mirror when `retain_in_memory=True`.
- Disk-only when `retain_in_memory=False`, rejecting `keep_grad=True`.

Forward exceptions are opt-in partials. The default is the historical
`on_forward_error="raise"` path. `on_forward_error="attach_partial"` attaches a failed
`Recording` to `exc.partial_recording` and re-raises the original exception.
`on_forward_error="return_partial"` swallows the forward exception and returns the failed
partial; with `return_output=True` the output slot is `None`, and manual `Recorder.log()`
returns `None` while `recorder.recording` holds the partial. Failed partials set
`status="partial_error"`, `failed=True`, `error_repr`, `error_traceback`,
`n_ops_completed`, `last_successful_op_label`, and best-effort `last_event_*` fields.
user-op failures exclude the failing call; TL-side capture failures may include a
skipped/partial current-call event. `Recording.to_trace()` and `Recording.log_backward()`
reject failed partials because the topology is incomplete.

Trace has a separate failed-capture diagnostic: failed `tl.trace(...)` exceptions carry
`exc.partial_log`, which can be retrieved with `tl.partial.from_failed_capture(exc)`.

## Training Semantics
`backward_ready=True` promotes omitted defaults to graph-connected out capture. Explicit
defaults still win, but incompatible `keep_grad=False` or disk-only settings raise
configuration errors. Disk mirrors are detached inspection copies; trainable payloads are RAM
copies.

## Conventions
- Do not use `torch.no_grad()` in recording hot paths.
- Do not call `.detach()` directly; route payload handling through `_storage_resolver.py`.
- Keep record dataclasses frozen/slots where possible.
- Wrap predicate calls in `pause_logging()`.
- Preserve rank-local DDP semantics; fastlog unwraps `.module` but does not claim true
  distributed forward semantics.

## Future Work
- Async disk drain once benchmarks support it.
- True distributed capture semantics.
- Narrow sparse-to-Trace conversion only if it can preserve `Trace` invariants.
