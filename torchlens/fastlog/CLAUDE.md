# fastlog/ - Predicate-Based Sparse Recording

## What This Does
`torchlens.fastlog` records a predicate-selected subset of forward-pass events. It is a
sibling path to `log_forward_pass()`: the same decorated wrappers fire, but fastlog builds
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

## Training Semantics
`train_mode=True` promotes omitted defaults to graph-connected activation capture. Explicit
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
- Narrow sparse-to-ModelLog conversion only if it can preserve `ModelLog` invariants.
