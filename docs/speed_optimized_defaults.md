# Speed-Optimized Defaults

TorchLens defaults favor complete metadata and debuggability. For high-throughput capture, use the
smallest capture surface that answers your question.

Backend note: these defaults describe torch capture. `tl.record()`/fastlog is torch-only in
backend v1. JAX, tinygrad, Paddle, and TensorFlow preview `.tlspec` payloads materialize
arrays, but loaded traces report replay validation as unavailable because runtime replay captures
are stripped.

## Recommended Capture Settings

```python
import torchlens as tl

log = tl.trace(
    model,
    x,
    save=tl.func("linear"),
    save_code_context=False,
)
```

Use these defaults when speed matters:

| Setting | Speed-oriented value | Why |
| --- | --- | --- |
| Visualization | Omit `Trace.draw()` | Avoid graph rendering while collecting data. |
| Saved layers | `save=tl.func(...)` / `save=tl.in_module(...)` | Saves activation payloads only where needed in the primary single-pass path. |
| Windowed selection | `lookback=K` with `tl.followed_by(...)` / `tl.preceded_by(...)` | Enables local graph-context predicates without post-hoc filtering. |
| Lookback payloads | Default `lookback_payload_policy="metadata_only"` | Avoids pinning recent tensors unless retroactive payload saving is required. |
| Source context | `save_code_context=False` | Keeps file/line identity but avoids source-text loading. |
| Validation | Run separately | Validation replays the model and should be a gate, not the hot path. |
| Streaming | `storage=tl.to_disk(path)` | Moves predicate-selected payload storage to disk while preserving the manifest. |

## Predicate Capture

Use `tl.trace(..., save=...)` when you need a full `Trace` with graph metadata, and use torch
`tl.record(..., save=...)` when you only need selected records during tight loops.
`Recording.to_trace()` materializes the full graph structure later, but reading an unsaved payload
raises a clear error.

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()

conv_before_relu = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
windowed = tl.trace(
    model,
    x,
    save=conv_before_relu,
    lookback=4,
    lookback_payload_policy="detached_raw",
)

streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
```

`tl.fastlog.record(...)` remains available as a torch compatibility path. `keep_op=` and
`keep_module=` are deprecated aliases; use `record(save=...)` for new code. Keep predicate
functions small and deterministic because they run in the logging hot path.

Fastlog forward exceptions default to the historical `on_forward_error="raise"` behavior.
Use `on_forward_error="attach_partial"` to attach `exc.partial_recording` and re-raise the
original exception, or `on_forward_error="return_partial"` to return a failed partial
`Recording`. With `return_output=True`, return-partial mode returns `(None, partial)` because
there is no valid model output; manual `Recorder.log()` returns `None` and exposes the partial
on `recorder.recording`. Failed partials set `status="partial_error"`, `failed=True`,
`error_repr`, `error_traceback`, `n_ops_completed`, `last_successful_op_label`, and best-effort
`last_event_*` fields. user-op failures exclude the failing call; TL-side capture failures may
include a skipped/partial current-call event. Failed partials reject `Recording.to_trace()` and
`Recording.log_backward()`.

Full `tl.trace(...)` failures expose a separate honest partial as `exc.partial_log`; retrieve it
with `tl.partial.from_failed_capture(exc)`.

## Final-Label Saves

`layers_to_save=[...]` is still supported when selection depends on final labels that are only known
after postprocessing. That path runs the legacy two-pass strategy. For recurrent layers,
`layers_to_save=["attn"]` saves all passes, while `layers_to_save=["attn:2"]` saves only pass 2
using TorchLens's 1-based pass-label syntax.

## What Not To Optimize Away

Do not wrap internal TorchLens tensor operations yourself. TorchLens already uses `pause_logging()`
where needed around internal tensor work. If a model is nondeterministic, fix the model/input RNG
context before comparing captures.
