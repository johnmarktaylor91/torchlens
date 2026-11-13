# Speed-Optimized Defaults

TorchLens defaults favor complete metadata and debuggability. For high-throughput capture, use the
smallest capture surface that answers your question.

Backend note: these defaults describe torch capture. `tl.record()` (sparse recording) is torch-only. JAX, tinygrad, Paddle, and TensorFlow `.tlspec` payloads materialize
arrays, but loaded traces report replay validation as unavailable because runtime replay captures
are stripped.

## Recommended Capture Settings

```python
import torchlens as tl

log = tl.trace(
    model,
    x,
    save=tl.func("linear"),
    capture=tl.options.CaptureOptions(save_source_context=False),
)
```

Use these defaults when speed matters:

| Setting | Speed-oriented value | Why |
| --- | --- | --- |
| Visualization | Omit `Trace.draw()` | Avoid graph rendering while collecting data. |
| Saved layers | `save=tl.func(...)` / `save=tl.in_module(...)` | Saves activation payloads only where needed in the primary single-pass path. |
| Windowed selection | `lookback=K` with `tl.followed_by(...)` / `tl.preceded_by(...)` | Enables local graph-context predicates without post-hoc filtering. |
| Lookback payloads | Default `lookback_payload_policy="metadata_only"` | Avoids pinning recent tensors unless retroactive payload saving is required. |
| Source context | `CaptureOptions(save_source_context=False)` (the default) | Keeps file/line identity and defers source text to read time (from the file on disk, only while it is unchanged since capture). |
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
capture=CaptureOptions(lookback=4, lookback_payload_policy="detached_raw"),
)

streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
```

`tl.record(...)` remains available as a torch compatibility path; `record(save=...)`
is the only predicate spelling (the old alias kwargs are removed). Keep predicate
functions small and deterministic because they run in the logging hot path.

`tl.record` forward exceptions default to the historical `on_forward_error="raise"` behavior.
Use `on_forward_error="attach_partial"` to attach `exc.partial_recording` and re-raise the
original exception, or `on_forward_error="return_partial"` to return a failed partial
`Recording`, whose `output` refuses with `output_forward_failed` because there is no valid model
output; manual `Recorder(...)` returns `None` and exposes the partial
on `recorder.recording`. Failed partials set `status="partial_error"`, `failed=True`,
`error_repr`, `error_traceback`, `num_committed_ops`, `last_successful_op_label`, and best-effort
`last_event_*` fields. User-op failures exclude the failing call; TL-side capture failures may
include a skipped/partial current-call event. Failed partials reject `Recording.to_trace()` and
`Recording.backward()`.

Full `tl.trace(...)` failures raise `tl.errors.TraceCaptureError` (chained `from` the forward's
exception) with a separate honest partial on `exc.partial_trace`.

## Final-Label Saves

`save=[...]` is still supported when selection depends on final labels that are only known
after postprocessing. That path runs the legacy two-pass strategy. For recurrent layers,
`save=["attn"]` saves all passes, while `save=["attn:2"]` saves only pass 2
using TorchLens's 1-based pass-label syntax.

## What Not To Optimize Away

Do not wrap internal TorchLens tensor operations yourself. TorchLens already uses `pause_logging()`
where needed around internal tensor work. If a model is nondeterministic, fix the model/input RNG
context before comparing captures.
