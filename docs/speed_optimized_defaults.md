# Speed-Optimized Defaults

TorchLens defaults favor complete metadata and debuggability. For high-throughput capture, use the
smallest capture surface that answers your question.

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

Use `tl.trace(..., save=...)` when you need a full `Trace` with graph metadata, and use
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

`tl.fastlog.record(...)` remains available as a compatibility path. `keep_op=` and
`keep_module=` are deprecated aliases; use `record(save=...)` for new code. Keep predicate
functions small and deterministic because they run in the logging hot path.

## Final-Label Saves

`layers_to_save=[...]` is still supported when selection depends on final labels that are only known
after postprocessing. That path runs the legacy two-pass strategy. For recurrent layers,
`layers_to_save=["attn"]` saves all passes, while `layers_to_save=["attn:2"]` saves only pass 2
using TorchLens's 1-based pass-label syntax.

## What Not To Optimize Away

Do not wrap internal TorchLens tensor operations yourself. TorchLens already uses `pause_logging()`
where needed around internal tensor work. If a model is nondeterministic, fix the model/input RNG
context before comparing captures.
