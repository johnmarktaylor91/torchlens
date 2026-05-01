# Speed-Optimized Defaults

TorchLens defaults favor complete metadata and debuggability. For high-throughput capture, use the
smallest capture surface that answers your question.

## Recommended Capture Settings

```python
import torchlens as tl
from torchlens.options import CaptureOptions

log = tl.log_forward_pass(
    model,
    x,
    vis_opt="none",
    capture=CaptureOptions(
        layers_to_save=["linear_1_1"],
        keep_unsaved_layers=True,
        save_source_context=False,
    ),
)
```

Use these defaults when speed matters:

| Setting | Speed-oriented value | Why |
| --- | --- | --- |
| Visualization | `vis_opt="none"` | Avoid Graphviz/ELK rendering while collecting data. |
| Saved layers | Specific labels/selectors | Saves activation payloads only where needed. |
| Unsaved metadata | `keep_unsaved_layers=True` | Keeps graph context while avoiding unnecessary tensor copies. |
| Source context | `save_source_context=False` | Keeps file/line identity but avoids source-text loading. |
| Validation | Run separately | Validation replays the model and should be a gate, not the hot path. |
| Streaming | Use when activations are large | Moves payload storage to disk while preserving the manifest. |

## Fastlog

For predicate-based capture, use `tl.fastlog.record(...)` when you only need selected events and
can express the selection with `keep_op` / `keep_module` predicates. Keep predicate functions
small and deterministic; they run in the logging hot path.

## What Not To Optimize Away

Do not wrap internal TorchLens tensor operations yourself. TorchLens already uses `pause_logging()`
where needed around internal tensor work. If a model is nondeterministic, fix the model/input RNG
context before comparing captures.
