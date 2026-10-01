# Memory debugging: which question, which tool

<!-- torchnative W0.10: the defer-by-name recipe page. The snapshot API and
     pickle format are underscore-private and allocator block reuse breaks
     per-op causality, so TorchLens NEVER parses snapshots -- it names the
     authority and hands you the tested recipe. Verified on torch 2.13.0. -->

| Question | Tool |
|---|---|
| "Why did I OOM?" (allocator autopsy) | CUDA memory snapshot + memory_viz (recipe below) |
| "Which tensors do I retain, per op?" | TorchLens's tensor-scope timeline (`tl.export.memory_timeline`) |
| "Am I leaking tensors through reference cycles?" | `torch.utils.viz._cycles.warn_tensor_cycles()` |
| "What was device memory doing over time, categorized?" | Torch deprecated its categorized view; TorchLens's rebuilt categorized timeline (in progress) emits torch's category words only where they map 1:1 to observed record facts |

## The snapshot recipe (OOM autopsy)

The snapshot is the allocator's own account — every block, every stream,
process-wide. Its API is private and its format undocumented, so treat it
as a torch artifact end to end:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)  # bounded!
try:
    pass  # ... run the workload that OOMs ...
finally:
    torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)  # always disarm
```

Open `oom_snapshot.pickle` at <https://pytorch.org/memory_viz> (drag and
drop; nothing uploads — it renders locally). The flamegraph view answers
"what was alive at peak"; the trace view answers "what allocated right
before the failure".

Notes from the panel's measurements:

- **Bound `max_entries`.** Unbounded history on a long run is its own
  memory problem.
- **Disarm in `finally`.** Recording left on degrades everything after it.
- **Do not expect per-op causality.** The allocator reuses blocks; a
  block's stack trace names its *allocation* site, not the op that holds
  it now. That join is exactly the false-join class TorchLens deletes from
  its profiler timing, which is why no `explain_snapshot` exists here.
- TorchLens reports can hold an opaque reference to a snapshot you dumped
  (kind + digest + byte size), but the payload is never parsed.

## Leak detection

```python
from torch.utils.viz._cycles import warn_tensor_cycles

warn_tensor_cycles()  # warns when a reference cycle keeps CUDA tensors alive
```

## Where the categorized timeline stands

Torch's `export_memory_timeline` is deprecated; its private categorizer
still runs and TorchLens pins its behavior on a frozen torch-2.13 fixture
(the parity oracle, built while the oracle can still be built). The
rebuilt TorchLens view emits `parameter / gradient / input / activation /
optimizer_state / unknown` — each an observed record fact — and never
emits `TEMPORARY` or `AUTOGRAD_DETAIL`, torch's heuristic-only inferences
(17.7% of its keyed tensors on a real training step). Totals are
correspondingly lower and the migration table says so; the allocator
snapshot above remains the process-level account.

Review trigger, recorded: if torch ever ships a stable allocator
annotation whose exact IDs survive into snapshot events across our torch
floor (the FX-only `augment_with_fx_traces` frames show upstream reaching
for this), the no-parse posture gets re-reviewed.
