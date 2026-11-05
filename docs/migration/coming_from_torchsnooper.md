# Coming from torchsnooper

This page is for people who used torchsnooper (or pysnooper) to watch a forward pass and see the
last lines before a crash. The examples were written against torchsnooper's `snoop` decorator and
the TorchLens version on this branch. Checked 2026-11-03.

torchsnooper printed one line per executed Python source line with tensors rendered as
`tensor<(6, 16), float32, cpu>`; its niche was watching a forward pass happen and seeing the last
lines before a crash. TorchLens covers that niche with `narrate=`: one line per captured
operation rather than per source line, plus crash workflows torchsnooper did not have. The
decorator idiom maps in five lines:

```text
# @torchsnooper.snoop()          ->  narrate= on the capture entry
# with torchsnooper.snoop():     ->  recording.Recorder(model, narrate=...)
import torchlens as tl

recording = tl.record(model, x, narrate=True)          # the one-liner replacement
```

## The same task in both

The task: print a one-line summary of every tensor as a small forward runs, into a buffer.

### In torchsnooper

Needs torchsnooper; the fence is skipped where it is not installed. torchsnooper traces with
`sys.settrace`, line by line.

```python
# migration-test: tool=torchsnooper expected=True
import io

import torch
import torchsnooper

sink = io.StringIO()
weight = torch.eye(2)


@torchsnooper.snoop(output=sink)
def forward(x):
    return torch.relu(x @ weight)


forward(torch.tensor([[2.0, 3.0]]))
# every traced line summarizes tensors as tensor<shape, dtype, device>
RESULT = "tensor<" in sink.getvalue()
```

### In TorchLens

One line per captured event (input source, linear, relu), through a sink:

```python
# migration-test: tool=torchlens expected=3
import io

import torch
from torch import nn

import torchlens as tl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(self, x):
        return torch.relu(self.proj(x))


sink = io.StringIO()
torch.manual_seed(0)
tl.record(
    Tiny(),
    torch.tensor([[2.0, 3.0]]),
    narrate=tl.options.NarrationOptions(sites=True, sink=sink),
)
# one line per captured event: input source, linear, relu
RESULT = sum(1 for line in sink.getvalue().splitlines() if line.lstrip().startswith("#"))
```

## Translation table

| torchsnooper construct | TorchLens equivalent |
| --- | --- |
| `@torchsnooper.snoop()` on a forward | `tl.record(model, x, narrate=True)` -- op-granular lines, module structure included |
| `with torchsnooper.snoop():` around a training step | `with tl.recording.Recorder(model, narrate=tl.options.NarrationOptions(...)) as r: r(batch)` |
| `snoop(output='file.log')` | `NarrationOptions(sink='file.log')` -- flushed per line, so bytes survive a hard death |
| `snoop(depth=N)` | module-depth indentation is built in; scope with `tl.in_module(...)` |
| `snoop(prefix=...)` per-line prefix | callable sink (`NarrationOptions(sink=my_logger)`) -- the seam to loggers/trackers |
| `normalize=True` diffable output | transcripts are byte-stable by default (no addresses/wall-clock on lines) |
| last lines before a crash | `exc.partial_trace.narrate(20)` -- works with echo OFF, plus the live tail when echo is armed |

## What TorchLens adds

- **Selector scoping** (torchsnooper structurally could not): narrate ONLY the
  region you care about -- `narrate=tl.in_module("transformer.h.3")` prints a
  transformer block's ops and nothing else; inside an HF forward there is no
  Python function that corresponds to a block, so a decorator could never do
  this.
- **Module structure lines**: `>`/`<` enter/exit lines indent by MODULE depth,
  not Python call depth (which in an HF model is a tour of
  `torch/nn/modules/module.py`).
- **Crash tails without foresight**: crashes are not scheduled. The partial
  TorchLens already attaches to a failed capture renders the last lines
  RETROACTIVELY -- `exc.partial_trace.narrate(last=20)` -- with graph coordinates
  and module context. Live narration remains for the deaths Python cannot
  survive (segfault, OOM kill): file sinks flush per line, so only-on-disk
  bytes survive.
- **The attempted-op marker**: the op that raised never produced an output to
  log, so the tail's last line names the in-flight call with its input shapes
  -- always qualified `(not proven culprit)`, never over-claimed (on CUDA an
  async kernel error can surface at a later sync point; the caveat prints in
  the output).
- **Evidence-qualified stats**: per-line descriptive statistics (the one open
  torchsnooper issue) ride a four-rung cost ladder.
- **Aligned diffs**: byte-stable transcripts diff cleanly by construction, and
  `tl.debug.compare(trace_a, trace_b)` is the aligned, value-level upgrade of
  text-diffing two logs.

### Cost ladder (`value_stats=`)

Echo rides the record substrate or the trace substrate; metadata echo adds
formatting and I/O only -- zero payload reads, zero device syncs by
construction (CI-enforced). Value stats are opt-in per rung:

| rung | prints | cost shape | soundness rule |
| --- | --- | --- | --- |
| `off` (default) | metadata only | zero value reads, zero syncs | -- |
| `reuse` | only facts another armed feature already paid for (`on_nonfinite='track'`, `on_nonfinite='raise'`) | zero additional reads | exact where present; ABSENT where not, never `0%` |
| `sampled` | `~`-marked moments from a bounded seeded budget with `sampled=k/N` evidence | roughly flat per line across tensor sizes | **no finiteness claim, ever** (a planted NaN in 12.6M elements is invisible to an 8k subsample) |
| `exact` | full exact stats + exact nonfinite census | scales with bytes | typed refusal above the documented numel budget; the remedy names `sampled` |

`on_nonfinite='track', narrate=..., value_stats="reuse"` is the recommended recipe for
a numerically suspicious run. At full volume the terminal itself can dominate:
prefer `sink="file.log"` and the opt-in bounds (`max_lines`) on big models.
Substrate multipliers and per-line costs are measured by
`benchmarks/snoop_echo_bench.py` on real resnet18/gpt2; ratios are
box-dependent -- run it on yours.

Rely on the partial when you expect an exception; turn narration on when you
expect the process to DIE.

## What torchsnooper does that TorchLens does not

The `sys.settrace` mechanism, `watch=` locals, `watch_explode`, thread_info,
and arbitrary Python-expression tracing are pysnooper's identity, not
TorchLens's: narration here is a display of CAPTURE events. No environment
variable ever turns narration on (`narrate=` is code-explicit).

- torchsnooper is a one-decorator, no-setup tool that prints Python source lines as they run.
  TorchLens narrates capture events, so it needs a model and an input to run (`tl.record` or
  `tl.trace`) and reports operations and modules, not arbitrary Python lines.
- Plain Python code with no tensors in a model is outside TorchLens; for line-level tracing of
  general Python, pysnooper remains the tool. For other destinations see
  [where to go instead](where_to_go_instead.md).

## Using both

### torchsnooper feeds TorchLens

There is no bridge. torchsnooper prints text; TorchLens consumes a model and an input. Keep the
decorator on helper functions outside the model and use `narrate=` on the model call.

### TorchLens feeds torchsnooper

There is no bridge. TorchLens narration goes to a sink (a file path, a stream, or a callable such
as a logger), so its lines can sit in the same log file as torchsnooper's output.

## Semantic differences

- Granularity. torchsnooper lines follow Python source lines; TorchLens lines follow captured
  operations, and `>`/`<` enter and exit lines indent by module depth, not Python call depth.
- Reading a crash. Rely on the partial trace when you expect an exception
  (`exc.partial_trace.narrate(20)`); turn narration on when you expect the process to die.
- No environment variable ever turns narration on: `narrate=` is code-explicit.
- Credit. **torchsnooper** (zasdfgbnm): the compact fixed-order per-line tensor summary and the crash-diagnosis workflow. **pysnooper** (cool-RR): the narration idiom, `normalize=` for diffable transcripts, and the depth/prefix/output vocabulary. **lovely-tensors** (xl0): the stats-line aesthetic (via the shared TorchLens stats grammar).
