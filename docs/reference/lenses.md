# Premade views

`Trace.draw(view=...)` draws one of nine premade views. A view is a bundle of draw
settings plus evidence checks and a caption, and each view answers one question about the
model. Leaving `view=` unset draws the plain graph, which already shows every op.

**The view names and their captions are DOCUMENTED-UNSTABLE**: they may still change before
they are frozen.

```python
from pathlib import Path

import torch
import torch.nn as nn

import torchlens as tl

trace = tl.trace(nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)), torch.randn(2, 8))
out = Path(globals().get("DOCS_TMPDIR", "/tmp")) / "speed"
trace.draw(view="speed", theme="paper", file_format="svg", path=str(out))
```

## Views and themes

A **view** chooses what to show (which draw settings, colour and size channels, and label
rows). A **theme** (`theme=`: `"torchlens"`, `"paper"`, `"dark"`, `"colorblind"` or
`"high_contrast"`) chooses how it looks (palette, ramps, typography). The two compose
freely: any theme works with any view.

A view's settings are defaults, never overrides. Every keyword you pass to `trace.draw`
wins over the view's own setting, and your `node_spec_fn` has the last word on every node.
Combinations are therefore spelled with keywords, not with extra view names:

```text
trace.draw(view="debug", show_edge_shapes=True)
trace.draw(view="speed", size_by="bytes")
trace.draw(view="dims", direction="leftright", color_by="flops")
```

The full order, lowest to highest:

```text
theme defaults < view settings < explicit keywords < intervention and status marks
< active channel specs < view node styling < your node_spec_fn
```

`node_style="profiling"` stays a separate setting: `"speed"`, `"memory"` and `"compute"`
turn its rows on themselves, and you can pass it with any other view or none.

## The nine views

| View | Question | When it refuses |
|---|---|---|
| `overview` | what is this model? | never refuses |
| `debug` | what is broken? | above 2000 ops (`view_above_detail_budget`) |
| `speed` | where does time go? | no timing evidence (`view_headline_evidence_missing`, or `view_source_zero_coverage`) |
| `dims` | what shape is the data? | above 2000 ops (`view_above_detail_budget`) |
| `transformer` | how is attention wired? | no attention structure (`view_attention_subject_absent`) |
| `memory` | what is big in bytes? | no activation-memory evidence |
| `sequence` | how do the passes unfold? | a single-pass trace (`view_single_pass_no_sequence`) |
| `compute` | where are the FLOPs? | no FLOP evidence |
| `interventions` | what did the edits change? | no recorded edit (`surgery_evidence_missing`) |

Every refusal is a typed `InvalidArgumentError` with a remedy. The subject refusals of
`transformer` and `sequence` point to the domain-neutral picture: leave `view=` unset, or
pass `view="overview"`. Any value outside the nine refuses `draw_view_invalid`, naming the
allowed values.

What each view sets:

- **overview**: draws rolled (a model with only one pass draws unrolled), folds repeated
  blocks, and picks a collapse level that keeps the drawing legible.
- **debug**: draws every op uncollapsed, with buffers, redundant arguments, orphans and the
  legend shown; marks ops whose output holds NaN or Inf, and names the usual dtype and
  device so only the exceptions need a label.
- **speed**, **memory**, **compute**: turn on the profiling rows and colour every node by
  time, bytes or FLOPs; `memory` also shows what autograd saved for backward.
- **dims**: draws every op uncollapsed and sizes each box by its element count per batch
  item (square-root scale, capped at four times a plain box).
- **transformer**: focuses the drawing on the first repeated block that holds attention
  (for GPT-2, `transformer.h.0`), draws it open, and names it in the caption. An explicit
  `module=` wins.
- **sequence**: draws unrolled, colours each node by its pass number, and stacks
  repeated structure into columns where the trace allows it (boxes side by side ran in
  the same pass, not in parallel).
- **interventions**: outlines every edited op in purple (dashed when the site is inferred,
  not recorded) and every other op whose saved value differs from the clean run in light
  purple, with its largest absolute change. The clean values come from the unedited trace
  the fork came from; when it is gone the caption says so and only the edits are marked.
  The full record, including the capture-status lines the header leaves out, is written
  beside a written figure as `<path>.census.txt`.

## The caption

Each view prints its caption lines under the header block. The first line is
`view: <name> -- <question>`; the lines after it say what the picture does and does not
show, for example:

- `speed`, `memory` and `compute` report coverage ("encoded N of M ops") and the colour
  unit; on a rolled drawing the caption says the value is summed over passes. `speed` adds
  that colour shows rank order, not proportion, and that times come from an instrumented
  run: compare ops with each other, not with production latency.
- When the drawing is collapsed, the caption names the collapse level used, and collapsed
  boxes under a colour channel get a neutral fill that means "aggregate, not encoded".
- `debug` says which nonfinite states it checked. With no saved activations there is
  nothing to check: the caption says so and TorchLens warns `view_nonfinite_not_checked`
  rather than drawing every op as finite.
- `dims` says that box area shows element count, not shape (512x7x7 and 64x28x28 draw the
  same size).

A setting a view wants but cannot honour on this trace (for example stacking passes when
the trace does not allow it) is dropped with a warning (`view_secondary_degraded`) and a
note in the caption, never silently.

## How the honesty checks work

- **Rank colours.** `speed`, `memory` and `compute` colour by rank, not by linear value: a
  linear ramp put most nodes of real models into the bottom tenth of the scale. When every
  op has the same value the nodes are left uncoloured with a note.
- **Rolled and unrolled sources.** An unrolled drawing colours by each op's own value
  (for example `func_duration`); a rolled drawing colours by the total over passes
  (`total_func_duration`) and says so in the caption.
- **Collapse level.** `overview` searches the collapse settings for the coarsest one that
  keeps the visible node count inside a fixed band (100 to 220, target 160), and keeps at
  least 80 percent of the values visible when a colour channel is on. If the full drawing
  already fits, nothing is collapsed.
- **Nonfinite status.** `debug` derives a status for every op (finite, NaN, +Inf, -Inf,
  mixed, or not checked) from the saved activations and draws a motif per status. An op
  that was not checked is never drawn as finite.
- **Theme colours.** Role colours come from the active theme. The default `torchlens`
  theme keeps the released look: light green input, salmon output, light grey trainable
  and darker grey frozen weights, yellow branch tests (the label reads `IF: TRUE` or
  `IF: FALSE`) and every other node white. `colorblind` uses the Okabe-Ito set with grey
  weights (green input, vermillion output); `paper` and `high_contrast` keep the default
  role colours and `dark` remaps them for a dark background. No theme draws blue.
