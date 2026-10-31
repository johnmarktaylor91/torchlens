# Coming from torchviz or hiddenlayer

This page maps torchviz's `make_dot` and hiddenlayer's graph folding to TorchLens: their
spelling on the left, the TorchLens spelling on the right.

## torchviz

| torchviz | TorchLens | Notes |
| --- | --- | --- |
| `make_dot(y)` on a tensor you already have | `tl.report.sketch_grad_fn(y)` | Walks `y.grad_fn` after the fact, with no re-run and no capture. Returns the DOT source. |
| `make_dot(y, params=dict(model.named_parameters()))` | `tl.report.sketch_grad_fn(y, params=dict(model.named_parameters()))`, or `model=model` | Same leaf-naming convention. |
| `dot.render("graph", format="png")` | `tl.report.sketch_grad_fn(y, "graph", file_format="png")` | Writes `graph.png`; `view=True` opens it. |
| The walk as data | `tl.report.walk_grad_fn(y)` | A `GradFnSketch` with nodes, edges and a truncation flag. |
| `make_dot(..., show_attrs=True, show_saved=True)` | Capture a trace instead (see below) | The sketch stays structure-only by design: it never reads values. |
| `make_dot_from_trace` (JIT trace) | `tl.trace(model, x).draw()` | The forward graph of every executed op. |

`sketch_grad_fn` and `walk_grad_fn` read only `grad_fn` and `next_functions`: no tensor
values, no timing, no verification, and every render says so in its legend. When you
want the values torchviz's `show_saved` prints, capture the forward with `tl.trace`:
every op keeps its saved output (`trace[label].out`) and its size in bytes
(`trace[label].activation_memory`). To draw the backward graph of a captured run, with
the module that owns each node and a gradient summary on each node, use
`trace.draw_backward(show_module_paths=True, show_grad_values=True)` after a backward
pass the trace recorded (see `docs/backward.md`).

Their construct:

```python
# migration-test: tool=torchviz expected=True
import torch
from torch import nn
from torchviz import make_dot

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
y = model(torch.randn(2, 4)).sum()
dot = make_dot(y, params=dict(model.named_parameters()))
RESULT = "AddmmBackward0" in dot.source
```

TorchLens equivalent:

```python
# migration-test: tool=torchlens expected=(True, True)
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
y = model(torch.randn(2, 4)).sum()
dot_source = tl.report.sketch_grad_fn(y, params=dict(model.named_parameters()))
RESULT = ("AddmmBackward0" in dot_source, "0.weight" in dot_source)
```

## hiddenlayer

hiddenlayer draws a simplified graph and folds repeated op chains into one node with
`hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu")`. TorchLens folds named op
chains at draw time with `fold_patterns=`.

| hiddenlayer | TorchLens | Notes |
| --- | --- | --- |
| `hl.build_graph(model, x)` | `tl.trace(model, x).draw()` | |
| `transforms=[hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu")]` | `trace.draw(fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"})` | Atoms name op functions (`conv2d`, `batch_norm`, `relu`) or module classes (`Conv2d`). |
| hiddenlayer's built-in fold set | `trace.draw(fold_patterns="idiomatic")` | Folds `ConvBnRelu`, `ConvBn`, `ConvRelu` and `LinearRelu`. |
| A fold built from an earlier fold | `{"LinRelu": "linear > relu", "LinRelu2": "LinRelu{2}"}` | An atom naming an earlier pattern inlines it; `{k}` repeats an atom exactly `k` times. |
| `hl.transforms.Prune(...)` | `trace.draw(skip_fn=...)` | `skip_fn` takes a layer and returns `True` to hide it. |
| `graph.save(path, format="pdf")` | `trace.draw(vis_outpath=path, vis_fileformat="pdf", vis_save_only=True)` | |

`fold_patterns=` cannot be combined with automatic collapse (`collapse=`) and refuses
with `pattern_collapse_combination_unsupported`. hiddenlayer has had no release since
2020 and is not executed here; the TorchLens side is:

```python
# migration-test: tool=torchlens expected=(True, True)
import os
import tempfile

import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU()).eval()
trace = tl.trace(model, torch.randn(1, 3, 8, 8))
outpath = os.path.join(tempfile.mkdtemp(), "folded")
dot_source = trace.draw(
    fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"},
    vis_outpath=outpath,
    vis_fileformat="svg",
    vis_save_only=True,
)
RESULT = ("ConvBnRelu" in dot_source, os.path.exists(outpath + ".svg"))
```
