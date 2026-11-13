# Coming from torchviz or hiddenlayer

This page is for people who draw autograd graphs with torchviz's `make_dot`, or simplified
forward graphs with hiddenlayer's folding transforms, and want the TorchLens equivalent.
TorchLens's test environment does not pin torchviz; its example skips where torchviz is not
installed. hiddenlayer has had no release since 2020 and is not executed here. Checked
2026-11-03.

## The same task in both

Draw the autograd graph of a scalar loss, with parameter names on the leaves, and check it
names the backward nodes.

### In torchviz

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

### In TorchLens

```python
# migration-test: tool=torchlens expected=(True, True)
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
y = model(torch.randn(2, 4)).sum()
dot_source = tl.debug.sketch_grad_fn(y, params=dict(model.named_parameters()))
RESULT = ("AddmmBackward0" in dot_source, "0.weight" in dot_source)
```

## Translation table

### torchviz

| torchviz | TorchLens | Notes |
| --- | --- | --- |
| `make_dot(y)` on a tensor you already have | `tl.debug.sketch_grad_fn(y)` | Walks `y.grad_fn` after the fact, with no re-run and no capture. Returns the DOT source. |
| `make_dot(y, params=dict(model.named_parameters()))` | `tl.debug.sketch_grad_fn(y, params=dict(model.named_parameters()))`, or `model=model` | Same leaf-naming convention. |
| `dot.render("graph", format="png")` | `tl.debug.sketch_grad_fn(y, "graph", file_format="png")` | Writes `graph.png`; `view=True` opens it. |
| The walk as data | `tl.debug.walk_grad_fn(y)` | A `GradFnSketch` with nodes, edges and a truncation flag. |
| `make_dot(..., show_attrs=True, show_saved=True)` | Capture a trace instead (see below) | The sketch stays structure-only by design: it never reads values. |
| `make_dot_from_trace` (JIT trace) | `tl.trace(model, x).draw()` | The forward graph of every executed op. |

`sketch_grad_fn` and `walk_grad_fn` read only `grad_fn` and `next_functions`: no tensor
values, no timing, no verification, and every render says so in its legend. When you want the
values torchviz's `show_saved` prints, capture the forward with `tl.trace`: every op keeps its
saved output (`trace[label].out`) and its size in bytes (`trace[label].cost.activation_memory`). To
draw the backward graph of a captured run, with the module that owns each node and a gradient
summary on each node, use `trace.draw(graph="backward", show_module_paths=True,
show_grad_values=True)` after a backward pass the trace recorded (see `docs/backward.md`).

### hiddenlayer

hiddenlayer draws a simplified graph and folds repeated op chains into one node with
`hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu")`. TorchLens folds named op chains
at draw time with `fold_patterns=`.

| hiddenlayer | TorchLens | Notes |
| --- | --- | --- |
| `hl.build_graph(model, x)` | `tl.trace(model, x).draw()` | |
| `transforms=[hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu")]` | `trace.draw(fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"})` | Atoms name op functions (`conv2d`, `batch_norm`, `relu`) or module classes (`Conv2d`). |
| hiddenlayer's built-in fold set | `trace.draw(fold_patterns="idiomatic")` | Folds `ConvBnRelu`, `ConvBn`, `ConvRelu` and `LinearRelu`. |
| A fold built from an earlier fold | `{"LinRelu": "linear > relu", "LinRelu2": "LinRelu{2}"}` | An atom naming an earlier pattern inlines it; `{k}` repeats an atom exactly `k` times. |
| `hl.transforms.Prune(...)` | `trace.draw(skip_fn=...)` | `skip_fn` takes a layer and returns `True` to hide it. |
| `graph.save(path, format="pdf")` | `trace.draw(path=path, file_format="pdf")` | |

`fold_patterns=` cannot be combined with automatic collapse (`collapse=`) and refuses with
`pattern_collapse_combination_unsupported`. The TorchLens side of the hiddenlayer fold:

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
    path=outpath,
    file_format="svg",
).dot
RESULT = ("ConvBnRelu" in dot_source, os.path.exists(outpath + ".svg"))
```

## What TorchLens adds

- A forward graph of every executed op with values: `tl.trace(model, x).draw()` and
  `trace[label].out`, where `make_dot` draws only the autograd structure.
- The backward graph of a captured run, with the owning module on each node and a gradient
  summary: `trace.draw(graph="backward", show_module_paths=True, show_grad_values=True)`.
- The same structure-only walk as torchviz, honestly labelled: `tl.debug.sketch_grad_fn(y)`
  states in its legend that it read no values.
- Named, composable folds at draw time (`fold_patterns=`, including `"idiomatic"`), and
  `skip_fn=` to hide layers, without changing the capture.

## What torchviz and hiddenlayer do that TorchLens does not

- torchviz draws from a tensor you already have, with no model and no input. TorchLens's
  `sketch_grad_fn` does the same walk, but the richer picture needs a `tl.trace` of a run.
- torchviz's `show_attrs=True` prints each backward node's attributes. The TorchLens sketch
  does not read them.
- hiddenlayer's graph building from a PyTorch or TensorFlow model and its framework-neutral
  transform set are not mirrored; TorchLens traces PyTorch eager models (a JAX backend
  exists, see `docs/backends.md`).

See [where to go instead](where_to_go_instead.md) for tools outside TorchLens's scope.

## Using both

### torchviz or hiddenlayer feeds TorchLens

There is no bridge. Both draw a model or a tensor you already have; pass the same `nn.Module`
to `tl.trace`, or the loss tensor to `tl.debug.sketch_grad_fn`.

### TorchLens feeds torchviz or hiddenlayer

There is no bridge. `trace.draw` returns a `tl.viz.Drawing` whose `.dot` is DOT source, which
Graphviz tools can read, but it is not a `graphviz.Digraph` made by torchviz.

## Semantic differences

- **Structure only.** `sketch_grad_fn` and `walk_grad_fn` never read values, so a missing
  value in the picture does not mean zero. A trace carries the values.
- **Forward and backward are separate drawings.** `trace.draw()` is the forward graph; the
  backward graph is `graph="backward"` after a recorded backward pass.
- **Folds name ops, not graph nodes.** `fold_patterns=` atoms are op function names
  (`conv2d`) or module classes (`Conv2d`), where hiddenlayer matches its own op names.
- **No fold with collapse.** `fold_patterns=` and `collapse=` together refuse with
  `pattern_collapse_combination_unsupported`.
