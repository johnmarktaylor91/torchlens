# Coming from torchview

This page is for people who draw model graphs with torchview's `draw_graph` and want the same
picture, or the full executed graph, from TorchLens. TorchLens's test environment does not pin
torchview; the torchview example below skips where torchview is not installed. In TorchLens,
drawing is a method of the trace: capture once with `tl.trace(model, x)`, then call
`trace.draw(...)` as often as you like. Checked 2026-11-03.

## The same task in both

Draw a small convolutional network two module levels deep and read the graph's DOT source.

### In torchview

```python
# migration-test: tool=torchview expected=True
import torch
from torch import nn
from torchview import draw_graph

torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 14 * 14, 10)).eval()
graph = draw_graph(model, input_size=(1, 3, 16, 16), depth=2)
RESULT = "Conv2d" in graph.visual_graph.source
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=(True, True)
import os
import tempfile

import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 14 * 14, 10)).eval()
trace = tl.trace(model, torch.randn(1, 3, 16, 16))
outpath = os.path.join(tempfile.mkdtemp(), "model")
dot_source = trace.draw(depth=2, path=outpath, file_format="svg").dot
RESULT = ("conv2d" in dot_source, os.path.exists(outpath + ".svg"))
```

## Translation table

| torchview `draw_graph(...)` argument | TorchLens | Notes |
| --- | --- | --- |
| `draw_graph(model, input_size=(1, 3, 32, 32))` | `tl.trace(model, torch.randn(1, 3, 32, 32)).draw()` | TorchLens runs a real input; build one of the size you would pass. |
| `input_data=x` | `tl.trace(model, x)` | Tuples, dicts and keyword inputs work as in `model(...)`. |
| `depth=3` | `trace.draw(depth=3)` | How many levels of nested modules are drawn. |
| `roll=True` | `trace.draw(rolled=True)` | Repeated passes through one module are drawn once. |
| `graph_dir="LR"` / `"TB"` | `trace.draw(direction="leftright")` / `"topdown"` | `"bottomup"` is the default. |
| `hide_inner_tensors=True` (the default) | `trace.draw(collapse="auto")` | TorchLens draws every executed op by default; `collapse=` folds modules into boxes for an overview (`"none"`, `"auto"`, `"max"` or a level from 0 to 1). |
| `save_graph=True, filename=..., directory=...` | `trace.draw(path="dir/name", file_format="png")` | The file is written and never opened unless `open_viewer=True`. |
| `graph.visual_graph` (a `graphviz.Digraph`) | `trace.draw(...)` | `draw` returns a `tl.viz.Drawing`; `str(drawing)` is the DOT source. |
| `mode="eval"`, `device=...`, `dtypes=...` | Set them on your model and input before tracing | |

## What TorchLens adds

- Every executed op is a node, including the functional ones a module tree does not show
  (attention arithmetic, reshapes, residual additions); `trace.layers` lists them.
- Folding and overviews are draw-time choices on one capture: `trace.draw(collapse="auto")`
  folds modules into boxes, and `fold_patterns=` folds named op chains (see
  [torchviz and hiddenlayer](from_torchviz_hiddenlayer.md)).
- The same trace carries the saved values and shapes of each node, so the picture and the
  data come from one capture (`trace[label].out`).
- `trace.draw(graph="backward", ...)` draws the backward graph of a captured run (see
  `docs/backward.md`).
- `trace.draw(...)` returns a `tl.viz.Drawing` whose `.dot` is the DOT source, so the picture
  can be post-processed with Graphviz tools.

## What torchview does that TorchLens does not

- It draws from `input_size=` alone, with no real input: it builds the input from the
  size itself. TorchLens runs a real input of the size you would pass.
- Its module-level view (`hide_inner_tensors=True`, the default) is the picture it was built
  for. TorchLens's equivalent is `collapse=`, which folds executed ops into module boxes but
  starts from the full op graph.
- Some of its drawing options have no one-for-one TorchLens keyword (for example
  `expand_nested`); `trace.draw` has its own keyword set, listed in its docstring.

See [where to go instead](where_to_go_instead.md) for tools outside TorchLens's scope.

## Using both

### torchview feeds TorchLens

There is no bridge. torchview draws a model you give it, and the same `nn.Module` goes to
`tl.trace`; nothing needs converting.

### TorchLens feeds torchview

There is no bridge. `trace.draw(...)` returns its own `tl.viz.Drawing`; it is not a
`graphviz.Digraph` built by torchview. If you need torchview's picture, call `draw_graph` on
the same model.

## Semantic differences

- **A real input.** torchview can draw from `input_size`; TorchLens runs a tensor you build of
  that size, so data-dependent control flow takes the branch that input takes.
- **Module tree versus executed ops.** torchview records module calls and the tensors between
  them, so its picture of a model is the module tree. TorchLens records every executed
  operation. On a transformer that is many more nodes, which is why the readable overview
  comes from `collapse="auto"`, while the full graph stays one call away. The example builds a
  two-block GPT-2 from a config (no download) and counts both views:

```python
# migration-test: tool=torchlens expected=(True, True)
import os
import tempfile

import torch
import transformers
import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
model = transformers.GPT2LMHeadModel(config).eval()
trace = tl.trace(model, torch.tensor([[1, 2, 3, 4]]))

module_calls = len(trace.modules)
executed_ops = len(trace.layers)
print(f"{module_calls} module calls, {executed_ops} executed ops")
outpath = os.path.join(tempfile.mkdtemp(), "gpt2")
overview = trace.draw(collapse="auto", path=outpath, file_format="svg")
RESULT = (executed_ops > module_calls, os.path.exists(outpath + ".svg"))
```
