# Coming from torchview

This page maps torchview's `draw_graph` to TorchLens: their spelling on the left, the
TorchLens spelling on the right. In TorchLens, drawing is a method of the trace:
capture once with `tl.trace(model, x)`, then call `trace.draw(...)` as often as you like.

| torchview `draw_graph(...)` argument | TorchLens | Notes |
| --- | --- | --- |
| `draw_graph(model, input_size=(1, 3, 32, 32))` | `tl.trace(model, torch.randn(1, 3, 32, 32)).draw()` | TorchLens runs a real input; build one of the size you would pass. |
| `input_data=x` | `tl.trace(model, x)` | Tuples, dicts and keyword inputs work as in `model(...)`. |
| `depth=3` | `trace.draw(depth=3)` | How many levels of nested modules are drawn. |
| `roll=True` | `trace.draw(vis_mode="rolled")` | Repeated passes through one module are drawn once. |
| `graph_dir="LR"` / `"TB"` | `trace.draw(direction="leftright")` / `"topdown"` | `"bottomup"` is the default. |
| `hide_inner_tensors=True` (the default) | `trace.draw(collapse="auto")` | TorchLens draws every executed op by default; `collapse=` folds modules into boxes for an overview (`"none"`, `"auto"`, `"max"` or a level from 0 to 1). |
| `save_graph=True, filename=..., directory=...` | `trace.draw(path="dir/name", file_format="png", vis_save_only=True)` | Without `vis_save_only=True` the file also opens. |
| `graph.visual_graph` (a `graphviz.Digraph`) | `trace.draw(..., return_graph=True)` | By default `draw` returns the DOT source as a string. |
| `mode="eval"`, `device=...`, `dtypes=...` | Set them on your model and input before tracing | |

Their construct:

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

TorchLens equivalent:

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
dot_source = trace.draw(depth=2, path=outpath, file_format="svg", vis_save_only=True)
RESULT = ("conv2d" in dot_source, os.path.exists(outpath + ".svg"))
```

## The GPT-2 contrast

torchview records module calls and the tensors between them, so its picture of a model
is the module tree. TorchLens records every executed operation, including the
functional ones a module tree does not show (the attention arithmetic, reshapes,
additions of the residual stream). On a transformer that is many more nodes, which is
why the readable overview comes from `collapse="auto"`, while the full graph stays one
call away. The example builds a two-block GPT-2 from a config (no download) and counts
both views:

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
overview = trace.draw(collapse="auto", path=outpath, file_format="svg", vis_save_only=True)
RESULT = (executed_ops > module_calls, os.path.exists(outpath + ".svg"))
```
