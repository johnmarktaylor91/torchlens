# `tl.export` reference

`tl.export` writes a completed trace to static graph files, performance-viewer formats,
tabular files, tracker objects, and an xarray activation assembly. File exporters return the
written `path`; tracker exporters accept existing tracker objects and do not create runs. The
examples use `DOCS_TMPDIR` when the documentation test provides it, otherwise `/tmp`.

## Static graph files

### `svg`

`tl.export.svg(log, path, *, editable=True)` writes a lightweight SVG graph. `editable=True`
adds stable IDs and semantic CSS classes. [`html`](#html) provides a self-contained interactive
viewer of the same graph data.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.svg(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "trace.svg")
print(path.name, path.exists())
```

Output:

```text
trace.svg True
```

### `html`

`tl.export.html(log, path)` writes a self-contained HTML viewer with pan, zoom, and node-hover
support; it has no network dependency. See [`svg`](#svg) for the lighter static output.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.html(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "trace.html")
print(path.name, path.exists())
```

Output:

```text
trace.html True
```

### `model_explorer`

`tl.export.model_explorer(log, path)` writes a static JSON graph with nodes, edges, labels,
shapes, and memory fields for graph explorer tools. It is a data export, not a runnable model.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.model_explorer(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "graph.json")
print(path.name, path.exists())
```

Output:

```text
graph.json True
```

### `netron`

`tl.export.netron(log, path)` writes a lossy ONNX-shaped JSON description for static Netron-style
inspection. It deliberately is not a runnable ONNX model; use it only for graph inspection.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.netron(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "netron.json")
print(path.name, path.exists())
```

Output:

```text
netron.json True
```

## Performance formats

### `chrome_trace`

`tl.export.chrome_trace(log, path)` writes a Chrome tracing JSON timeline for one trace. Open the
file in Chrome's tracing viewer or another Chrome-trace-compatible tool. For a bundle, use
[`chrome_trace_diff`](#chrome_trace_diff).

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.chrome_trace(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "trace.json")
print(path.name, path.exists())
```

Output:

```text
trace.json True
```

### `chrome_trace_diff`

`tl.export.chrome_trace_diff(bundle, path)` writes one Chrome timeline comparing the members of a
TorchLens `Bundle`.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

bundle = tl.bundle({"first": tl.trace(nn.ReLU(), torch.ones(1, 2)), "second": tl.trace(nn.ReLU(), torch.zeros(1, 2))})
path = tl.export.chrome_trace_diff(bundle, Path(globals().get("DOCS_TMPDIR", "/tmp")) / "diff.json")
print(path.name, path.exists())
```

Output:

```text
diff.json True
```

### `speedscope`

`tl.export.speedscope(log, path)` writes an evented [speedscope](https://www.speedscope.app/)
profile using TorchLens operation durations.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.speedscope(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "profile.speedscope.json")
print(path.name, path.exists())
```

Output:

```text
profile.speedscope.json True
```

### `flamegraph`

`tl.export.flamegraph(log, path)` writes folded-stack text containing operation durations. Feed
that text to a flamegraph renderer; it is not itself an SVG.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.flamegraph(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "stacks.folded")
print(path.name, path.exists())
```

Output:

```text
stacks.folded True
```

### `memory_timeline`

`tl.export.memory_timeline(log, path)` writes a tensor-scope memory timeline. It reports tensor
bytes observed and retained by TorchLens, not allocator or process peak memory.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.memory_timeline(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "memory.json")
print(path.name, path.exists())
```

Output:

```text
memory.json True
```

## Tabular and activation formats

### `csv`

`tl.export.csv(log, path, **kwargs)` writes `Trace.to_pandas()` as CSV. Extra keywords are
forwarded to `DataFrame.to_csv`; [`json`](#json) and [`parquet`](#parquet) write the same table
in other forms.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.csv(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "trace.csv")
print(path.name, path.exists())
```

Output:

```text
trace.csv True
```

### `parquet`

`tl.export.parquet(log, path, **kwargs)` writes `Trace.to_pandas()` as Parquet. It requires
`pyarrow` (install `torchlens[tabular]` if it is unavailable).

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.parquet(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "trace.parquet")
print(path.name, path.exists())
```

Output:

```text
trace.parquet True
```

### `json`

`tl.export.json(log, path, *, orient="records", **kwargs)` writes `Trace.to_pandas()` as JSON.
`orient` and extra keywords are forwarded to pandas.

```python
from pathlib import Path
import torch
from torch import nn
import torchlens as tl

path = tl.export.json(tl.trace(nn.ReLU(), torch.ones(1, 2)), Path(globals().get("DOCS_TMPDIR", "/tmp")) / "trace.json")
print(path.name, path.exists())
```

Output:

```text
trace.json True
```

### `xarray`

`tl.export.xarray(log)` returns an `xarray.DataArray` of saved tensor outputs with
`presentation` and `neuroid` dimensions. All exported outputs must have the same presentation
count; xarray must be installed.

```python
import torch
from torch import nn
import torchlens as tl

assembly = tl.export.xarray(tl.trace(nn.ReLU(), torch.ones(1, 2)))
print(assembly.dims, assembly.shape)
```

Output:

```text
('presentation', 'neuroid') (1, 6)
```

## Tracker integrations

### `tensorboard`

`tl.export.tensorboard(log, writer, step=0, prefix="torchlens")` writes scalar and text summaries
to an existing object with TensorBoard's `add_scalar` interface, then returns that writer.

```python
import torch
from torch import nn
import torchlens as tl

class Writer:
    def __init__(self): self.calls = []
    def add_scalar(self, *args): self.calls.append(args)
    def add_text(self, *args): self.calls.append(args)
    def flush(self): pass

writer = Writer()
tl.export.tensorboard(tl.trace(nn.ReLU(), torch.ones(1, 2)), writer)
print(len(writer.calls))
```

Output:

```text
3
```

### `wandb`

`tl.export.wandb(log, run=None, name="torchlens_trace")` creates a Weights & Biases table and
logs it only if an existing run is supplied or active. It requires the `wandb` extra.

```python
import torch
from torch import nn
import torchlens as tl

result = tl.export.wandb(tl.trace(nn.ReLU(), torch.ones(1, 2)))
print(sorted(result))
```

Output:

```text
['artifact', 'table']
```

### `mlflow`

`tl.export.mlflow(log, client=None, prefix="torchlens")` prepares summary metrics and sends them
to an optional existing object with `log_metric`; it returns the metrics either way.

```python
import torch
from torch import nn
import torchlens as tl

metrics = tl.export.mlflow(tl.trace(nn.ReLU(), torch.ones(1, 2)))
print(sorted(metrics))
```

Output:

```text
['num_layers', 'num_saved_ops', 'total_activation_memory']
```

### `aim`

`tl.export.aim(log, run=None, prefix="torchlens")` prepares summary metrics and sends them to an
optional existing object with `track`; it returns the metrics either way.

```python
import torch
from torch import nn
import torchlens as tl

metrics = tl.export.aim(tl.trace(nn.ReLU(), torch.ones(1, 2)))
print(sorted(metrics))
```

Output:

```text
['num_layers', 'num_saved_ops', 'total_activation_memory']
```
