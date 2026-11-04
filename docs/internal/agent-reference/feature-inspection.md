# Edge shapes, source annotations and feature optimization

`Trace.draw(show_edge_shapes=True)` labels forward dataflow edges with the producer's
output shape. Separate outputs retain their own shapes; rolled edges list distinct
observed shapes. The default `False` leaves drawing bytes unchanged. The
`VisualizationOptions(show_edge_shapes=True)` group and the `debug_edge_shapes`
composition use the same channel. Backward gradient edges have no forward tensor label.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(3, 2), nn.ReLU()).eval()
x = torch.ones(1, 3)
trace = tl.trace(model, x)
# Requires the optional visualization dependencies and Graphviz.
drawing = trace.draw("feature-edges.svg", show_edge_shapes=True)
```

`trace.source.annotated()` returns numbered captured root `forward` source with op
labels beside each line. It uses capture-time text and call frames, including nested
ops at the root caller's line; helper bodies are not printed. Synthetic input, output
and buffer rows are omitted. A source-less capture or an artifact saved without source
raises `InvalidArgumentError` with code `source_unavailable` and a capture/save remedy.

```python
# Put the model in a Python source file so capture can obtain its source.
class SourceModel(nn.Module):
    def forward(self, x):
        return torch.relu(x)

trace = tl.trace(SourceModel(), torch.ones(1, 3))
annotated = trace.source.annotated()
assert "return torch.relu(x)" in annotated and "# ops:" in annotated
```

`tl.mechinterp.visualize_feature(model, site, *, inputs=None, input_shape=None,
unit=None, steps=64, lr=0.05, seed=0, sae=None)` uses the shared
`tl.when(site, action).bind(model)` mechanism. Module addresses, op labels and
structural selectors target one tensor-producing site firing exactly once. It returns
a plain `(optimized_input, objective_history)` tuple, with `steps + 1` float values.
Inputs must be floating and differentiable; integer token IDs are unsupported.
Exactly one of `inputs` (cloned) or `input_shape` (local seeded noise) is required.
`unit` is a channel integer, tuple of non-batch indices, or `None` for the full mean.
With `sae`, `encode` must preserve gradients and integer `unit` selects a last-axis
latent. `steps` and `lr` must be positive. Unsupported settings, empty/multiple fires,
and disconnected or non-finite objectives refuse with `feature_visualization_invalid`.

```python
model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False)).eval()
with torch.no_grad():
    model[0].weight.fill_(1)
image, values = tl.mechinterp.visualize_feature(
    model, "0", input_shape=(1, 1, 3, 3), unit=0, steps=4, lr=0.1, seed=0,
)
assert image.shape == (1, 1, 3, 3) and values[-1] > values[0]
```

The helper leaves mode, parameter values and parameter gradients as supplied. Normal
forward buffer updates still occur. It optimizes unconstrained inputs; the field guide's
[advanced recipe](../../field-guide/feature_visualisation.md#how-it-works) adds Fourier
parameterization, color decorrelation and jitter when those image priors are desired.
