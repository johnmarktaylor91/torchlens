# The same three lines on any architecture

The same three lines, `tl.trace`, `trace.draw()` and one `do`, on six architecture families
from the [Model Menagerie](https://modelmenagerie.ai), plus one non-PyTorch backend. The Menagerie
runs TorchLens on thousands of models across all major architecture families; the models here
are small, randomly initialised versions so the page runs on a CPU with nothing downloaded.


In every fence the last three lines are the same; only the model, the input and the site differ.

## Decoder

```python
# strip: family=decoder requires=transformers
import torch
import transformers

import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(
    vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2, use_cache=False
)
model = transformers.GPT2LMHeadModel(config).eval()
x = torch.tensor([[1, 2, 3, 4, 5, 6]])
site = tl.module("transformer.h.1.mlp")

trace = tl.trace(model, x)
trace.draw()
edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)
```

## CNN

```python
# strip: family=cnn requires=torchvision
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.rand(1, 3, 64, 64)
site = tl.module("layer3")

trace = tl.trace(model, x)
trace.draw()
edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)
```

## Vision transformer

```python
# strip: family=vision-transformer requires=torchvision
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.VisionTransformer(
    image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64
).eval()
x = torch.rand(1, 3, 32, 32)
site = tl.module("encoder.layers.encoder_layer_1.mlp")

trace = tl.trace(model, x)
trace.draw()
edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)
```

## Recurrent network

```python
# strip: family=recurrent requires=torch
import torch
from torch import nn

import torchlens as tl


class Tagger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.lstm = nn.LSTM(16, 16, batch_first=True)
        self.mix = nn.Linear(16, 16)
        self.head = nn.Linear(16, 5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(self.embed(tokens))
        return self.head(torch.tanh(self.mix(states)))


torch.manual_seed(0)
model = Tagger().eval()
x = torch.tensor([[3, 1, 4, 1, 5, 9]])
site = tl.module("mix")

trace = tl.trace(model, x)
trace.draw()
edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)
```

## Diffusion U-Net

```python
# strip: family=diffusion-unet requires=diffusers
import diffusers
import torch

import torchlens as tl

torch.manual_seed(0)
model = diffusers.UNet2DModel(
    sample_size=16,
    in_channels=3,
    out_channels=3,
    layers_per_block=1,
    block_out_channels=(16, 32),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
    norm_num_groups=8,
).eval()
x = (torch.randn(1, 3, 16, 16), torch.tensor([10]))
site = tl.module("mid_block")

trace = tl.trace(model, x)
trace.draw()
edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)
```

## Graph network

```python
# strip: family=graph-net requires=torch_geometric
import torch
import torch_geometric

import torchlens as tl

torch.manual_seed(0)
model = torch_geometric.nn.GCN(
    in_channels=8, hidden_channels=16, num_layers=2, out_channels=4
).eval()
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
x = (torch.randn(4, 8), edge_index)
site = tl.module("convs.0")

trace = tl.trace(model, x)
trace.draw()
edited = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)
```

## Another framework: JAX

The JAX backend records and draws a JAX function through its jaxpr; edits
are not supported there yet, so this fence has two of the three lines. Its capability table is
on [the JAX backend page](../backends/jax.md).

```python
# strip: family=preview-jax requires=jax
import jax.numpy as jnp

import torchlens as tl


def model(params, x):
    hidden = jnp.tanh(x @ params["in_proj"])
    return hidden @ params["out_proj"]


params = {"in_proj": jnp.ones((3, 4)) * 0.1, "out_proj": jnp.ones((4, 2)) * 0.1}
x = (params, jnp.ones((2, 3)))

trace = tl.trace(model, x, backend="jax")
trace.draw()
```

## Further

- Structure-only Menagerie entries are not executed checks.
- Per-backend capabilities: [backends](../backends.md).
