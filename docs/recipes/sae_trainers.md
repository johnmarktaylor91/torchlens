# Train an SAE with dictionary_learning or sparsify on TorchLens activations

TorchLens extracts the activations, a trainer fits the sparse autoencoder,
and TorchLens splices the trained SAE back into the model to measure it. The
hand-off is four steps:

1. `tl.extract_dataset(model, stimuli, layers, output_dir=...)` writes the
   site's activations to disk, one shard per batch.
2. `reader.to_torch_dataset()` wraps them as a
   `torch.utils.data.Dataset`; a `DataLoader` batches them.
3. A trainer fits the SAE on those batches: dictionary_learning's
   `trainSAE`, or sparsify's `SparseCoder`.
4. `torchlens.bridge.sae.splice(model, x, site=..., sae=...)` puts the trained
   SAE's reconstruction back at the site and reports what it changed.

```text
pip install torchlens
pip install "dictionary_learning @ git+https://github.com/science-of-finetuning/crosscoder_learning@d1fd6f3fea41fc1123699cc09f4cdad47ca35160" "nnsight>=0.6" wandb
pip install "eai-sparsify==1.3.3"
```

The dictionary_learning pin is the crosscoder fork of the library (the
package is still `dictionary_learning`); its standard SAE trainers are the
same as upstream's, and the model-diffing recipe uses its crosscoder.

## 1 and 2. Extract the activations and wrap them

The model here is a small MLP; the site is its ReLU, module `"1"`, 16 wide.

```python
# recipe-test: tool=sae_trainers expected=(256, (16,))
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchlens as tl
from torchlens.dataset_extraction import open_extraction

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
stimuli = torch.randn(256, 8)
output_dir = Path("relu-activations")  # any empty directory of yours

tl.extract_dataset(model, stimuli, ["1"], batch_size=64, output_dir=output_dir, progress=False)
reader = open_extraction(output_dir)
dataset = reader.to_torch_dataset()
key = reader.keys[0]


def batches(epochs, batch_size=32):
    """Activation batches for a trainer, reshuffled every epoch with a fixed seed."""
    generator = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator):
            yield batch[key].float()


RESULT = (len(dataset), tuple(dataset[0][key].shape))
```

## 3a. dictionary_learning: `trainSAE`

`trainSAE` takes any iterator of activation batches, so the batches above go
in unchanged. `wandb` is installed with the library and stays disabled here.

```python
# recipe-test: tool=sae_trainers expected=('AutoEncoder', True)
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

ae = trainSAE(
    data=batches(epochs=10),
    trainer_config={
        "trainer": StandardTrainer,
        "activation_dim": 16,
        "dict_size": 64,
        "lr": 1e-2,
        "l1_penalty": 1e-3,
        "warmup_steps": 10,
        "layer": 1,
        "lm_name": "tiny-mlp",
        "device": "cpu",
        "seed": 0,
    },
    steps=80,
)
acts = torch.cat([batch[key] for batch in DataLoader(dataset, batch_size=256)]).float()
with torch.no_grad():
    fvu = float(((ae(acts) - acts) ** 2).sum() / ((acts - acts.mean(0)) ** 2).sum())
RESULT = (type(ae).__name__, fvu < 0.5)
```

## 4a. Splice it back

dictionary_learning's `AutoEncoder` has the `encode` and `decode` pair
`splice` expects, so it goes in as it is:

```python
# recipe-test: tool=sae_trainers expected=(True, True)
from torchlens.bridge.sae import splice

x = stimuli[:64]
result = splice(model, x, site="1", sae=ae)
with torch.no_grad():
    expected = ae.decode(ae.encode(result.clean_site_out))
RESULT = (
    bool(torch.allclose(result.reconstruction, expected)),
    result.fraction_variance_explained_pooled > 0.5,
)
```

## 3b. sparsify: `SparseCoder`

sparsify's `Trainer` runs its own forward over a Hugging Face model and token
dataset, so it cannot take extracted activations. The hand-off trains its
`SparseCoder` directly on the objective that `Trainer` minimises: the
`fvu` of the coder's forward (plus `auxk_loss` when `auxk_alpha` is set).

```python
# recipe-test: tool=sae_trainers expected=('SparseCoder', True)
from sparsify import SparseCoder, SparseCoderConfig

torch.manual_seed(0)
coder = SparseCoder(16, SparseCoderConfig(num_latents=64, k=4))
optimizer = torch.optim.Adam(coder.parameters(), lr=1e-2)
for activations in batches(epochs=10):
    loss = coder(activations).fvu
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    fvu = float(coder(acts).fvu)
RESULT = (type(coder).__name__, fvu < 0.5)
```

## 4b. Splice it back

sparsify's `encode` returns the top-k activations and their indices, not one
dense latent vector, so a three-line adapter gives `splice` the dense pair;
its reconstruction equals sparsify's own forward:

```python
# recipe-test: tool=sae_trainers expected=(True, True)
class DenseCoder:
    """sparsify SparseCoder as a dense encode/decode pair."""

    def __init__(self, coder):
        self.coder = coder

    def encode(self, x):
        top = self.coder.encode(x)
        dense = torch.zeros(*x.shape[:-1], self.coder.num_latents, dtype=top.top_acts.dtype)
        return dense.scatter(-1, top.top_indices, top.top_acts)

    def decode(self, latents):
        return latents @ self.coder.W_dec + self.coder.b_dec


result = splice(model, x, site="1", sae=DenseCoder(coder))
with torch.no_grad():
    own = coder(result.clean_site_out).sae_out
RESULT = (
    bool(torch.allclose(result.reconstruction, own, atol=1e-5)),
    result.fraction_variance_explained_pooled > 0.5,
)
```
