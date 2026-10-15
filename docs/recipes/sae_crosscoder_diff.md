# Model diffing: two traces and a crosscoder from dictionary_learning

A crosscoder is one sparse dictionary trained on the activations of two
models at once: every latent has one decoder direction per model. Latents
whose decoder norm is about the same in both models are shared; latents whose
norm is large in one model and near zero in the other are specific to it. That
is the standard way to diff a base model against a fine-tune (Lindsey et al.,
2024, "Sparse Crosscoders"; Minder et al., 2025, on chat-tuning).

TorchLens supplies the paired activations: trace both models on the same
inputs and stack the same site. dictionary_learning's `CrossCoderTrainer`
trains the crosscoder on the pairs.

```text
pip install torchlens
pip install "dictionary_learning @ git+https://github.com/science-of-finetuning/crosscoder_learning@d1fd6f3fea41fc1123699cc09f4cdad47ca35160" "nnsight>=0.6" wandb
```

## Pair two traces

A small MLP stands in for the base model, and a copy with four of its hidden
units changed stands in for the fine-tune. Both are traced on the same
stimuli and the same site, the ReLU (module `"1"`), and the two activations
are stacked as `[examples, models, features]`, the layout the crosscoder
reads:

```python
# recipe-test: tool=crosscoder expected=((512, 2, 16), True)
import copy

import torch
from torch import nn

import torchlens as tl

torch.manual_seed(0)
base = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
tuned = copy.deepcopy(base)
with torch.no_grad():
    tuned[0].weight[:4] = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
stimuli = torch.randn(512, 8)

base_trace = tl.trace(base, stimuli)
tuned_trace = tl.trace(tuned, stimuli)
# detach(): a trace keeps each activation's autograd history; the trainer needs plain data.
paired = torch.stack([base_trace["1"].out, tuned_trace["1"].out], dim=1).detach()

base_to_relu, tuned_to_relu = base[:2], tuned[:2]
with torch.no_grad():
    by_hand = torch.stack([base_to_relu(stimuli), tuned_to_relu(stimuli)], dim=1)
RESULT = (tuple(paired.shape), bool(torch.equal(paired, by_hand)))
```

For a dataset too large for one forward, run `tl.extract_dataset` once per
model with the same stimuli and site, and stack row by row.

## Train the crosscoder

`trainSAE` takes any iterator of batches; each batch is `[batch, models,
features]`.

```python
# recipe-test: tool=crosscoder expected=('CrossCoder', True)
from dictionary_learning.trainers import CrossCoderTrainer
from dictionary_learning.training import trainSAE


def pair_batches(epochs, batch_size=64):
    generator = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for rows in torch.randperm(len(paired), generator=generator).split(batch_size):
            yield paired[rows]


crosscoder = trainSAE(
    data=pair_batches(epochs=40),
    trainer_config={
        "trainer": CrossCoderTrainer,
        "num_layers": 2,
        "activation_dim": 16,
        "dict_size": 48,
        "lr": 1e-2,
        "l1_penalty": 1e-3,
        "warmup_steps": 20,
        "layer": 1,
        "lm_name": "base-vs-tuned",
        "device": "cpu",
        "seed": 0,
    },
    steps=300,
)
with torch.no_grad():
    recon = crosscoder(paired)
fvu = float(((recon - paired) ** 2).sum() / ((paired - paired.mean(0)) ** 2).sum())
RESULT = (type(crosscoder).__name__, fvu < 0.5)
```

## Read the diff from the decoder norms

`crosscoder.decoder.weight` is `[models, latents, features]`. The relative
decoder norm `tuned / (base + tuned)` is near 0.5 for a shared latent, near 1
for a latent only the tuned model uses and near 0 for one only the base model
uses. The units the fine-tune left alone are identical in both models, so the
latents the crosscoder spends on them should be shared:

```python
# recipe-test: tool=crosscoder expected=((48,), True, True)
with torch.no_grad():
    norms = crosscoder.decoder.weight.norm(dim=-1)  # [models, latents]
    relative = norms[1] / (norms[0] + norms[1])
    # How much of each latent's decoder lies on the four changed units, per model.
    on_changed = crosscoder.decoder.weight[:, :, :4].norm(dim=-1) / norms.clamp_min(1e-12)
alive = norms.sum(0) > 1e-6
untouched = alive & (on_changed.max(0).values < 0.1)
RESULT = (
    tuple(relative.shape),
    bool(((relative >= 0) & (relative <= 1))[alive].all()),
    bool(((relative[untouched] - 0.5).abs() < 0.15).all()),
)
```

Sort `relative` to list the most model-specific latents, then read what they
do with the other recipes: their top examples in each model with
`tl.stats.TopK(..., encoder=...)` and the export for SAEDashboard
(`docs/recipes/sae_latent_export.md`).
