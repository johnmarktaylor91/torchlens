# Training an SAE on TorchLens activations

TorchLens does not train sparse autoencoders or transcoders; it extracts the activations and hands
them to the trainer you choose, such as SAELens or your own loop. The hand-off has three steps:

1. `tl.extract_dataset(model, stimuli, layers, output_dir=...)` runs the model over the dataset in
   batches and writes the named layers to disk, one immutable shard per batch, with a manifest
   that records the model's identity and the input order.
2. `torchlens.dataset_extraction.open_extraction(output_dir)` opens the artifact lazily, and
   `torchlens.dataset_extraction.as_torch_dataset(reader, keys=[...])` wraps it as a map-style
   `torch.utils.data.Dataset` whose items are `{layer: activation}` rows.
3. A `DataLoader` over that dataset feeds the trainer: your own loop, dictionary_learning's
   `trainSAE`, sparsify's `SparseCoder`, or SAELens configured to train from pre-computed
   activations where your SAELens version supports that (check its documentation for the entry
   point). The [trainer recipe](field-guide/sae_trainers.md) runs dictionary_learning and sparsify on
   extracted activations and splices each trained SAE back.

The example runs all three steps on a small model, with a few optimizer steps of a minimal
plain-PyTorch SAE standing in for the trainer:

```python
# docs-test: expected=[True, True]
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchlens as tl
from torchlens.dataset_extraction import as_torch_dataset, open_extraction

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
stimuli = torch.randn(64, 8)
output_dir = Path(DOCS_TMPDIR) / "relu-activations"  # any empty directory of yours

# 1. Extract the ReLU output for every stimulus, 16 per forward.
tl.extract_dataset(model, stimuli, ["1"], batch_size=16, output_dir=output_dir, progress=False)

# 2. Open the artifact and wrap it as a torch Dataset.
reader = open_extraction(output_dir)
dataset = as_torch_dataset(reader)
key = reader.keys[0]

# 3. Feed a trainer; here, a few steps of a tiny SAE.
sae = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
losses = []
for epoch in range(3):
    for batch in DataLoader(dataset, batch_size=16, shuffle=True):
        activations = batch[key].float()
        loss = ((sae(activations) - activations) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

RESULT = [len(dataset) == 64, losses[-1] < losses[0]]
```

`DOCS_TMPDIR` is a scratch directory the docs test supplies; use any directory of your own.

## After training

- To test what the trained SAE's features do, splice it back into the model with
  `torchlens.bridge.sae.splice(model, x, site=..., sae=..., latents_edit=...)`; any object with
  `encode` and `decode` works. The [SAELens page](migration/saelens.md) maps SAELens's own
  operations, and `notebooks/sae_splice_tutorial.ipynb` runs the splice experiment end to end.
- `notebooks/external_training_tutorial.ipynb` walks the same extraction hand-off for other
  external training loops.
- Large models: name the layers you will train on in `layers=`, never the whole graph; see
  [long sequences](performance.md#long-sequences-and-long-contexts).
