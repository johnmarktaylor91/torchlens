# Seeded subsampling for `extract_dataset`

`tl.extract_dataset` has no `sample_size=` or `seed=` argument. Draw the
subsample yourself with a seeded generator, pass the chosen stimuli, and pass
their ORIGINAL identifiers as `stimulus_ids=`. The artifact then names exactly
which items were used, in row order, and a later reader never has to trust a
seed it cannot see.

## The recipe

```python
# recipe-test: tool=torchlens expected=True
import numpy as np
import torch
from torch import nn

import torchlens as tl
from torchlens.dataset_extraction import open_extraction

torch.manual_seed(0)
model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(144, 5)).eval()
stimuli = torch.randn(100, 3, 8, 8)
image_ids = [f"img{i:03d}" for i in range(len(stimuli))]

rng = np.random.default_rng(seed=1234)  # keep the seed with your analysis code
chosen = np.sort(rng.choice(len(stimuli), size=12, replace=False))

tl.extract_dataset(
    model,
    stimuli[torch.as_tensor(chosen)],
    ["1"],  # the ReLU output
    batch_size=4,
    path="subsample",
    stimulus_ids=[image_ids[i] for i in chosen],  # original ids, not 0..11
    progress=False,
)
reader = open_extraction("subsample")
RESULT = reader.stimulus_ids() == [image_ids[i] for i in chosen]
```

Row `i` of every stored tensor is stimulus `reader.stimulus_ids()[i]`, so the
rows line up with the original dataset by name:

```python
# recipe-test: tool=torchlens expected=True
from torchlens.dataset_extraction import load_extraction

loaded = load_extraction("subsample")
(key,) = loaded.activations
prefix = model[:2]  # conv + ReLU, the extracted site
with torch.no_grad():
    direct = prefix(stimuli[torch.as_tensor(chosen)])
RESULT = torch.allclose(loaded.activations[key], direct)
```

## Limits

- `stimulus_ids=` needs disk mode (`output_dir=`); in-memory extraction
  refuses it with `extraction_stimulus_ids_in_memory_unsupported`.
- The manifest records the identifiers, not your sampling seed. If a reader
  needs to redraw the same subsample, store the seed beside the artifact.
