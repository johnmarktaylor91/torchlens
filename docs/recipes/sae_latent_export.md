# SAE latents and top examples for SAEDashboard and delphi

SAEDashboard draws feature dashboards and delphi writes and scores
automatic explanations of SAE latents. Both start from the same data: each
latent's activations and the examples where it fires most. TorchLens produces
that data and writes it in the files each tool already reads:

- `torchlens.bridge.sae.write_sae_dashboard(latents, tokens, path=...)` writes
  the JSON that SAEDashboard's `SaeVisData.load_json` loads (sae-dashboard
  0.8.0): per latent, its top-activation sequences.
- `torchlens.bridge.sae.write_delphi(latents, tokens, site=..., model_name=...,
  directory=...)` writes the latent cache delphi's `LatentDataset` loads
  (eai-delphi 0.1.3): the latent activations in safetensors splits, the token
  ids and a `config.json`.

`latents` is either the latents of one batch, such as `SpliceResult.latents`,
or the per-latent top-k over a whole dataset that
`tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` collects under
`tl.stats.aggregate`. `tokens` holds the token ids, one row per example. Neither
tool is needed to write the files.

```text
pip install torchlens
pip install "sae-dashboard==0.8.0"   # to render the dashboard
pip install "eai-delphi==0.1.3"      # to explain and score the latents
```

## One batch: the latents a splice kept

A tiny next-token model and a hand-built SAE stand in for a real model and a
pretrained SAE (see the pretrained-SAE recipe for loading one). `splice` keeps
the SAE latents of the batch, and both writers take them as they are.

```python
# recipe-test: tool=torchlens expected=(True, ['0_4.safetensors', '5_9.safetensors', 'config.json'], True)
import json
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.bridge.sae import splice, write_delphi, write_sae_dashboard


class TinyLM(nn.Module):
    def __init__(self, vocab=50, width=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.hidden = nn.Linear(width, width)
        self.unembed = nn.Linear(width, vocab)

    def forward(self, tokens):
        return self.unembed(torch.relu(self.hidden(self.embed(tokens))))


class TinySAE(nn.Module):
    def __init__(self, d_in=6, d_sae=10):
        super().__init__()
        self.enc = nn.Linear(d_in, d_sae)
        self.dec = nn.Linear(d_sae, d_in)

    def encode(self, x):
        return torch.relu(self.enc(x))

    def decode(self, latents):
        return self.dec(latents)


torch.manual_seed(0)
model, sae = TinyLM().eval(), TinySAE().eval()
tokens = torch.randint(0, 50, (8, 8), generator=torch.Generator().manual_seed(1))

result = splice(model, tokens[:4], site=tl.func("relu"), sae=sae, tokens=tokens[:4])
dashboard = write_sae_dashboard(result.latents, tokens[:4], k=3, path="batch/dashboard.json")
cache = write_delphi(
    result.latents, tokens[:4], site="hidden", model_name="gpt2", n_splits=2,
    directory="batch/latents",
)

data = json.loads(Path(dashboard).read_text())
first = next(iter(data["feature_data_dict"].values()))["sequence_data"]["seq_group_data"][0]
RESULT = (
    len(first["seq_data"]) == 3,
    sorted(path.name for path in Path(cache).iterdir()),
    data["feature_stats"]["max"][0] == float(result.latents[..., int(next(iter(data["feature_data_dict"])))].max()),
)
```

## A whole dataset: the top examples per latent

Over a dataset, keep only each latent's top examples while the batches go
by: `tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` encodes every batch
of site activations and keeps, per latent, the `k` largest activations with
the example and position each came from. The example id is the example's
index in the order `tl.stats.aggregate` saw the data, so row `i` of `tokens` must be
example `i`. The top examples equal a hand ranking of every activation (value
first, then the lowest example, then the lowest position):

```python
# recipe-test: tool=torchlens expected=(True, True)
from safetensors.numpy import load_file

relu = next(op.layer_label for op in tl.trace(model, tokens[:1]).ops if op.type == "relu")
stat = tl.stats.TopK(k=3, unit_axis=-1, encoder=sae.encode, name="latents")
batches = [tokens[0:3], tokens[3:6], tokens[6:8]]
top = tl.stats.aggregate(model, batches, metrics={relu: tl.stats.Aggregator(stat)})[relu]["latents"]

write_sae_dashboard(top, tokens, path="dataset/dashboard.json")
write_delphi(top, tokens, site="hidden", model_name="gpt2", directory="dataset/latents")

with torch.no_grad():
    every = sae.encode(torch.relu(model.hidden(model.embed(tokens))))  # [examples, positions, latents]
data = json.loads(Path("dataset/dashboard.json").read_text())
unit = int(next(iter(data["feature_data_dict"])))  # the first latent that fired
ranked = sorted(
    ((float(every[e, p, unit]), e, p) for e in range(8) for p in range(8)),
    key=lambda item: (-item[0], item[1], item[2]),
)[:3]
seqs = data["feature_data_dict"][str(unit)]["sequence_data"]["seq_group_data"][0]["seq_data"]
split = load_file("dataset/latents/hidden/0_1.safetensors")
RESULT = (
    [(s["original_index"], s["qualifying_token_index"]) for s in seqs] == [(e, p) for _, e, p in ranked],
    split["tokens"].tolist() == tokens.tolist(),
)
```

With a dataset top-k, the dashboard sequences carry the ranked token's
activation and 0.0 at every other token (SAEDashboard's own display when it
computes no context), and the delphi cache holds only the retained top
activations, so delphi's per-latent frequencies count those. For full
context, encode the top examples again and pass their dense latents.

## Open them in the tools

Not run here. SAEDashboard needs the model's tokenizer to render; the file has
whole sequences, so render them unwindowed (`buffer=None`) with the sequence
column only, since TorchLens writes no logit tables or histograms:

```text
from sae_dashboard.components_config import SequencesConfig
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_dashboard.layout import Column, SaeVisLayoutConfig
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData

layout = SaeVisLayoutConfig(columns=[Column(SequencesConfig(buffer=None, stack_mode="stack-none"))])
cfg = SaeVisConfig(hook_point="blocks.12.hook_resid_post", features=[], feature_centric_layout=layout)
vis = SaeVisData.load_json("dataset/dashboard.json", cfg=cfg, model=hooked_model, encoder=None)
save_feature_centric_vis(vis, "dashboard.html")
```

delphi reads the cache folder as its `raw_dir`; pass the site as the module
name and the tokenizer the model used:

```text
from delphi.config import ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset

dataset = LatentDataset(
    "dataset/latents", SamplerConfig(), ConstructorConfig(),
    tokenizer=tokenizer, modules=["hidden"],
)
for record in dataset:  # one LatentRecord per latent, ready for delphi's explainers
    ...
```
