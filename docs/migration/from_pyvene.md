# Migrating from Pyvene

Pyvene and its companion library pyreft are idle upstream. If you are starting new intervention
work, or moving pyvene code to a maintained stack, this page maps each pyvene construct to its
TorchLens spelling. [`pyvene.md`](pyvene.md) is the one-table summary.

The model shape is the same in both libraries: a site in the model, an intervention applied
there, and a run. Pyvene configures sites as `(layer, component)` pairs on a wrapped model;
TorchLens names sites with selectors on the unwrapped model (`tl.module("transformer.h.0")`,
`tl.func("relu")`, `tl.head(3, "q")`) and applies edits through `tl.when(site, action)`. Which
engine to run the edit with is on the [intervention explainers](../intervention_explainers.md#which-engine-when)
page.

| Pyvene construct | TorchLens spelling |
| --- | --- |
| `IntervenableConfig(representations=[{"layer": L, "component": C}])` | a selector: `tl.module(...)`, `tl.func(...)`, or a facet such as `tl.head(i, "q")` for one head's query |
| `ZeroIntervention` | `tl.zero_ablate()` |
| `VanillaIntervention` (interchange from a source run) | `tl.patch_from(source_trace, source_site=..., positions=...)` |
| `AdditionIntervention`, steering vectors | `tl.steer(direction, strength=..., dim=...)` |
| Generation under an intervention | `spec.bind(model).generate(ids, ...)` |
| Fixed-direction subspace edits | `tl.project_out(direction, ...)`, `tl.project_onto(direction, ...)` |
| `CollectIntervention` | a `save=` selector on `tl.trace` or `tl.record` |
| SAE features as the intervention basis | `torchlens.bridge.sae.splice(..., latents_edit=...)` |
| Trainable interventions (`RotatedSpaceIntervention`, DAS, pyreft) | your own training loop over a bound spec; TorchLens ships no trainer |
| A trained pyreft `LoreftIntervention` (`ReftModel.save` directory) | `tl.bridge.pyreft.read(directory)`, then `fork.do(artifact.subspace(site, positions=mask), artifact.edit)` |
| `unit_locations` | `positions=` on `tl.patch_from`, or `tl.units(label, indices)` |
| Saving the configuration | `trace.save_intervention(path, level=...)` writes a `.tlspec/` recipe |

## Zero-ablate a site

Pyvene side, for comparison only. Pyvene is not a TorchLens test dependency, so this fence is
collected but never run; check the arguments against the pyvene version you have:

```python
# migration-test: tool=pyvene expected=[[0.0, 0.0]]
import pyvene as pv
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


config = pv.IntervenableConfig(
    representations=[{"component": "proj.output", "intervention_type": pv.ZeroIntervention}]
)
intervenable = pv.IntervenableModel(config, model=Tiny())
_, counterfactual = intervenable(base=torch.tensor([[2.0, 3.0]]))
RESULT = counterfactual.detach().tolist()
```

The TorchLens side:

```python
# migration-test: tool=torchlens expected=[[0.0, 0.0]]
import torch
from torch import nn
import torchlens as tl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


model = Tiny()
x = torch.tensor([[2.0, 3.0]])
trace = tl.trace(
    model,
    x,
    save=tl.func("linear"),
    intervene=tl.when(tl.func("linear"), tl.zero_ablate()),
)
RESULT = trace["linear_1_1"].out.detach().tolist()
```

## Interchange from a source run

Pyvene's interchange intervention writes a source run's representation into the base run.
`tl.patch_from` does the same; `positions=` maps source positions to target positions, so the two
prompts may differ in length. The check below compares it with the same patch written as plain
PyTorch forward hooks.

```python
# migration-test: tool=torchlens expected=True
import torch
import torchlens as tl
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(0)
gpt2 = GPT2LMHeadModel(
    GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=16, vocab_size=64)
).eval()
source_ids = torch.tensor([[5, 6, 7, 8]])
base_ids = torch.tensor([[1, 2, 3]])
site = tl.module("transformer.h.0")

source = tl.trace(gpt2, source_ids)
patched = tl.trace(
    gpt2,
    base_ids,
    intervene=tl.when(site, tl.patch_from(source, positions={3: 2})),
)
output = patched[patched.output_layers[0]].out
logits = output.logits if hasattr(output, "logits") else output

# The same patch with plain hooks: copy source position 3 into base position 2.
# A GPT-2 block returns its hidden state alone or first in a tuple, by version.
store = {}
block = gpt2.transformer.h[0]


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def grab(module, args, out):
    store["h"] = hidden_of(out).detach().clone()


def put(module, args, out):
    hidden = hidden_of(out).clone()
    hidden[:, 2] = store["h"][:, 3]
    return (hidden, *out[1:]) if isinstance(out, tuple) else hidden


handle = block.register_forward_hook(grab)
with torch.no_grad():
    gpt2(source_ids)
handle.remove()


handle = block.register_forward_hook(put)
with torch.no_grad():
    reference = gpt2(base_ids).logits
handle.remove()
RESULT = torch.allclose(logits, reference, atol=1e-6)
```

## Steering during generation

```python
# migration-test: tool=torchlens expected=True
import torch
import torchlens as tl
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
lm = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
).eval()
ids = torch.randint(0, 128, (1, 6))
spec = tl.when(
    tl.module("model.layers.0.mlp"),
    tl.steer(torch.randn(32), strength=8.0, dim=-1),
)
bound = spec.bind(lm)
outputs = bound.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
RESULT = outputs.shape[-1] == ids.shape[-1] + 4 and bound.last_report is not None
```

A bound `generate` holds the edit across every decoding step, KV cache included; the
[KV-cache recipe](../guides/kv_cache.md) covers what that costs and how it differs from editing
the cache itself.

## Fixed-direction subspace edits

Pyvene's subspace interventions act on a learned or given basis. For a fixed direction,
`tl.project_out` removes the component along it and `tl.project_onto` keeps only that component.
A `tl.subspace(...)` selection is the selection-side spelling of the same idea:
`do(tl.subspace(site, basis, origin=...), edit)` edits the component in the span of the basis, and
`tl.patch_from(source)` there is the basis interchange (see
[Subspace edits](../intervention_explainers.md#subspace-edits-the-component-along-a-basis)).

```python
# migration-test: tool=torchlens expected=True
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
x = torch.randn(2, 4)
direction = torch.tensor([0.0, 1.0, 0.0, 0.0])
site = tl.module("0")
edited = tl.trace(
    model, x, save=site, intervene=tl.when(site, tl.project_out(direction, dim=-1))
)
RESULT = bool((edited.find_sites(site).first().out[:, 1] == 0).all())
```

## SAE features as the intervention basis

`torchlens.bridge.sae.splice` swaps a site's activation for an SAE's reconstruction and pushes the
change downstream; `latents_edit=` edits the encoded features first. Any object with `encode` and
`decode` works; the identity "SAE" below makes the effect easy to check. The
[SAELens page](saelens.md) covers pretrained SAEs.

```python
# migration-test: tool=torchlens expected=[True, True, True]
import torch
from torch import nn
import torchlens as tl
from torchlens.bridge.sae import splice

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(3, 4)


class IdentitySAE(nn.Module):
    def encode(self, h):
        return h

    def decode(self, z):
        return z


def drop_feature_0(z):
    z = z.clone()
    z[..., 0] = 0.0
    return z


plain = splice(model, x, site=tl.module("1"), sae=IdentitySAE())
edited = splice(model, x, site=tl.module("1"), sae=IdentitySAE(), latents_edit=drop_feature_0)
RESULT = [
    plain.reconstruction_mse == 0.0,
    plain.output_delta_max == 0.0,
    edited.output_delta_max > 0.0,
]
```

## Trainable interventions

Pyvene and pyreft train intervention parameters (rotations, low-rank adapters) inside their own
trainer. TorchLens does not ship a trainer: the training loop is yours. A bound spec runs the
model's own forward with the edit applied and keeps autograd, so any parameter the edit reads
trains with an ordinary optimizer while the model's weights stay frozen.

```python
# migration-test: tool=torchlens expected=[True, True]
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1)).eval()
for weight in model.parameters():
    weight.requires_grad_(False)
frozen = [weight.clone() for weight in model.parameters()]
x = torch.randn(8, 4)
target = torch.ones(8, 1)

vector = nn.Parameter(torch.zeros(4))


def add_vector(activation, *, ctx):
    return activation + vector


bound = tl.when(tl.module("1"), add_vector).bind(model)
optimizer = torch.optim.SGD([vector], lr=0.1)
losses = []
for _ in range(20):
    optimizer.zero_grad()
    loss = ((bound(x) - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

RESULT = [
    losses[-1] < losses[0],
    all(torch.equal(a, b) for a, b in zip(frozen, model.parameters())),
]
```

The same hook trains inside `tl.trace(..., intervene=...)` when you also want the graph of one
step, but a trace per step costs far more than the bound call.

## Reading a pyreft LoReFT artifact

TorchLens does not train a LoReFT intervention, but it reads one pyreft trained and applies it.
`tl.bridge.pyreft.read(directory)` takes the directory `ReftModel.save` wrote (`config.json` plus
`intkey_<key>.bin`, under the key's dotted or underscored name) and returns a `LoreftArtifact`
with `R`, `W` and `b` as saved, the layer and component pyreft names, and the provenance: each
file's sha256, each tensor's dtype, and the `repo=` and `revision=` you pass. It imports neither
pyreft nor pyvene. LoReFT's `h + R (W h + b - R^T h)` is then
`fork.do(artifact.subspace(site, positions=mask), artifact.edit)`: a `tl.subspace` over `R.T`
restricted to pyreft's `unit_locations`, with `tl.intervention.set_affine_coordinates(W, bias=b)`. pyreft
trains `R` in float32, orthonormal to about `1e-7`, so the subspace orthonormalises it once in
float64 and the saved edit records both the digest of `R` as read and that of the rows applied. The
edit saves at `level="portable"`. A directory that is not one LoReFT intervention refuses
`pyreft_artifact_invalid` or `pyreft_artifact_unsupported`. The edit acts on the site's trailing
axis, as pyreft's does on a `[batch, sequence, d]` block output. Pass the site from the trace,
`artifact.subspace(trace[label], positions=mask)`, to have its shape checked at once: a trailing
axis that is not `d` wide (a site of another width, or one whose feature axis is not last) refuses
`pyreft_site_invalid`, and a mask that does not cover the other axes refuses
`pyreft_positions_invalid`. A label is checked when `do()` resolves it.

Checked against pyreft 0.1.0 with a LoReFT artifact for TinyLlama from the Hugging Face Hub, on
one CPU thread: the `R`, `W` and `b` the reader returns hash equal to the ones pyreft loads, the
logits before the edited token are unchanged exactly, and the unedited trace equals the plain
transformers forward exactly. At and after the edited token, TorchLens and pyreft run on the same
machine differ by at most `1.24e-5` across four prompts, and TorchLens is within `1.24e-5` of
logits pyreft recorded on that machine (the bar is `2e-5`). pyreft's own logits move by up to
`5.4e-5` between machines, so compare against pyreft run where you run TorchLens.

The example writes a stand-in artifact in pyreft's file layout; with a real one, `directory` is the
folder pyreft saved (for example a Hugging Face snapshot) and `site` is the label of the decoder
layer's output that pyreft's `block_output` names.

```python
# migration-test: tool=torchlens expected=[True, True, True]
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
key = "layer.0.comp.block_output.unit.pos.nunit.1#0"
directory = Path(tempfile.mkdtemp())
(directory / "config.json").write_text(json.dumps({
    "intervention_types": ["<class 'pyreft.interventions.LoreftIntervention'>"],
    "representations": [[0, "block_output", "pos", 1, 2]],
    "sorted_keys": [key],
    "intervention_dimensions": [4],
}))
rotation = torch.linalg.qr(torch.randn(4, 2))[0]  # float32, as pyreft trains it
weight, bias = torch.randn(2, 4).bfloat16(), torch.randn(2).bfloat16()
torch.save({"weight": weight, "bias": bias, "rotate_layer": rotation}, directory / f"intkey_{key}.bin")

artifact = tl.bridge.pyreft.read(directory, repo="example/loreft", revision="0" * 40)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).double().eval()
tokens = torch.randn(1, 5, 4, dtype=torch.float64)
site = "linear_1_1"
positions = torch.zeros(1, 5, dtype=torch.bool)
positions[:, -1] = True  # pyreft's unit_locations: the last token

base = tl.trace(model, tokens, capture=tl.options.CaptureOptions(intervention_ready=True))
fork = base.fork()
fork.do(artifact.subspace(site, positions=positions), artifact.edit)
h, edited = base[site].out[0, -1], fork.push()[site].out
R, W, b = rotation.double(), weight.double(), bias.double()

RESULT = [
    # LoReFT at the last token (R orthonormalised in float64 moves it by float32 rounding) ...
    torch.allclose(edited[0, -1], h + R @ (W @ h + b - R.T @ h), atol=1e-5),
    torch.equal(edited[0, :-1], base[site].out[0, :-1]),  # ... and nothing elsewhere
    artifact.provenance["source_dtype"]["rotation"] == "float32",
]
```

## Batched evaluation over a dataset

Pyvene evaluates interventions batch by batch inside its own loop. In TorchLens, either call the
bound spec on each batch (no trace), or trace the first batch with the intervention and append
further batches with `trace.run(model, batch, replay=tl.options.ReplayOptions(append=True))` when
the graph is the same for every batch. The append constraints are on the
[intervention explainers](../intervention_explainers.md#append-constraints) page.
