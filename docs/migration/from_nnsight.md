# Migrating from NNsight

This page maps NNsight's tracing idioms to TorchLens: their spelling on the left, the
TorchLens spelling on the right. The rows and both examples were checked with nnsight
0.8.0rc1 (the 0.8 release candidate, published 2026-09-09; the latest stable release is
0.7.0) and transformers 5.18.0 (torch 2.13). In transformers 5, a GPT-2 block returns its hidden
states as a tensor, so the older `.output[0]` tuple idiom is gone from both sides.

| NNsight (0.8.0rc1, transformers 5.18.0) | TorchLens (transformers 5.18.0) | Notes |
| --- | --- | --- |
| `with model.trace(ids): h = model.transformer.h[0].output.save()` | `tl.extract(model, ids, ["transformer.h.0"])["transformer.h.0"]` | TorchLens runs the forward first and you read the result afterwards. |
| `model.transformer.h[0].output[:, :, :2] = 2.5` inside the trace | Capture with `capture=tl.options.CaptureOptions(replayable=True)`, then `trace.fork().set(tl.module("transformer.h.0"), edit, confirm_mutation=True).push()` | `edit` is a tensor or a function of the original output; `.push()` replays the recorded graph downstream. |
| Save every op, including ones no module owns (`.source`) | Nothing to ask for: `tl.trace` records every executed op (`trace.layers`) | |
| `model.trace(...)` on the wrapped model | `tl.trace(model._model, ids)` on the model NNsight wraps | See below. |
| Remote execution (NDIF), vLLM and tensor-parallel back ends | No equivalent | TorchLens captures a local eager forward. |

## Running TorchLens on the model NNsight wraps

An `NNsight` (or `LanguageModel`) object keeps the PyTorch module it wraps as
`model._model`. Pass that module to `tl.trace` or `tl.extract`; the NNsight wrapper
itself is not a plain `nn.Module` forward.

Their construct (needs nnsight; not run in the default test environment):

```python
# migration-test: tool=nnsight expected=([1, 4, 32], True)
import torch
import transformers
from nnsight import NNsight

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
hf_model = transformers.GPT2LMHeadModel(config).eval()
model = NNsight(hf_model)
ids = torch.tensor([[1, 2, 3, 4]])

with model.trace(ids):
    model.transformer.h[0].output[:, :, :2] = 2.5
    edited = model.transformer.h[0].output.save()

RESULT = (list(edited.shape), bool((edited[:, :, :2] == 2.5).all()))
```

TorchLens equivalent:

```python
# migration-test: tool=torchlens expected=([1, 4, 32], True, True)
import torch
import transformers
import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])


def edit(out):
    out = out.clone()
    out[:, :, :2] = 2.5
    return out


trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(replayable=True))
edited = trace.fork("edited").set(tl.module("transformer.h.0"), edit, confirm_mutation=True)
edited.push()
block0 = edited["transformer.h.0"].out
logits_changed = not torch.equal(
    edited[edited.output_layers[0]].out, trace[trace.output_layers[0]].out
)
RESULT = (list(block0.shape), bool((block0[:, :, :2] == 2.5).all()), logits_changed)
```
