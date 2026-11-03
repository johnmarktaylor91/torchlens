# Coming from nnsight

This page is for people who read and edit activations with `model.trace(...)` blocks. The rows and
both examples were checked with nnsight 0.8.0rc1 (the 0.8 release candidate, published 2026-09-09;
the latest release is v0.8.2, published 2026-11-02) and transformers 5.18.0 (torch 2.13). In
transformers 5, a GPT-2 block returns its hidden states as a tensor, so the older `.output[0]`
tuple idiom is gone from both sides. Checked 2026-11-03.

## The same task in both

The task: overwrite the first two features of block 0's output with 2.5 and read the edited block
output.

### In nnsight

Needs nnsight; the fence is skipped where it is not installed.

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

### In TorchLens

```python
# migration-test: tool=torchlens expected=([1, 4, 32], True, True)
import torch
import transformers
import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])


def edit(activation, *, ctx):
    activation = activation.clone()
    activation[:, :, :2] = 2.5
    return activation


trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(replayable=True))
edited = trace.fork("edited")
edited.do(tl.module("transformer.h.0"), edit)
block0 = edited["transformer.h.0"].out
logits_changed = not torch.equal(
    edited[edited.output_layers[0]].out, trace[trace.output_layers[0]].out
)
RESULT = (list(block0.shape), bool((block0[:, :, :2] == 2.5).all()), logits_changed)
```

## Translation table

| nnsight operation (0.8.0rc1, transformers 5.18.0) | TorchLens (transformers 5.18.0) | Notes |
| --- | --- | --- |
| `with model.trace(ids): h = model.transformer.h[0].output.save()` | `tl.extract(model, ids, ["transformer.h.0"])["transformer.h.0"]` | TorchLens runs the forward first and you read the result afterwards. |
| `model.transformer.h[0].output[:, :, :2] = 2.5` inside the trace | Capture with `capture=tl.options.CaptureOptions(replayable=True)`, then `trace.fork().do(tl.module("transformer.h.0"), edit)` | `edit` is a tensor or a function of the original output; `do` replays the recorded graph downstream. |
| Save every op, including ones no module owns (`.source`) | Nothing to ask for: `tl.trace` records every executed op (`trace.layers`) | |
| `model.trace(...)` on the wrapped model | `tl.trace(model._model, ids)` on the model nnsight wraps | See Using both. |
| Trace a model invocation | `tl.trace(model, x, save=tl.func("relu"))` | TorchLens stores a completed `Trace` after execution. |
| Save a node value | `save=selector`, then `log.find_sites(selector).first().out` | Equivalent for captured activations. |
| Assign into a traced value | `log.fork().do(site, value)` (replay the recorded graph) or `.run(model, x)` (run the model again) | TorchLens recipes are explicit and auditable. |
| Invoke with intervention in one block | `tl.trace(model, x, intervene=tl.when(site, helper_or_value), save=site)` | Capture-time intervention is the direct one-forward path. |
| Scan modules by path | `tl.in_module("path")`, `tl.module("path")`, `log.find_sites(...)` | Similar workflow; labels differ. |
| Cross-prompt patching | Capture separate traces and patch with `tl.patch_from(source_trace)` or `attach_hooks` | Equivalent for local PyTorch models. |
| Repeated interventions from one base | `fork()` first, then edit each branch | TorchLens uses fork-first branch isolation. |
| Save an intervention for sharing | `log.save_intervention(path, level="portable")` | TorchLens has `.tlspec/` recipes. |
| Intervene inside fused attention internals | Default fused SDPA load: by-head and by-pattern patching through `torchlens.mechinterp.patch_head` (no eager reload); eager-attention load: the softmax and score ops are ordinary intervention sites | Head-level and pattern-level edits work on fused models; other edits inside the fused kernel need an eager recapture. |
| Remote execution (NDIF), vLLM and tensor-parallel back ends | No equivalent | TorchLens captures a local eager forward. |

## What TorchLens adds

- Verified capture: `tl.validate` replays the recorded graph and checks it against the real
  forward.
- Saved and diffable traces: `.tlspec/` persistence, `tl.debug.compare`, and `trace.fork()` branch
  isolation.
- Typed refusals: unsupported cases raise a named error with a remedy instead of failing partway.
- Source lines and pass identity per operation, and the executed dataflow graph
  (`trace.layers`).

Module-less operations, diffusers pipelines, quantized models and vision-language models are
shared ground with nnsight 0.8, not a TorchLens-only capability. TorchLens records module-less
operations by default (no `.source` step), and treats a vision-language model as an ordinary
PyTorch model (see [landscape.md](landscape.md)).

## What nnsight does that TorchLens does not

- Remote execution on NDIF, vLLM and tensor-parallel back ends, and `scan`. TorchLens captures a
  local eager forward; for serving-scale capture see
  [where to go instead](where_to_go_instead.md#capture-at-serving-scale-on-remote-gpus-or-across-devices)
  and [landscape.md](landscape.md), which lists
  Neuronpedia's interp-engine and vllm-lens beside nnsight.
- Source-local tracing syntax: edits are written inline inside a `with model.trace(...)` block and
  read as ordinary indexing. TorchLens separates capture from edit, so the same task is two steps.

## Using both

### nnsight feeds TorchLens

There is no bridge module. An `NNsight` (or `LanguageModel`) object keeps the PyTorch module it
wraps as `model._model`. Pass that module to `tl.trace` or `tl.extract`; the NNsight wrapper itself
is not a plain `nn.Module` forward.

### TorchLens feeds nnsight

There is no bridge. A tensor from `trace[label].out` is a plain `torch.Tensor`, so it can be
assigned into an nnsight trace as a value, for example as a patch source captured from another
prompt.

## Semantic differences

- Order. nnsight code inside `with model.trace(...)` is deferred and runs when the block exits;
  TorchLens runs the forward first and you read results afterwards.
- Edit scope. An nnsight assignment edits the live forward. `trace.fork().do(...)` without
  `model=` and `inputs=` pushes the edit down the recorded graph from saved values and does not
  call `model.forward`; with both it re-executes (see
  [intervention_explainers.md](../intervention_explainers.md)). Pushing is exact only for
  graph-stable edits.
- The edit in the example must have a stored graph to replay: capture with `replayable=True`.
