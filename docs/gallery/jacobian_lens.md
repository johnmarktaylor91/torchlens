# A Jacobian lens across two architectures

The Jacobian lens transports residual vectors into the final layer's basis; this entry
compares the estimator with the paper's reference implementation on GPT-2 small and exact
autograd on Mamba-130m, with measurements pending.

| Field | Value |
| --- | --- |
| Claim | *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits, 6 July 2026, arXiv 2607.15495 v1 |
| Script and run line | `python examples/gallery/jacobian_lens.py --out outputs/jacobian_lens --reference-repo URL` (supply the paper's reference repository URL) |
| Pinned inputs | GPT-2 small `607a30d783dfa663caf39e06633721c8d4cfcd7e`; Mamba-130m `1e76775f628fbf1350fbe4dbb3d971ba64af25a1`; reference commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`; embedded two-sentence CC0 corpus; seed 0 |
| Verdict | Not run yet; no recorded machine, date, version or commit |
| Cost | Not measured |

## Oracles

For every pre-final block, the relative Frobenius error of its mean Jacobian must be below
1e-3, and top-ten token overlap must be at least 0.9 on every held-out prompt. The fitting
corpus is two original sentences embedded in the script, written verbatim as `corpus.txt`
and hashed. This small demonstration corpus is not a fit of the paper's full dataset.
Both estimators sum cotangents across target positions and average gradients across source
positions, excluding the first and final token. Prompt Jacobians receive equal weight.

The GPT-2 oracle is the paper's reference implementation at the pinned commit, checked out
inside the output directory. Mamba is outside that reference's recognized layouts; its
oracle computes every exact row with scalar reverse-mode autograd and ordinary block
hooks. The TorchLens estimator instead reads live outputs from `save_mode="view"` traces
and batches the vector-Jacobian products. Final normalization and unembedding belong to
the readout, outside the fitted Jacobian. Measurements are pending.

## Controls

Transposing a fitted Jacobian must fail the relative-error bar (error at least 1e-3).
Applying the Jacobian from a half-stack-distant layer must lower mean held-out top-ten
overlap below 0.9. These rules are fixed before running. A control that stays above its
bar fails the entry. Neither control has been measured yet.

## Artifact and replay

For each architecture, the script saves the final fitting trace as a portable `.tlspec`
and all fitted Jacobians as a tensor-only `.pt` sidecar. The trace records the sidecar's
SHA-256 and a held-out residual. Replay checks the sidecar hash and recomputes the readout
using the same pinned model's final norm and unembedding. It must reproduce the top-ten
token ids exactly. Model weights are a required replay input. No artifacts exist yet.

## Basis-grid row

Core verbs: capture, read, save. The TorchLens region counts fitting and held-out capture
code; the plain region counts the exact autograd estimator, decoder and overlap function.
Model loading, reference checkout and reporting are outside both regions. The pinned
reference's line count is not measured. Intended architectures: GPT-2 small and Mamba-130m.

## The stamp

```text
No completed measurement. The script prints STAMP and writes jacobian_lens.json on completion.
```

## Vary it

`--layer` selects the layer used in the controls and saved replay. `--prompt` supplies a
held-out readout prompt. `--arch` chooses `gpt2`, `mamba` or `both`; the gallery coverage
oracle requires both pretrained architectures. `--quick` uses a randomly initialized
GPT-2 to exercise the supported-model path and cannot pass the coverage oracle.

The executed stand-in below compares the live-state Jacobian with ordinary hooks and
scalar autograd. It downloads nothing.

```python
# recipe-test: tool=transformers expected=True
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

torch.manual_seed(0)
config = GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=32,
                   n_positions=16, use_cache=False)
config._attn_implementation = "eager"
model = GPT2LMHeadModel(config).eval()
ids = torch.tensor([[3, 7, 1, 9, 4]])
seen = {}

def retain(index):
    def hook(module, args, output):
        seen[index] = output[0] if isinstance(output, tuple) else output
    return hook

handles = [block.register_forward_hook(retain(i))
           for i, block in enumerate(model.transformer.h)]
try:
    model(ids)
finally:
    for handle in handles:
        handle.remove()
target = seen[1][0, 1:-1].sum(0)
plain = torch.stack([
    torch.autograd.grad(value, seen[0], retain_graph=True)[0][0, 1:-1].mean(0)
    for value in target
])
trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(save_mode="view"))
source = trace["transformer.h.0"].out
target = trace["transformer.h.1"].out[0, 1:-1].sum(0)
captured = torch.autograd.grad(target, source, grad_outputs=torch.eye(8),
                               is_grads_batched=True)[0][:, 0, 1:-1].mean(1)
RESULT = torch.allclose(captured, plain, rtol=1e-3, atol=1e-6)
```
