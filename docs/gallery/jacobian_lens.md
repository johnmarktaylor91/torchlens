# A Jacobian lens across two architectures

The Jacobian lens transports residual vectors into the final layer's basis. This entry fits
the lens from live-state captures and compares it with the paper's reference implementation
on GPT-2 small and with exact scalar autograd on Mamba-130m, an architecture the reference
repository cannot run. The entry passes: the fitted Jacobians match both independent
estimators at every pre-final layer, the held-out readouts agree exactly, both portable
artifacts reload and replay, and the two controls fail their bars as declared.

| Field | Value |
| --- | --- |
| Claim | *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits, 6 July 2026, arXiv 2607.15495 v1 |
| Script and run line | `python examples/gallery/jacobian_lens.py --out outputs/jacobian_lens --reference-repo URL` (supply the paper's reference repository URL) |
| Pinned inputs | GPT-2 small `607a30d783dfa663caf39e06633721c8d4cfcd7e`; Mamba-130m `1e76775f628fbf1350fbe4dbb3d971ba64af25a1`; reference commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`; embedded two-sentence CC0 corpus (SHA-256 `5ddd996b8358a88420bf5076c4166c1beba68c003e68fb42f1c212616923624e`); seed 0 |
| Verdict | Passed; CPU Linux runner, 3 November 2026, TorchLens 2.36.3, commit `994796df4d098d98b95089539a1ca58393a0a74d` |
| Cost | 2 CPU cores, 1,777.5 s script wall time (29 min 38 s), 5,175,792 KiB peak process RSS |

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
the readout, outside the fitted Jacobian.

| Oracle | Rule | Measured |
| --- | --- | --- |
| GPT-2 Jacobian | relative Frobenius error below 1e-3 at every pre-final layer | 0.0 at all 11 layers |
| GPT-2 readout | top-ten overlap at least 0.9 at every layer on every held-out prompt | 1.0 at all 11 layers on all 3 prompts |
| GPT-2 replay | portable trace and companion matrices reproduce the held-out readout | matched |
| Mamba Jacobian | relative Frobenius error below 1e-3 at every pre-final layer | between 4.8e-7 (layer 0) and 4.8e-8 (layer 22) over 23 layers |
| Mamba readout | top-ten overlap at least 0.9 at every layer on every held-out prompt | 1.0 at all 23 layers on all 3 prompts |
| Mamba replay | portable trace and companion matrices reproduce the held-out readout | matched |
| Coverage | both pretrained GPT-2 small and Mamba-130m measured | both measured |

The GPT-2 errors are exactly zero because the reference implementation and the TorchLens
estimator take the same autograd path over the same residual tensors; the Mamba errors are
float32 round-off between batched and scalar reverse mode over the sequential scan.

## Controls

Transposing a fitted Jacobian must fail the relative-error bar (error at least 1e-3).
Applying the Jacobian from a half-stack-distant layer must lower mean held-out top-ten
overlap below 0.9. These rules are fixed before running. A control that stays above its
bar fails the entry.

| Control | Measured |
| --- | --- |
| GPT-2 transpose, layer 5 | relative error 1.1439 |
| GPT-2 wrong layer, layer 10's Jacobian applied at layer 5 | overlaps 0.1, 0.1, 0.1 (mean 0.1) |
| Mamba transpose, layer 5 | relative error 1.3322 |
| Mamba wrong layer, layer 16's Jacobian applied at layer 5 | overlaps 0.2, 0.4, 0.4 (mean 0.333) |

## Artifact and replay

For each architecture, the script saves the final fitting trace as a portable `.tlspec`
and all fitted Jacobians as a tensor-only `.pt` sidecar. The trace records the sidecar's
SHA-256 and a held-out residual. Replay checks the sidecar hash and recomputes the readout
using the same pinned model's final norm and unembedding. It must reproduce the top-ten
token ids exactly. Model weights are a required replay input. Both replays matched.

The Mamba trace is 415,280,294 bytes with gallery sorted-path/content SHA-256
`5ff1c7dd6f49185f5d62bbf50525b87bd4736fba70c5ee705f9e9c898ded4cb7`; its matrix sidecar is
54,271,174 bytes, SHA-256 `bd42f446eefc3c425075fc832462f4ddfcad7b963223d4fb5f4d9802dd7dc870`.
The GPT-2 trace is 52,548,070 bytes, SHA-256
`912c70fbd5881c3d4369fb99616ae793bad79c3a93a8b50c5ae3fc80840fb840`; its sidecar is
25,956,473 bytes, SHA-256 `ece32eff20c8dcbd9c6fab7f7babeecc445eadfc15fee4f1bed5a933e80e8112`.

An earlier attempt at this entry stopped here: the Mamba trace, with about 24,000 ops each
holding one gradient record per backward pass of the fit, saved 123 MB of metadata and
`tl.load` refused it at the loader's 24,000,000-opcode allocation ceiling. The ceiling is
unchanged. The persisted gradient records and the recorded grad_fn calls are now stored as
one columnar table per op, so the opcode count grows with the number of ops, not with ops
times backward passes, and the same trace reloads. A first rerun after that fix exceeded
its 90-minute budget because backward-pass recording had become about six times slower in
between (every grad_fn resolved its op by scanning the op list); the lookup now reads the
label index, and this run's fit took its expected time.

The script also writes `jacobian_lens.html`, a side-by-side layer-by-position top-token
grid for the ASCII-face prompt, and one `*_readout_grid.json` per architecture containing
the exact displayed data. The grid keeps token whitespace and full cell text. It shows
this miniature-corpus fit's readouts and does not assume that the paper's full-corpus
semantic example will reappear.

## Basis-grid row

Core verbs: capture, read, save. The TorchLens region counts fitting and held-out capture
code; the plain region counts the exact autograd estimator, decoder and overlap function.
Model loading, reference checkout and reporting are outside both regions. The pinned
reference's line count is not measured. Marked source counts are 41 TorchLens lines and
33 plain lines. Architectures: GPT-2 small and Mamba-130m.

## The stamp

```text
{"commit": "994796df4d098d98b95089539a1ca58393a0a74d", "date_utc": "2026-11-03T22:03:52Z",
 "device": "cpu", "libraries": {"numpy": "2.2.6", "torchvision": "0.28.0+cpu",
 "transformers": "5.18.0"}, "machine": "x86_64",
 "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31", "python": "3.10.19",
 "torch": "2.13.0+cpu", "torch_threads": 2, "torchlens": "2.36.3", "wall_seconds": 1777.5}
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
