# Causal tracing of MLP outputs (GPT-2 small)

Meng et al. (2022) corrupt a fact's subject tokens with noise, then restore one hidden state at
a time from the clean run and measure how much of the correct answer's probability comes back.
This entry measures the MLP-output grid on GPT-2 small with TorchLens and again with plain
PyTorch hooks on the same facts and noise draws. The entry passes: the two implementations agree exactly (max abs difference 0 over all 72 cells), both wrong-site controls recover almost nothing, and the largest effect sits at the last subject token as the paper reports, but at layer 0 on GPT-2 small rather than in the paper's middle-layer band on GPT-2 XL.

| | |
|---|---|
| Claim | Meng, Bau, Andonian and Belinkov, "Locating and Editing Factual Associations in GPT" (NeurIPS 2022, arXiv 2202.05262 v5), Section 2.2 and Figure 2b: restoring MLP outputs at the last subject token in early-to-middle layers recovers the most correct-answer probability (MLP average indirect effect 6.6% on GPT-2 XL). |
| Script | `examples/gallery/causal_tracing.py`; run `python examples/gallery/causal_tracing.py --out DIR` |
| Pinned inputs | `openai-community/gpt2` at revision `607a30d7`, float32, eager attention; the first 10 kept records of the ROME `known_1000.json` facts vendored in `examples/replications/data/known_facts.json`; noise 3 x the embedding standard deviation, 10 draws per fact, seed = fact index |
| Verdict | passed; CPU (x86_64, 2 threads), 3 Nov 2026, TorchLens 2.36.3, commit `e90c3400` |
| Cost | 16 min 52 s wall on 2 CPU cores, peak memory 5.4 GiB (10 facts, both implementations, both controls, save, reload and replay) |

## Oracles

| Oracle | Rule (fixed before the first run) | Measured | Outcome |
|---|---|---|---|
| Paper's claim | the (layer, token category) with the largest average indirect effect (AIE) of restoring one MLP output over the first 10 facts is the last subject token, in layers 0-7 | peak at the last subject token, layer 0, AIE 0.0063 (0.63 percentage points of answer probability; the paper reports 6.6% on GPT-2 XL) | pass |
| Plain hooks | an independent plain-hooks implementation on the same facts and noise draws gives the same AIE grid, max abs difference at most 1e-6 | max abs difference 0.0 over 72 cells (12 layers x 6 token categories) | pass |

The average total effect of the corruption (clean minus corrupted answer probability) is 0.132.
The single-layer AIE grid, rounded to four places (rows are token categories, columns layers 0
to 11):

| Token category | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| first subject token | 0.0031 | 0.0006 | 0.0002 | 0.0000 | 0.0001 | 0.0000 | 0.0002 | -0.0002 | -0.0001 | -0.0001 | -0.0001 | 0.0000 |
| middle subject tokens | 0.0050 | -0.0003 | -0.0007 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0003 | -0.0001 | 0.0000 | 0.0000 | 0.0000 |
| last subject token | 0.0063 | -0.0002 | -0.0004 | 0.0002 | 0.0001 | 0.0001 | 0.0000 | 0.0001 | 0.0001 | 0.0000 | -0.0001 | 0.0000 |
| first subsequent token | 0.0011 | -0.0002 | 0.0008 | -0.0002 | 0.0001 | -0.0003 | -0.0001 | 0.0000 | 0.0000 | -0.0001 | -0.0001 | 0.0000 |
| further tokens | 0.0006 | 0.0003 | 0.0007 | 0.0002 | 0.0002 | 0.0002 | -0.0001 | 0.0001 | 0.0000 | 0.0000 | -0.0001 | 0.0000 |
| last token | 0.0005 | -0.0010 | 0.0000 | 0.0005 | 0.0003 | 0.0008 | 0.0025 | 0.0026 | 0.0048 | 0.0037 | 0.0019 | -0.0063 |

Single-layer restoration on GPT-2 small recovers far less than the paper's ten-layer windows
on GPT-2 XL, and the subject-token effect concentrates in layer 0, the MLP that reads the
corrupted embedding directly. The paper's middle-layer band does not appear at this size and
window; `--window 10` reproduces the paper's window rule for comparison.

## Controls

| Control | Rule | Measured | Outcome |
|---|---|---|---|
| Wrong layer | restoring layer (L* + 6) mod 12's clean MLP output (layer 6) into the peak site recovers less than 0.1 of the peak AIE (bar 0.00063) | AIE -0.00035 | pass (failed as declared) |
| Wrong fact | restoring the peak layer's clean MLP output from the next fact's clean run (at its own last subject token) into the peak site recovers less than 0.1 of the peak AIE | AIE -0.00014 | pass (failed as declared) |

## Artifact and replay

The clean trace of fact 0 is saved as a portable `causal_tracing_clean_fact0.tlspec` (24.3 MB,
sha256 `f00d774a86f3e16f`...). The script reloads it with `tl.load`, takes the layer-0 MLP value
at the last subject token from the reloaded trace, reruns the restoration on a fresh corrupted
trace, and gets the in-memory effect for that cell exactly (0.0013961, difference 0.0).

## Basis-grid row

| Verbs | TorchLens lines | Plain-hooks lines | Shared setup | Architecture |
|---|---|---|---|---|
| capture, select, read, do, compare, save | 37 | 30 | 76 | GPT-2 small |

The counts are the load-bearing lines (no blanks, comments or docstrings) of the script's
`torchlens` region (capture the clean and corrupted runs, read the MLP values, fork, attach the
restoring edits, rerun) and its `plain` region (the same with forward hooks). Shared setup
(tokenising, corrupting, the restoration writer, scoring) is the same for both. The TorchLens
side is longer here: it keeps the traces for the artifact and the controls, which the
plain-hooks side does not need. No count against the original reference repository was
measured.

## The stamp

```text
{"commit": "e90c3400944265f4d4398767ad02d5b29f56a94b", "date_utc": "2026-11-03T14:47:26Z",
 "device": "cpu", "libraries": {"numpy": "2.2.6", "torchvision": "0.28.0+cpu",
 "transformers": "5.18.0"}, "machine": "x86_64",
 "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31", "python": "3.10.19",
 "torch": "2.13.0+cpu", "torch_threads": 2, "torchlens": "2.36.3", "wall_seconds": 1007.0}
```

## Vary it

The script takes `--facts N` (measure more of the 1,000 facts; the first 10 still decide the
verdict) and `--window W` (restore W consecutive MLP layers at once, the paper's window of 10 on
GPT-2 XL; the verdict always uses one layer). Both add a reported grid beside the verdict.

The examples below run the same restoration on a two-layer stand-in GPT-2 with random weights,
so they finish in seconds: TorchLens forks the corrupted trace, attaches an edit to one MLP's
`down_out` facet that writes the clean value at the last subject token, and reruns; a plain
forward hook does the same and must give the same answer probability.

```python
# recipe-test: tool=transformers expected=True
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=32, n_positions=16, use_cache=False)
config._attn_implementation = "eager"
torch.manual_seed(0)
lm = GPT2LMHeadModel(config).eval()


class EmbedsGPT2(nn.Module):  # the corrupted run is an ordinary input: token embeddings
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, embeds):
        return self.lm(inputs_embeds=embeds).logits[:, -1]


wrapped = EmbedsGPT2(lm).eval()
ids = torch.tensor([[3, 7, 1, 9, 4]])
subject, target = [1, 2], 5  # the subject's positions and the answer token
with torch.no_grad():
    clean_embeds = lm.transformer.wte(ids)
noise = torch.randn(4, len(subject), 16, generator=torch.Generator().manual_seed(0))
corrupted = clean_embeds.repeat(4, 1, 1)
corrupted[:, subject] += 3.0 * noise

clean = tl.trace(wrapped, clean_embeds, save="all")
corrupt = tl.trace(wrapped, corrupted, save="all")
values = {
    layer: clean.modules[f"lm.transformer.h.{layer}.mlp"].facets["down_out"].value.detach()
    for layer in range(2)
}


def restorer(layer):
    def restore(out, *, ctx=None):
        patched = out.clone()
        patched[:, subject[-1]] = values[layer][0, subject[-1]]
        return patched

    return restore


def p_answer_torchlens(layers):
    restored = corrupt.fork("restore")
    for layer in layers:
        site = tl.facet("down_out").in_module(f"lm.transformer.h.{layer}.mlp")
        restored.attach_hooks(site, restorer(layer))
    restored.run(wrapped, corrupted)
    return float(restored.output_ops[-1].out.softmax(-1)[:, target].mean())


def p_answer_plain(layers):
    handles = [
        lm.transformer.h[layer].mlp.register_forward_hook(
            lambda module, args, out, layer=layer: restorer(layer)(out)
        )
        for layer in layers
    ]
    with torch.no_grad():
        p = float(wrapped(corrupted).softmax(-1)[:, target].mean())
    for handle in handles:
        handle.remove()
    return p


one_layer = {layer: (p_answer_torchlens([layer]), p_answer_plain([layer])) for layer in range(2)}
RESULT = all(abs(ours - plain) <= 1e-6 for ours, plain in one_layer.values())
```

A window restores several layers in one rerun; widening it to both layers of the stand-in is one
more edit on the same fork, and the plain hooks agree again:

```python
# recipe-test: tool=transformers expected=True
window = [0, 1]
RESULT = abs(p_answer_torchlens(window) - p_answer_plain(window)) <= 1e-6
```
