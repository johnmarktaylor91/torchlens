# A Jacobian lens in a few lines of TorchLens

The Jacobian lens reads out what a block's residual-stream vector is disposed to
make the model say. Where the logit lens decodes a block's residual state
directly through the model's final norm and unembedding, the Jacobian lens first
transports it into the final layer's basis with a fixed linear map `J_l`, the
corpus-average Jacobian of the final block's residual with respect to block
`l`'s:

```text
lens_l(h) = unembed( final_norm( J_l @ h ) ),    J_l = E[ d h_final / d h_l ]
```

The logit lens is the special case `J_l = I`. TorchLens ships no fitted lenses
and no Jacobian-lens API; this page fits one with public TorchLens primitives:
a trace that keeps the live residual stream, `torch.autograd.grad`, and the
model's own final norm and unembedding. Every spelling here is
DOCUMENTED-UNSTABLE.

The page runs on a small randomly initialized GPT-2 built from a config, with
no download. `examples/jacobian_lens_demo.py` runs the same recipe on the
pretrained GPT-2 small checkpoint and cross-checks it the same way as the last
section here.

```text
pip install torchlens transformers
```

## Minimal example on the running model

This page uses [the decoder on the IOI
prompt](running_examples.md#the-decoder-on-the-ioi-prompt). The logit lens is the
Jacobian lens with `J_l = I`: decode a block's saved residual stream directly through
the model's final norm and unembedding. The recipe below builds its own model and fits
`J_l`.

```python
# recipe-test: tool=transformers expected=True
import torch
import transformers

import torchlens as tl

PROMPT = "When Mary and John went to the store, John gave a drink to"
words = PROMPT.replace(",", " ,").split()
vocab = sorted(set(words))
x = torch.tensor([[vocab.index(word) for word in words]])
torch.manual_seed(0)
config = transformers.GPT2Config(
    vocab_size=len(vocab),
    n_positions=32,
    n_embd=64,
    n_layer=2,
    n_head=4,
    use_cache=False,
    attn_implementation="eager",
)
model = transformers.GPT2LMHeadModel(config).eval()

trace = tl.trace(model, x)

with torch.no_grad():
    residual = trace["transformer.h.0"].out
    logits = model.lm_head(model.transformer.ln_f(residual))
RESULT = logits.shape == trace["lm_head"].out.shape
```

## Build the model and a fitting corpus

```python
# recipe-test: tool=transformers expected=(3, 64, ['transformer.h.0', 'transformer.h.1', 'transformer.h.2'])
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

torch.manual_seed(0)
config = GPT2Config(n_layer=3, n_head=2, n_embd=64, vocab_size=512, n_positions=64, use_cache=False)
model = GPT2LMHeadModel(config).eval()
blocks = [f"transformer.h.{index}" for index in range(config.n_layer)]
corpus = torch.randint(0, 512, (3, 16), generator=torch.Generator().manual_seed(1))
RESULT = (len(corpus), config.n_embd, blocks)
```

## Keep the live residual stream and fit the lens

A trace in the default `save_mode="copy"` clones what it saves, so a saved
activation stays on the autograd graph as a side branch: you can backpropagate
from it to the parameters, but the model's output cannot be differentiated with
respect to it. `save_mode="view"` stores the live tensor itself, the one the
next block consumed, so `trace[address].out` is a valid input for
`torch.autograd.grad` against anything downstream. One trace per prompt then
yields every block's residual stream, and batched cotangents (one per output
dimension) give the Jacobian rows. The estimator follows the reference method:
every valid target position is differentiated at once, the gradient at a source
position sums over the later targets, and the rows are averaged over valid
source positions and prompts. The first positions are skipped (`skip_first`)
and the last position has no later target.

```python
# recipe-test: tool=transformers expected=(2, (64, 64))
d_model, skip_first, chunk = config.n_embd, 4, 32


def residual_states(ids):
    """Run one traced forward and return each block's live output, in block order."""
    trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(save_mode="view"))
    return [trace[address].out for address in blocks]


def jacobian_lens(prompts):
    """Return the corpus-average Jacobian of the last block's residual w.r.t. each earlier block's."""
    total = [torch.zeros(d_model, d_model) for _ in blocks[:-1]]
    for ids in prompts:
        states = residual_states(ids[None])
        valid = torch.arange(skip_first, ids.shape[0] - 1)
        target = states[-1][0, valid].sum(0)
        rows = [[] for _ in blocks[:-1]]
        for cotangents in torch.eye(d_model).split(chunk):
            grads = torch.autograd.grad(
                target,
                states[:-1],
                grad_outputs=cotangents,
                is_grads_batched=True,
                retain_graph=True,
            )
            for row, grad in zip(rows, grads):
                row.append(grad[:, 0, valid].sum(1) / len(valid))
        for index, row in enumerate(rows):
            total[index] += torch.cat(row)
    return [matrix / len(prompts) for matrix in total]


lenses = jacobian_lens(corpus)
RESULT = (len(lenses), tuple(lenses[0].shape))
```

`chunk` trades memory for speed: each chunk of cotangents is one batched
backward pass. GPT-2 small (`d_model = 768`) fits in about ten seconds per
12-token prompt on two CPU threads at `chunk=96`.

## Read the lens out

The readout is the lens formula applied to the same door: index the residual
states of a traced forward, transport each through its `J_l`, and decode with
the model's own final norm and unembedding. With `J_l = I` the same line is the
logit lens.

```python
# recipe-test: tool=transformers expected=(2, (512,))
final_norm, unembed = model.transformer.ln_f, model.lm_head
with torch.no_grad():
    states = residual_states(corpus[:1])
    readout = [unembed(final_norm(h[0, -1] @ J.T)) for h, J in zip(states, lenses)]
RESULT = (len(readout), tuple(readout[-1].shape))
```

`readout[i].topk(5).indices` are block `i`'s top tokens at the last position.
The logit-lens tooling takes the same lenses: `logit_lens(trace, lens=...)`
accepts a mapping from block address to a callable from residual state to
logits, and its entries carry `top_k()` and a `summary()` that names the lens
source (`lens=user`).

```python
# recipe-test: tool=transformers expected=(['transformer.h.0', 'transformer.h.1'], (1, 16, 512))
from torchlens._semantic import logit_lens

lens = {address: (lambda h, J=J: unembed(final_norm(h @ J.T))) for address, J in zip(blocks, lenses)}
trace = tl.trace(model, corpus[:1], save="all")
result = logit_lens(trace, lens=lens, modules=list(lens), verify=False)
RESULT = ([entry.address for entry in result.entries], tuple(result.entries[-1].logits.shape))
```

## Check the fit against `torch.func.jacrev`

The check never touches TorchLens: it replaces block `l`'s output inside a
plain forward of the whole model with a forward hook and differentiates the
last block's valid-position sum with `torch.func.jacrev`, one position at a
time, then reduces the same way as the fit. The two agree to float precision.

```python
# recipe-test: tool=transformers expected=True
def block_output(out):
    """Unwrap a GPT-2 block's output, a tensor or a tuple whose first item is the tensor."""
    return out[0] if isinstance(out, tuple) else out


def model_jacobian(index, ids):
    """Differentiate the whole model with block ``index``'s output replaced by the jacrev input."""
    layers, seen = model.transformer.h, {}
    handles = [
        layer.register_forward_hook(lambda m, i, out, k=k: seen.__setitem__(k, block_output(out)))
        for k, layer in enumerate(layers)
    ]
    model(ids)
    for handle in handles:
        handle.remove()
    valid = torch.arange(skip_first, ids.shape[1] - 1)

    def suffix(h):
        patch = layers[index].register_forward_hook(
            lambda m, i, out: (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        )
        grab = layers[-1].register_forward_hook(
            lambda m, i, out: seen.__setitem__("last", block_output(out))
        )
        model(ids)
        patch.remove()
        grab.remove()
        return seen["last"][0, valid].sum(0)

    jac = torch.func.jacrev(suffix, chunk_size=16)(seen[index].detach())
    return jac[:, 0, valid].sum(1) / len(valid)


def rel_error(index, ids):
    """Relative Frobenius error of the fitted lens against the whole-model Jacobian."""
    reference = model_jacobian(index, ids)
    return float((jacobian_lens(ids)[index] - reference).norm() / reference.norm())


RESULT = max(rel_error(index, corpus[:1]) for index in range(config.n_layer - 1)) < 1e-4
```

## Limits

- The lens is a corpus statistic: a Jacobian averaged over three random prompts
  is a demonstration, not a fitted lens. The reference method averages over
  thousands of positions.
- The fit differentiates through the whole suffix of the model once per
  cotangent chunk, so cost grows with `d_model / chunk` backward passes per
  prompt; `chunk` is bounded by memory.
- `residual_states` keeps every block's live output, so the autograd graph of
  one full forward stays alive until the prompt's rows are done. Sequences
  longer than a few hundred tokens on large models need the forward split or
  `chunk` lowered.
- `save_mode="view"` stores live aliases: a model that writes its residual
  stream in place after a block would change the saved state under you.
  GPT-2 does not; for a model that might, `save_mode="reference"` keeps the
  same live tensor and raises `MutatedReferenceError` on a read after an
  in-place write (it warns once per process that it does so).
- A trace captured under `torch.no_grad()` has no graph to differentiate; the
  fit runs with gradients enabled and only the readout is wrapped in
  `torch.no_grad()`.
- `logit_lens(lens=...)` applies the lens to the saved `resid_post` facet of a
  separate, ordinary trace; the lens callable must accept a `(batch, seq,
  d_model)` tensor and return `(batch, seq, vocab)` logits.
