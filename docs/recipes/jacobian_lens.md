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
and no Jacobian-lens API; this page fits one with public TorchLens primitives
and plugs it into `logit_lens(lens=...)`. Every spelling here is
DOCUMENTED-UNSTABLE.

The page runs on a small randomly initialized GPT-2 built from a config, with
no download. `examples/jacobian_lens_demo.py` runs the same recipe on the
pretrained GPT-2 small checkpoint and cross-checks it the same way as the last
section here.

```text
pip install torchlens transformers
```

## Build the model and a fitting corpus

```python
# recipe-test: tool=transformers expected=(3, 64, ['transformer.h.0', 'transformer.h.1', 'transformer.h.2'])
import functools
import operator

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
from torchlens._semantic import logit_lens

torch.manual_seed(0)
config = GPT2Config(n_layer=3, n_head=2, n_embd=64, vocab_size=512, n_positions=64, use_cache=False)
model = GPT2LMHeadModel(config).eval()
blocks = [f"transformer.h.{index}" for index in range(config.n_layer)]
corpus = torch.randint(0, 512, (3, 16), generator=torch.Generator().manual_seed(1))
RESULT = (len(corpus), config.n_embd, blocks)
```

## Keep the live residual stream and fit the lens

A trace copies what it saves, so a saved activation cannot be differentiated
after the forward. A hook callable bound with `tl.when(...).bind(model)` runs
inside the forward and sees the live tensor; returning it unchanged keeps the
block's output attached to the autograd graph. One forward per prompt then
yields every block's residual stream, and `torch.autograd.grad` with batched
cotangents (one per output dimension) gives the Jacobian rows. The estimator
follows the reference method: every valid target position is differentiated at
once, the gradient at a source position sums over the later targets, and the
rows are averaged over valid source positions and prompts. The first positions
are skipped (`skip_first`) and the last position has no later target.

```python
# recipe-test: tool=transformers expected=(['transformer.h.0', 'transformer.h.1'], (64, 64))
d_model, skip_first, chunk = config.n_embd, 4, 32


def residual_states(ids):
    """Run one forward and return each block's live output, in block order."""
    states = {}

    def keep(address):
        def hook(activation, *, ctx):
            states[address] = activation
            return activation

        return hook

    spec = functools.reduce(operator.and_, [tl.when(tl.module(a), keep(a)) for a in blocks])
    spec.bind(model)(ids)
    return [states[address] for address in blocks]


def jacobian_lens(prompts):
    """Return the corpus-average Jacobian of the last block's residual w.r.t. each earlier block's."""
    total = {address: torch.zeros(d_model, d_model) for address in blocks[:-1]}
    for ids in prompts:
        states = residual_states(ids[None])
        valid = torch.arange(skip_first, ids.shape[0] - 1)
        target = states[-1][0, valid].sum(0)
        rows = {address: [] for address in blocks[:-1]}
        for cotangents in torch.eye(d_model).split(chunk):
            grads = torch.autograd.grad(
                target,
                states[:-1],
                grad_outputs=cotangents,
                is_grads_batched=True,
                retain_graph=True,
            )
            for address, grad in zip(blocks[:-1], grads):
                rows[address].append(grad[:, 0, valid].sum(1) / len(valid))
        for address in blocks[:-1]:
            total[address] += torch.cat(rows[address])
    return {address: matrix / len(prompts) for address, matrix in total.items()}


lenses = jacobian_lens(corpus)
RESULT = (list(lenses), tuple(lenses[blocks[0]].shape))
```

`chunk` trades memory for speed: each chunk of cotangents is one batched
backward pass. GPT-2 small (`d_model = 768`) fits in about ten seconds per
12-token prompt on two CPU threads at `chunk=96`.

## Read the lens out through `logit_lens`

`logit_lens(lens=...)` takes a mapping from block address to a callable that
turns a residual state into logits; the default identity lens is the logit
lens. The lens callable applies the Jacobian, then the model's own final norm
and unembedding, so the two lenses differ only in the transport step.

```python
# recipe-test: tool=transformers expected=(['transformer.h.0', 'transformer.h.1'], (1, 16, 512))
final_norm, unembed = model.transformer.ln_f, model.lm_head
lens = {address: (lambda h, J=J: unembed(final_norm(h @ J.T))) for address, J in lenses.items()}
trace = tl.trace(model, corpus[:1], capture=tl.options.CaptureOptions(layers_to_save="all"))
result = logit_lens(trace, lens=lens, layers=list(lens), verify=False)
RESULT = ([entry.address for entry in result.entries], tuple(result.entries[-1].logits.shape))
```

`result.entries[i].top_k(5)` lists the lens's top tokens per position and
`result.summary()` names the lens source (`lens=user`). The plain logit lens of
the same trace is `logit_lens(trace)`; on the pretrained checkpoint the two
disagree most in the early blocks, where the residual basis is far from the
final one.

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
    return float((jacobian_lens(ids[None])[blocks[index]] - reference).norm() / reference.norm())


RESULT = max(rel_error(index, corpus[0]) for index in range(config.n_layer - 1)) < 1e-4
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
- The hook callable door runs with gradients enabled only when the caller does;
  do not wrap the fit in `torch.no_grad()`.
- `logit_lens(lens=...)` applies the lens to the saved `resid_post` facet of a
  separate, ordinary trace; the lens callable must accept a `(batch, seq,
  d_model)` tensor and return `(batch, seq, vocab)` logits.
