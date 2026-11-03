# LRP on transformers: use LXT

LRP methods may be patent-encumbered for commercial use. TorchLens keeps LRP
a recipe, not a product feature (see [the epsilon-LRP litmus
recipe](lrp_epsilon_litmus.md)). LXT, the reference implementation of AttnLRP
and CP-LRP for transformers, is released under the Clear BSD License, a
copyright license that states it grants no patent rights. Check both before
using either in a product.

For LRP on transformer language models, use LXT. Its efficient mode patches a
Hugging Face model family so that one ordinary backward pass yields relevance
as input times gradient. TorchLens traces the patched model like any other
model; it does not compute the relevance. This page runs both on one small
GPT-2.

```text
pip install torchlens transformers
pip install --no-deps "lxt @ git+https://github.com/rachtibat/LRP-eXplains-Transformers@f04344177551bfcc3665e88afe96501f43fe560b"
```

The LXT 2.1 release on PyPI does not import under transformers 5 (it imports
a helper transformers removed, and it replaces the attention registry with a
plain dict); the upstream repository fixed both after the release, so this
page pins a commit. LXT declares `accelerate`, `bitsandbytes`,
`open_clip_torch`, `zennit`, `tabulate`, and `matplotlib`, but the efficient
path below imports only `torch` and `transformers`, so `--no-deps` is enough.
Without `zennit`, LXT warns once at import that its vision-transformer patches
are unavailable.

## Build the model and keep LXT's patches reversible

LXT's `monkey_patch` changes classes, not one model: it replaces the
`forward` of `torch.nn.LayerNorm`, `torch.nn.Dropout`, and GPT-2's MLP, and
wraps the attention functions in transformers' shared registry, for the whole
process. The context manager below records the originals and puts them back
on exit, so other models in the same process are unaffected. `cp_LRP` is LXT's
CP-LRP rule set for GPT-2 (`attnLRP` is the other).

```python
# recipe-test: tool=lxt expected=(1, 6, 512)
from contextlib import contextmanager

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2 import modeling_gpt2

import torchlens as tl
from lxt.efficient import monkey_patch
from lxt.efficient.models.gpt2 import cp_LRP

original_layer_norm_forward = nn.LayerNorm.forward


@contextmanager
def lxt_patched(patch_map):
    """Apply LXT's GPT-2 patches, then restore every patched callable."""
    methods = {cls: cls.forward for cls in (nn.LayerNorm, nn.Dropout, modeling_gpt2.GPT2MLP)}
    eager = modeling_gpt2.eager_attention_forward
    registry = dict(modeling_gpt2.ALL_ATTENTION_FUNCTIONS.items())
    monkey_patch(modeling_gpt2, patch_map=patch_map)
    try:
        yield
    finally:
        for cls, forward in methods.items():
            cls.forward = forward
        modeling_gpt2.eager_attention_forward = eager
        for key, function in registry.items():
            modeling_gpt2.ALL_ATTENTION_FUNCTIONS[key] = function


torch.manual_seed(0)
config = GPT2Config(
    n_layer=2, n_head=2, n_embd=64, vocab_size=512, n_positions=64, use_cache=False
)
model = GPT2LMHeadModel(config).eval()
input_ids = torch.randint(0, 512, (1, 6), generator=torch.Generator().manual_seed(0))
with torch.no_grad():
    reference_logits = model(input_ids).logits
RESULT = tuple(reference_logits.shape)
```

## TorchLens traces the patched model

LXT's patches change gradients, not forward values, so the traced forward of
the patched model reproduces the unpatched logits. The trace records LXT's
operations (its gradient rules are custom autograd functions) like any other
ops.

```python
# recipe-test: tool=lxt expected=(True, True)
with lxt_patched(cp_LRP):
    trace = tl.trace(model, input_ids)
traced_logits = trace[trace.output_layers[0]].out
RESULT = (
    tuple(traced_logits.shape) == tuple(reference_logits.shape),
    bool(torch.allclose(traced_logits, reference_logits, atol=1e-4)),
)
```

## LXT relevance: input embeddings times gradient

Under the patch, relevance for a chosen logit is the input embeddings times
their gradient after one backward pass from that logit. Summing over the
embedding axis gives one relevance per token.

```python
# recipe-test: tool=lxt expected=((1, 6, 64), (1, 6), True)
with lxt_patched(cp_LRP):
    embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds).logits
    target = int(logits[0, -1].argmax())
    logits[0, -1, target].backward()
    relevance = embeds * embeds.grad

token_relevance = relevance.sum(-1)
RESULT = (
    tuple(relevance.shape),
    tuple(token_relevance.shape),
    bool(torch.isfinite(relevance).all()),
)
```

After the `with` block, `nn.LayerNorm.forward` (recorded in the first step) and
the other patched callables are the originals again:

```python
# recipe-test: tool=lxt expected=True
RESULT = nn.LayerNorm.forward is original_layer_norm_forward
```

## Limits

- The relevance rules, their correctness, and the list of supported
  architectures are LXT's. TorchLens records the patched forward; it does not
  check LXT's conservation or rules.
- LXT patches classes process-wide. Restore them (as above) or run LXT in its
  own process; otherwise every later LayerNorm and Dropout in that process
  keeps LXT's behavior, including Dropout returning its input unchanged in
  training mode.
- The tiny model is randomly initialized, so the relevance values mean
  nothing; the example checks shapes and finiteness only.
- Real use loads a pretrained checkpoint (`from_pretrained`, a download) and
  usually turns on gradient checkpointing for memory, as LXT's documentation
  describes.
