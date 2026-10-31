# A tuned lens in TorchLens's logit lens

A tuned lens (Belrose et al., the `tuned-lens` package) decodes each block's
hidden state into logits through a small learned translator per layer, where
the plain logit lens uses the model's final norm and unembedding alone.
`torchlens._semantic.logit_lens` reads the per-block hidden states from a trace
and accepts any decoder through `lens=`: one callable for every layer, or a
mapping from block address to a per-layer callable. This page builds that
mapping from a `tuned_lens.TunedLens`. TorchLens does not train or ship tuned
lenses.

```text
pip install torchlens transformers
pip install --no-deps tuned-lens==0.2.0
```

`tuned-lens` 0.2.0 declares training and data dependencies (`datasets`,
`torchdata`, `wandb`, `accelerate`, and others) that this route never
imports, and some of them do not install against current torch. The lens
modules import only `torch`, `transformers`, `huggingface_hub`, and `numpy`,
which `transformers` already brings, so `--no-deps` is enough.

## Real use: published lens weights

Published lenses download from the Hugging Face Hub. This is not run here:

```text
from transformers import AutoModelForCausalLM
from tuned_lens import TunedLens

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
lens = TunedLens.from_model_and_pretrained(model)   # downloads the gpt2 lens
```

The rest of the page runs the same mechanics on a small randomly initialized
GPT-2 built from a config, with no download.

## Trace the model and build the lens

`TunedLens.from_model` builds a lens for a model without loading weights: it
copies the model's final norm and unembedding and adds one translator per
block, initialized to zero, so a fresh lens decodes exactly like the plain
logit lens. The trace needs the per-block residual facets, which the
exhaustive save provides.

```python
# recipe-test: tool=tuned_lens expected=(2, 2)
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
from torchlens._semantic import logit_lens
from tuned_lens import TunedLens

torch.manual_seed(0)
config = GPT2Config(
    n_layer=2, n_head=2, n_embd=64, vocab_size=512, n_positions=64, use_cache=False
)
model = GPT2LMHeadModel(config).eval()
input_ids = torch.randint(0, 512, (1, 6), generator=torch.Generator().manual_seed(0))
trace = tl.trace(model, input_ids, capture=tl.options.CaptureOptions(layers_to_save="all"))

lens = TunedLens.from_model(model).eval()
blocks = [f"transformer.h.{i}" for i in range(config.n_layer)]
RESULT = (len(lens), len(blocks))
```

## Plug it into `logit_lens`

`TunedLens.forward(h, idx)` takes the hidden state and the layer index, so the
mapping binds each block address to its index. `verify=False` skips the
model-identity check that `logit_lens` runs on user lenses; the check below
does the same comparison by hand: with zero translators, the last block's row
must equal the model's own logits.

```python
# recipe-test: tool=tuned_lens expected=(['transformer.h.0', 'transformer.h.1'], (1, 6, 512), True)
lens_by_block = {
    address: (lambda h, i=i: lens(h, idx=i)) for i, address in enumerate(blocks)
}
result = logit_lens(trace, lens=lens_by_block, verify=False)

with torch.no_grad():
    model_logits = model(input_ids).logits
RESULT = (
    [entry.address for entry in result.entries],
    tuple(result.entries[-1].logits.shape),
    bool(torch.allclose(result.entries[-1].logits, model_logits, atol=1e-4)),
)
```

## A trained lens changes the rows

Training moves the translators away from zero. Setting them to small seeded
values stands in for loaded weights here: the projected rows now differ from
the plain logit lens, while the number of rows and their shape stay the same.

```python
# recipe-test: tool=tuned_lens expected=(2, True)
plain_rows = [entry.logits.clone() for entry in result.entries]
generator = torch.Generator().manual_seed(1)
with torch.no_grad():
    for translator in lens:
        translator.weight.copy_(0.05 * torch.randn(translator.weight.shape, generator=generator))
        translator.bias.zero_()

tuned = logit_lens(trace, lens=lens_by_block, verify=False)
RESULT = (
    len(tuned.entries),
    all(not torch.allclose(row.logits, plain) for row, plain in zip(tuned.entries, plain_rows)),
)
```

## Limits

- TorchLens applies the lens to captured hidden states; training a lens, and
  judging whether a published lens fits your checkpoint, belong to
  `tuned-lens`.
- `tuned-lens` 0.2.0 recognizes a fixed set of Hugging Face architectures
  (GPT-2, GPT-Neo, GPT-J, GPT-NeoX, OPT, BLOOM, Llama) when it locates the
  final norm and unembedding.
- With `validate=False`, nothing checks that the lens belongs to the traced
  model. A lens may carry a `model_identity` record; see the `lens=` entry of
  `logit_lens` for how `verify=True` uses it.
- `logit_lens` keeps every per-layer projection in memory; for long sequences
  and large vocabularies use `logit_lens_predictions`, which takes the same
  `lens=`.
