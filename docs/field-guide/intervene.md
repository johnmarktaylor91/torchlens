# Intervene on a site and measure the effect

**Result in one line:** the model's output with one site edited, next to the clean output,
and a live version of the same edit that runs at the cost of a forward hook.

<!-- JMT review: plain factual prose, written for review. -->

**Requirements:** `pip install torchlens transformers`.

## Minimal example on the running model

This page uses [the decoder on the IOI prompt](running_examples.md#the-decoder-on-the-ioi-prompt).

```python
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
```

Zero the second block's MLP output and rerun the model under the edit. `fork()` keeps the
clean trace as it was.

```python
site = tl.module("transformer.h.1.mlp")
head = tl.module("lm_head")
ablated = trace.fork().do(site, tl.zero_ablate(), model=model, inputs=x)

clean_logits = trace.find_sites(head).first().out[0, -1]
ablated_logits = ablated.find_sites(head).first().out[0, -1]
mary, john = vocab.index("Mary"), vocab.index("John")
effect = (ablated_logits[mary] - ablated_logits[john]) - (clean_logits[mary] - clean_logits[john])
print(f"change in the Mary-minus-John logit difference: {effect.item():+.4f}")
```

## Output

The printed number is the change in the logit difference at the last position. With random
weights it measures nothing about names; with the trained GPT-2 small it is the quantity the
IOI analyses report.

## A check

The same edit written as a plain forward hook gives the same logits, and the edited site really
is zero in the ablated trace.

```python
handle = model.transformer.h[1].mlp.register_forward_hook(
    lambda module, args, out: torch.zeros_like(out)
)
hooked_logits = model(x).logits[0, -1]
handle.remove()
assert torch.allclose(ablated_logits, hooked_logits, atol=1e-6)
assert torch.count_nonzero(ablated.find_sites(site).first().out) == 0
```

## The same edit, live

`tl.when(site, edit).bind(model)` applies the edit on every call with no capture, so a sweep or
`generate()` runs at about the cost of a plain forward hook.

```python
bound = tl.when(site, tl.zero_ablate()).bind(model)
live_logits = bound(x).logits[0, -1]
assert torch.allclose(live_logits, hooked_logits, atol=1e-6)
```

## Vary it

- Another site: `tl.module("transformer.h.0.mlp")`, or every MLP at once with
  `tl.in_module("transformer.h.1") & tl.func("gelu")`-style selectors.
- Another edit: `tl.scale(0.5)`, `tl.steer(direction, strength=4.0, dim=-1)`,
  `tl.patch_from(other_trace)` for activation patching from a second prompt.
- A control: edit a site the answer should not depend on and check the effect stays near zero.

```python
control = trace.fork().do(tl.module("transformer.h.0.mlp"), tl.scale(1.0), model=model, inputs=x)
control_logits = control.find_sites(head).first().out[0, -1]
assert torch.allclose(control_logits, clean_logits, atol=1e-6)
```

## Performance note

`do(..., model=, inputs=)` reruns the forward with capture, so it costs one traced forward per
edit. For many forwards under one fixed edit, use the bound form above. The cost of each path
is measured on the [capture benchmark page](../benchmarks/capture.md).

## Limits

- An edit shows what the output does when a site changes; it is not by itself an explanation of
  the behaviour.
- Sites inside a fused kernel cannot be edited separately.

## Next task

[See the graph and inspect any operation](see_and_inspect.md), or [the intervention reference](../intervention_api.md).
