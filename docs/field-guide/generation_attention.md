# Attention at each generation step

Record a generation with the attention softmax saved, then draw any layer's
attention pattern as it stood at any step: heads by query position by key
position, with the row the step computed outlined. Step 0 is the prefill (one
row per prompt token); step `k` is the k-th decode forward, which computes one
new row, the newest position, against every key so far. The picture is the
softmax the model produced at each step, placed at its position; nothing is
recomputed. The renderer is the same one `tl.viz.attention` uses for a single
forward.

The step door is `tl.transformer.attention_view(generation, layer, step=k)`:
the same call that reads one forward's trace, given a generation recording and
the step to picture. The rendering call is `tl.viz.attention`.

The example uses the two-layer GPT-2 with random weights from the running
examples, so it runs offline. With a real checkpoint, load the model with
`from_pretrained` and pass `attn_implementation="eager"`: fused attention
kernels never materialize the pattern.

## Minimal example on the running model

This page uses [the decoder on the IOI
prompt](running_examples.md#the-decoder-on-the-ioi-prompt). One forward's
attention pattern is one `AttentionView`; the generation sections below build
one per step.

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

view = tl.transformer.attention_view(trace, "transformer.h.0.attn", tokens=words)
RESULT = tuple(view.pattern.shape) == (4, len(words), len(words))
```

## 1. Record the generation

`tl.record` with `generate_kwargs` runs the model's own `generate` and keeps one
record per forward. Saving `tl.func("softmax")` keeps every attention pattern
and nothing else. The cache must stay on (`use_cache=True`); the running
example's config turns it off for single forwards, so the call turns it back on.

```python
# recipe-test: tool=transformers expected=(4, 4)
generation = tl.record(
    model,
    x,
    save=tl.func("softmax"),
    generate_kwargs={
        "max_new_tokens": 4,
        "min_new_tokens": 4,
        "do_sample": False,
        "use_cache": True,
    },
)
RESULT = (len(generation.steps), generation.sequences.shape[1] - len(words))
```

## 2. One layer at one step

Four new tokens take four forwards: the prefill and three decode steps. The view
for step 3 covers the prompt plus the three positions the decode steps
computed. Each earlier row is the one its own step computed, so the cells to
the right of a row are keys that did not exist yet; they hold zero and carry no
mask, because zeros are never read as a mask. Label the axes with one string
per position of `generation.sequences`.

```python
# recipe-test: tool=transformers expected=((4, 17, 17), 'decode')
labels = [vocab[int(token)] for token in generation.sequences[0]]
view = tl.transformer.attention_view(generation, "transformer.h.1.attn", step=3, tokens=labels)
RESULT = (tuple(view.pattern.shape), view.episode.role)
```

## 3. Draw it

`tl.viz.attention` draws the view; for a decode step it outlines the last row,
the one that step computed, and says so in the footer.

```python
# recipe-test: tool=transformers expected=True
saved = tl.viz.attention(view, "step3.png", heads=(0, 1))
RESULT = saved.paths[0].exists() and any("outlined row" in line for line in saved.disclosure_lines)
```

## 4. Every layer at one step

Call `attention_view` once per attention layer to picture the same step across
the stack.

```python
# recipe-test: tool=transformers expected=('transformer.h.0.attn', 'transformer.h.1.attn')
views = [
    tl.transformer.attention_view(generation, layer, step=1, tokens=labels)
    for layer in ("transformer.h.0.attn", "transformer.h.1.attn")
]
RESULT = tuple(view.layer for view in views)
```

## What it refuses

A layer with no saved softmax, a step that does not exist, too few token labels
or a step whose earlier forwards were evicted by `GenerationOptions(keep_last=)`
each raise a typed error naming the fix. Branch on `exc.fields["code"]`.

```python
# recipe-test: tool=transformers expected='tv_facet_missing'
try:
    tl.transformer.attention_view(generation, "transformer.h.0.mlp", step=0)
except tl.errors.RenderError as exc:
    RESULT = exc.fields["code"]
```
