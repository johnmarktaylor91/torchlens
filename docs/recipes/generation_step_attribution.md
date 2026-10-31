# Per-step attribution for generated text

Generate a continuation, then attribute each generated token to the tokens
before it with the token-attribution two-liner,
`tl.attribution.text(model, tokenizer, text, target=...)`, run on the full
generated text. For the token at position `t`, the target is the logit that
predicted it: `target=(t - 1, token_t)`. Step scores with the names inseq uses
(`probability`, `entropy`, `contrast_prob_diff`) come straight from the same
logits. TorchLens has no packaged generation-attribution call; for a
maintained one, use inseq.

The example uses a two-layer GPT-2 with random weights and a word-level
tokenizer so it runs offline. With a real checkpoint, load the model and
tokenizer with `from_pretrained` (a download) and keep the rest.

## 1. Generate

```python
# recipe-test: tool=transformers expected=True
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers

import torchlens as tl

words = ["[PAD]", "the", "eiffel", "tower", "is", "in", "paris", "rome", "city", "a", "big"]
backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[PAD]"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]")

torch.manual_seed(0)
config = transformers.GPT2Config(
    vocab_size=len(words), n_positions=32, n_embd=32, n_layer=2, n_head=2
)
model = transformers.GPT2LMHeadModel(config).eval()

prompt = "the eiffel tower is in"
ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
n_prompt = ids.shape[1]
with torch.no_grad():
    for _ in range(3):  # greedy, never the pad token
        logits = model(ids).logits[0, -1]
        logits[tokenizer.pad_token_id] = float("-inf")
        ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
generated_text = tokenizer.decode(ids[0])
RESULT = tokenizer(generated_text)["input_ids"] == ids[0].tolist()  # the text round-trips
```

## 2. Attribute every generated step

Each step attributes the logit at position `t - 1` for the token actually
generated at `t`. Under a causal model, tokens at `t` and later cannot affect
that logit, so their scores are zero.

```python
# recipe-test: tool=transformers expected=True
steps = []
for t in range(n_prompt, ids.shape[1]):
    token_t = int(ids[0, t])
    result = tl.attribution.text(
        model, tokenizer, generated_text, target=(t - 1, token_t), num_steps=32
    )
    steps.append((t, result))

RESULT = all(float(r.scores[t:].abs().max()) < 1e-6 for t, r in steps)
```

## 3. Step scores from the logits

```python
# recipe-test: tool=transformers expected=True
foil = tokenizer.convert_tokens_to_ids("rome")
with torch.no_grad():
    probs = model(ids).logits[0].softmax(-1)

step_scores = []
for t, _result in steps:
    p = probs[t - 1]
    step_scores.append(
        {
            "token": words[int(ids[0, t])],
            "probability": float(p[ids[0, t]]),
            "entropy": float(-(p * p.clamp_min(1e-12).log()).sum()),
            "contrast_prob_diff": float(p[ids[0, t]] - p[foil]),
        }
    )
RESULT = all(0.0 < s["probability"] <= 1.0 and s["entropy"] >= 0.0 for s in step_scores)
```

## 4. A contrastive step

A callable target scores any scalar of the logits; here, the generated token
against a foil at one step.

```python
# recipe-test: tool=transformers expected=True
t = n_prompt  # the first generated token
token_t = int(ids[0, t])
contrast = tl.attribution.text(
    model,
    tokenizer,
    generated_text,
    target=lambda out: out[0, t - 1, token_t] - out[0, t - 1, foil],
    num_steps=32,
)
RESULT = contrast.scores.shape == (ids.shape[1],)
```

## Save a step as HTML

`result.payload().to_html()` returns a self-contained HTML table of the
token scores with the disclosure lines:

```python
# recipe-test: tool=transformers expected=True
with open("step.html", "w") as handle:
    handle.write(steps[0][1].payload().to_html())
RESULT = open("step.html").read().count("<td") == ids.shape[1]  # one cell per token
```

## Limits

- Each step is one Integrated Gradients run over the whole text, so cost grows
  with the number of generated tokens; attribute the steps you need.
- The target is the logit, not the probability. The step scores above read
  the probabilities separately; they are not attributions.
