# Coming from Hugging Face `output_hidden_states` and `output_attentions`

This page is for people who read a transformers model's internals with the keyword arguments
`output_hidden_states=True` and `output_attentions=True` (and its attention registry) and want
something below whole-block grain. If one keyword argument gives you what you need, keep using
it. The examples use a two-block GPT-2 built from a config (no download) and were run against
the transformers range TorchLens's test environment installs (`transformers>=4.45,<6`; the
CI rows install 5.14.1). Checked 2026-11-03.

## The same task in both

Read the first block's hidden state and the first layer's attention probabilities, loading with
eager attention so transformers can return the probabilities.

### In transformers

```python
# migration-test: tool=torchlens expected=((1, 4, 32), (1, 2, 4, 4), 3)
import torch
import transformers

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
config._attn_implementation = "eager"
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    native = model(ids, output_hidden_states=True, output_attentions=True)
RESULT = (
    tuple(native.hidden_states[1].shape),
    tuple(native.attentions[0].shape),
    len(native.hidden_states),
)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=(True, True, True)
import torch
import transformers
import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
config._attn_implementation = "eager"
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])

with torch.no_grad():
    native = model(ids, output_hidden_states=True, output_attentions=True)
trace = tl.trace(model, ids)

block0 = tl.extract(model, ids, ["transformer.h.0"])["transformer.h.0"]
pattern0 = trace.modules["transformer.h.0.attn"].facets["pattern"]
last_block = tl.extract(model, ids, ["transformer.h.1"])["transformer.h.1"]
RESULT = (
    torch.equal(block0, native.hidden_states[1]),
    torch.allclose(pattern0.value, native.attentions[0]),
    not torch.equal(last_block, native.hidden_states[-1]),  # hidden_states[-1] is after ln_f
)
```

## Translation table

| transformers | TorchLens | Notes |
| --- | --- | --- |
| `model(ids, output_hidden_states=True).hidden_states[i]` for `i >= 1` | `tl.extract(model, ids, ["transformer.h.<i-1>"])` (the block's module address) | Equal for every block except the last: transformers applies the final norm (`ln_f` on GPT-2) to `hidden_states[-1]`, while the last block's own output is before it. |
| `hidden_states[0]` (the embeddings) | The embedding module's output, or the op after the position embeddings are added | Find it with `tl.trace(model, ids).layers`. |
| `model(ids, output_attentions=True).attentions[i]` with `attn_implementation="eager"` | `trace.modules["transformer.h.<i>.attn"].facets["pattern"]` | Under eager attention the softmax is a captured op. |
| `output_attentions=True` under the default `sdpa` (or flash) backend | Read the reconstructed facets: `tl.trace(model, ids, reconstruction_ready=True)` | The fused kernel never materializes the probabilities, so transformers has nothing to return there (see below). |
| Registering your own function in `AttentionInterface` | `tl.bridge.transformers.register_attention(model)`, then `with tl.bridge.transformers.record_attention(model) as attn:` | Registers the attention function and its matching mask function together. |
| Queries, keys, values, per-head outputs | `module.facets["q"]`, `["k"]`, `["v"]`, `["z"]` | No keyword argument returns these. |

Two traps these keyword arguments set:

- **Silent empty attentions under fused backends.** With the `sdpa` or flash attention backend
  the probabilities are never computed as a tensor, so the call-level `output_attentions=True`
  gives you no maps, without an error. Load with `attn_implementation="eager"`, or read the
  TorchLens facets.
- **The mask trap.** A custom function registered in `AttentionInterface` alone gets no
  attention mask, because transformers builds masks only for keys its mask registry knows; the
  model then silently computes non-causal attention and its predictions change.
  `tl.bridge.transformers.register_attention` registers the matching mask function in the same
  call, so causal and padding masks always arrive. It never registers over a built-in key.

The TorchLens attention backend, on a model with a padded batch:

```python
# migration-test: tool=torchlens expected=((2, 2, 4, 4), True)
import torch
import transformers
import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
model = transformers.GPT2LMHeadModel(config).eval()
batch = {
    "input_ids": torch.tensor([[1, 5, 9, 2], [7, 3, 0, 0]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
}
tl.bridge.transformers.register_attention(model)
with torch.no_grad(), tl.bridge.transformers.record_attention(model, check_sdpa=True) as attn:
    model(**batch)
probs = attn["transformer.h.0.attn"]  # (batch, heads, query, key)
RESULT = (tuple(probs.shape), torch.count_nonzero(probs[1, :, :2, 2:]).item() == 0)
```

## What TorchLens adds

- Grain below the block: queries, keys, values and per-head outputs through
  `module.facets["q"]`, `["k"]`, `["v"]`, `["z"]`, and any executed op through
  `tl.trace(model, ids).layers`.
- Attention probabilities under the default fused `sdpa` backend, as reconstructed facets
  (`tl.trace(model, ids, reconstruction_ready=True)`, see `docs/facets.md`), where
  `output_attentions=True` returns nothing.
- Attention registered together with its mask function, so a custom function never loses its
  causal or padding mask (`tl.bridge.transformers.register_attention`).
- Selecting by module address after the call, with no keyword argument to pass and no change
  to how the model is loaded.

## What the transformers keyword arguments do that TorchLens does not

- `output_hidden_states=True` returns every block in one call with no lookups to write, and
  its last entry includes the final norm. If that tuple is all you need, it costs less than a
  trace.
- The keyword arguments run inside the model's own forward and add no capture step, so
  nothing is recorded beyond the tuples they return.
- Model-specific outputs (`past_key_values` and the like) come back in the model's own output
  object, which is where transformers code expects them.

See [where to go instead](where_to_go_instead.md) for tools outside TorchLens's scope.

## Using both

### transformers feeds TorchLens

A transformers model is an `nn.Module`: pass it to `tl.trace` or `tl.extract`.
`tl.bridge.transformers` adds the pieces that need Hugging Face knowledge:
`trace_text(model, text)` traces a language model from raw text, and `register_attention` and
`record_attention` hook the attention interface.

### TorchLens feeds transformers

`tl.bridge.transformers.register_attention(model)` registers a TorchLens attention function
in transformers' own `AttentionInterface`, so a normal `model(**batch)` call runs through it.
Outside that, results are plain tensors you can hand to any transformers code.

## Semantic differences

- **The last hidden state.** `hidden_states[-1]` is after the final norm; the last block's own
  output (`tl.extract(model, ids, ["transformer.h.1"])`) is before it, so the two differ, and
  the first example checks that.
- **Silent versus loud.** `output_attentions=True` under a fused backend returns no maps and
  no error; TorchLens attention reads either work from a captured op under eager attention or
  from a reconstructed facet that says so.
- **Registration.** `AttentionInterface` alone registers no mask function; TorchLens's
  `register_attention` registers both.
