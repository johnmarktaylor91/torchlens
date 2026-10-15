# Coming from Hugging Face `output_hidden_states` and `output_attentions`

transformers already returns some internals when you ask: `output_hidden_states=True`
gives each block's output and `output_attentions=True` gives each layer's attention
probabilities. This page maps those keyword arguments, and transformers' attention
registry, to TorchLens: their spelling on the left, the TorchLens spelling on the right.
If one keyword argument gives you what you need, keep using it; TorchLens is for
everything below whole-block grain (queries, keys, values, per-head views, the residual
stream inside a block, any op).

| transformers | TorchLens | Notes |
| --- | --- | --- |
| `model(ids, output_hidden_states=True).hidden_states[i]` for `i >= 1` | `tl.extract(model, ids, ["transformer.h.<i-1>"])` (the block's module address) | Equal for every block except the last: transformers applies the final norm (`ln_f` on GPT-2) to `hidden_states[-1]`, while the last block's own output is before it. |
| `hidden_states[0]` (the embeddings) | The embedding module's output, or the op after the position embeddings are added | Find it with `tl.trace(model, ids).layers`. |
| `model(ids, output_attentions=True).attentions[i]` with `attn_implementation="eager"` | `trace.modules["transformer.h.<i>.attn"].facets["pattern"]` | Under eager attention the softmax is a captured op. |
| `output_attentions=True` under the default `sdpa` (or flash) backend | Read the reconstructed facets: `tl.trace(model, ids, reconstruction_ready=True)` | The fused kernel never materializes the probabilities, so transformers has nothing to return there (see below). |
| Registering your own function in `AttentionInterface` | `tl.bridge.hf.register_attention(model)`, then `with tl.bridge.hf.record_attention(model) as attn:` | Registers the attention function and its matching mask function together. |
| Queries, keys, values, per-head outputs | `module.facets["q"]`, `["k"]`, `["v"]`, `["z"]` | No keyword argument returns these. |

Two traps these keyword arguments set:

- **Silent empty attentions under fused backends.** With the `sdpa` or flash attention
  backend the probabilities are never computed as a tensor, so the call-level
  `output_attentions=True` gives you no maps, without an error. Load with
  `attn_implementation="eager"`, or read the TorchLens facets.
- **The mask trap.** A custom function registered in `AttentionInterface` alone gets no
  attention mask, because transformers builds masks only for keys its mask registry
  knows; the model then silently computes non-causal attention and its predictions
  change. `tl.bridge.hf.register_attention` registers the matching mask function in
  the same call, so causal and padding masks always arrive. It never registers over a
  built-in key.

The example builds a two-block GPT-2 from a config (no download), loads it with eager
attention and checks both mappings against transformers' own outputs:

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
tl.bridge.hf.register_attention(model)
with torch.no_grad(), tl.bridge.hf.record_attention(model, check_sdpa=True) as attn:
    model(**batch)
probs = attn["transformer.h.0.attn"]  # (batch, heads, query, key)
RESULT = (tuple(probs.shape), torch.count_nonzero(probs[1, :, :2, 2:]).item() == 0)
```
