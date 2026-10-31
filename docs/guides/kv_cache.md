# Editing generation and the KV cache

During generation a decoder reads the keys and values of earlier positions from its KV cache
instead of recomputing them. That gives three different things you might mean by "intervene
during generation", and they cost very different amounts:

| You want to | Use | Cost |
| --- | --- | --- |
| Hold an edit on every decoding step, new tokens included | `torchlens.intervention.steer_generate(model, ids, spec, ...)`, or `spec.bind(model).generate(...)` | close to a plain forward hook per token; see [the fast-path guide](fast_paths.md#generation-with-hf-generate) |
| Edit what the cache holds for the prompt, then let generation read it unedited | the bound spec on the prefill call only (recipe below) | one prefill forward |
| Record every step of a generation as one inspectable product | `tl.trace(..., episode=tl.options.EpisodeOptions(...))` | diagnostic tier: superlinear in the number of steps, so tens of steps, not hundreds ([episode capture](../reference/episode_capture.md)) |

`steer_generate` never traces: it runs the model's real `generate()` with the edit installed as a
hook, so the edit fires once per forward call (`result.report.fire_count`), and the keys and values
it writes into the cache already carry the edit. Use a trace or an episode capture only when you
need the recorded graph, and then for a few steps.

## Recipe: edit the cached past

To change what later tokens see of the prompt without touching the new tokens, apply the edit to
the key or value projection during the prefill call only. The edited values are written into the
cache, and every later step reads them from there. The check below compares the result with the
same edit written as a plain forward hook.

```python
# docs-test: expected=[True, True]
import torch
import torchlens as tl
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
lm = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
).eval()
prompt = torch.tensor([[3, 17, 42, 8, 99]])
next_token = torch.tensor([[7]])
v_proj = lm.model.layers[0].self_attn.v_proj


def blank_position_1(values):
    values = values.clone()
    values[:, 1] = 0.0  # the cached value of prompt position 1 in layer 0
    return values


def decode_after(prefill):
    """Run one decoding step that reads the cache the prefill call wrote."""
    with torch.no_grad():
        cache = prefill(prompt, use_cache=True).past_key_values
        return lm(next_token, past_key_values=cache, use_cache=True).logits


# TorchLens: the edit is bound for the prefill call only.
spec = tl.when(
    tl.module("model.layers.0.self_attn.v_proj"),
    lambda values, *, hook: blank_position_1(values),
)
edited = decode_after(spec.bind(lm))

# The same surgery with a plain hook on the prefill call.
handle = v_proj.register_forward_hook(lambda module, args, out: blank_position_1(out))
try:
    with torch.no_grad():
        cache = lm(prompt, use_cache=True).past_key_values
finally:
    handle.remove()
with torch.no_grad():
    reference = lm(next_token, past_key_values=cache, use_cache=True).logits

clean = decode_after(lm)
RESULT = [torch.allclose(edited, reference, atol=1e-6), not torch.allclose(edited, clean)]
```

Keep a retention plan when you also trace these calls: a decoder's whole-graph default saves
every op's activation, and the attention matrix alone grows with the square of the sequence
length. Pass a `save=` selector naming the sites you will read (as above, `tl.module(...)`), and
see [long sequences](../performance.md#long-sequences-and-long-contexts) in the performance
guide.

## What the cache is to TorchLens

When you trace a decoding step, the cache passed as `past_key_values` is an input container like
any other: its tensors are graph inputs, addressed by their path inside the container
([containers](../containers.md)). An edit at a module inside the step changes what that step
computes and what it appends to the cache; it does not rewrite earlier entries. To change earlier
entries, edit them at the call that wrote them, as the recipe does.
