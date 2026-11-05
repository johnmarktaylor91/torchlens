# Cross-engine checks: serving-scale capture and how TorchLens verifies it

TorchLens captures one local eager forward pass and records every operation it runs. That
is the right tool for understanding a model and for checking other tools against it on a
small model. It is not a serving engine: it does not batch requests, page a KV cache across
requests, shard a model over GPUs or run quantized serving kernels.

## Where to go for serving-scale capture

For activations from a model under serving load, use the serving engine's own hooks or
extension points:

- vLLM: <https://docs.vllm.ai>
- SGLang: <https://docs.sglang.ai>
- TensorRT-LLM: <https://nvidia.github.io/TensorRT-LLM/>

Capture libraries that run inside these engines, on remote GPUs or across devices are
listed in [where to go instead](where_to_go_instead.md#capture-at-serving-scale-on-remote-gpus-or-across-devices).
TorchLens has no integration with these engines and makes no claim about which sites
their hooks expose or what tensors those hooks return. Serving engines run fused kernels,
so a hook named like a TransformerLens or nnterp site may read a different tensor than
the eager model computes there. The check below finds that out before you rely on it.

## How TorchLens checks an engine on a small model

Run the same small model through the engine and through TorchLens, read the same named
sites from both, and compare them. TorchLens is the eager reference: it records every
executed op, so a site that disagrees means the engine's hook reads a different tensor
(a fused kernel, a different hook point, a tuple element other than the one you expected),
or the two runs differ in mode, dtype, device or padding.

Name the sites the way your interpretability library does.
`torchlens.mechinterp.resolve_alias(trace, name)` accepts TransformerLens names (2.x,
3.x bridge-cache and 4.x canonical names) and nnterp's `StandardizedTransformer` accessor
names, and resolves either to the captured site:

| nnterp name | TransformerLens name | TorchLens facet |
| --- | --- | --- |
| `token_embeddings`, `embed_tokens.output` | `hook_embed`, `embed.hook_out` | `lookup` on the token embedding |
| `layers_input[i]` | `blocks.{i}.hook_resid_pre` | `resid_pre` |
| `attentions_input[i]` | `blocks.{i}.attn.hook_in` | `input` on the attention module |
| `attention_probabilities[i]` | `blocks.{i}.attn.hook_pattern` | `pattern` |
| `attentions_output[i]` | `blocks.{i}.hook_attn_out` | `attn_out` |
| `mlps_input[i]` | `blocks.{i}.mlp.hook_in` | `input` on the MLP module |
| `mlps_output[i]` | `blocks.{i}.hook_mlp_out` | `output` on the MLP module |
| `layers_output[i]` | `blocks.{i}.hook_resid_post` | `resid_post` |
| `ln_final.input` | `ln_final.hook_in` | `final_norm_input` |
| `lm_head.output`, `logits` | `unembed.hook_out` | `logits` |

nnterp standardizes no per-block norm names, and `ln_final.output` has no row (no facet
names the final norm's output apart from the per-block norms; read `ln_final.input` and
apply the norm, or address the module directly). An attention pattern exists only where
the softmax runs as its own op: under a fused attention kernel `attention_probabilities[i]`
resolves `structurally_absent` with a remedy, as it does in nnterp without
`enable_attention_probs=True`.

`alias_report(trace, generation="nnterp")` lists every site of the trace under its nnterp
name, with an honest status for each (`generation="2.x"` and `"4.x"` give the
TransformerLens names). The model below runs the default fused attention, so its two
attention patterns are the only sites that are not live:

```python
# migration-test: tool=torchlens expected=(19, ['attention_probabilities[0]', 'attention_probabilities[1]'], 'transformer.h.1', 'resid_post')
import torch
import transformers
import torchlens as tl
import torchlens.mechinterp as mi

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])

trace = tl.trace(model, ids, save="all")
report = mi.alias_report(trace, generation="nnterp")
not_live = sorted(r.query for r in report if r.status not in ("real", "reconstructed_read_only"))
site = mi.resolve_alias(trace, "layers_output[1]")
RESULT = (len(report), not_live, site.module_address, site.facet)
```

## interp-engine point addresses

interp-engine (<https://github.com/decoderesearch/interp-engine>) names a site by a
canonical point address: a point name, then the layer, then any extra coordinate
(`resid_post.5`, `embeddings`, `resid_streams.5.stream-2`). Its `capture(token_ids, points)`
takes those strings as they are. `resolve_alias` and `alias_report(trace,
generation="interp-engine")` accept the addresses below. The table is pinned to
interp-engine 1.14.3 (commit `2afe000da14f1fff8b9c611665f7fe343f523dd0`); its address fields
are `name`, `layer` and `stream`.

| interp-engine point | TransformerLens name | TorchLens facet |
| --- | --- | --- |
| `embeddings` | `hook_embed` | `lookup` on the token embedding |
| `resid_pre.L` | `blocks.L.hook_resid_pre` | `resid_pre` |
| `attn_in.L` | `blocks.L.attn.hook_in` | `input` on the attention module |
| `value.L` | `blocks.L.attn.hook_v` | `v` |
| `attn_scores.L` | `blocks.L.attn.hook_attn_scores` | `scores` |
| `attn_probs.L` | `blocks.L.attn.hook_pattern` | `pattern` |
| `z.L` | `blocks.L.attn.hook_z` | `z` |
| `attn_out.L` | `blocks.L.attn.hook_out` | `attn_out` |
| `resid_mid.L` | `blocks.L.hook_resid_mid` | `resid_mid` |
| `mlp_in.L` | `blocks.L.mlp.hook_in` | `input` on the MLP module |
| `mlp_pre.L` | `blocks.L.mlp.hook_pre` | `up_out` (plain MLP only) |
| `mlp_pre_linear.L` | `blocks.L.mlp.hook_pre_linear` | `up_out` (gated MLP only) |
| `mlp_act.L` | `blocks.L.mlp.hook_post` | `intermediate` |
| `mlp_out.L` | `blocks.L.mlp.hook_out` | `output` on the MLP module |
| `resid_post.L` | `blocks.L.hook_resid_post` | `resid_post` |
| `lm_head` | (the returned logits) | `logits` |

`alias_report(trace, generation="interp-engine")` lists the trace's sites under these
addresses, then one `structurally_absent` entry for each of the 18 other interp-engine points,
with the reason none of them has a TorchLens site: the QK-norm points (`q_norm_in`,
`q_norm_out`, `k_norm_in`, `k_norm_out`), `attn_gate`, the post-sublayer norm outputs
(`attn_out_post`, `mlp_out_post`), the MoE routing points (`router_logits`,
`expert_weights`, `expert_indices`), `final_norm`, and the seven hyper-connection points.
Hand interp-engine the addresses whose status says TorchLens has the site:

```python
# migration-test: tool=torchlens expected=(48, 18, ['mlp_pre_linear.0', 'mlp_pre_linear.1'], 'transformer.h.1')
import torch
import transformers
import torchlens as tl
import torchlens.mechinterp as mi

torch.manual_seed(0)
config = transformers.GPT2Config(
    n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16, attn_implementation="eager"
)
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])

trace = tl.trace(model, ids, save="all")
report = mi.alias_report(trace, generation="interp-engine")
has_site = ("real", "reconstructed_read_only", "needs_capture")
capture_points = [r.query for r in report if r.status in has_site]
no_site = [r.query for r in report if r.facet is None]
not_live = sorted(r.query for r in report if r.facet is not None and r.status not in has_site)
site = mi.resolve_alias(trace, "resid_post.1")
# With interp-engine: acts = await engine.capture(ids[0].tolist(), capture_points)
RESULT = (len(report), len(no_site), not_live, site.module_address)
```

Limits of the mapping:

- interp-engine returns one prompt without a batch dimension: `[n_tokens, width]` for most
  points (`value.L` with its heads flattened into the width) and `[heads, query, key]` for
  `attn_scores.L` and `attn_probs.L`. TorchLens keeps the model's own layout, so drop the
  batch dimension, flatten `value.L`'s heads (`v[0].reshape(n_tokens, -1)`), and for `z.L`
  move the head dimension behind the token dimension
  (`z[0].permute(1, 0, 2).reshape(n_tokens, -1)`) before comparing.
- `layer` is interp-engine's flattened forward order. It equals the block index on
  single-stack decoders (GPT-2, Llama-style models), which are the models this table was
  checked on; on other layouts check the engine's numbering before pairing.
- `embeddings` is the token embedding module's output. GPT-2's learned position embedding
  (`transformer.wpe`) has no interp-engine point.
- `attn_out.L` and `mlp_out.L` are the raw sublayer outputs. On families with a norm after
  each sublayer (Gemma-2, OLMo-2) interp-engine's `attn_out_post` and `mlp_out_post` differ
  from them, and TorchLens has no facet for those.
- The query and key projections, the attention `result` per head, the gated MLP's
  `gated_out`, `down_out`, the norms' `normalized` and `inv_rms`, the embedding `indices`
  and `final_norm_input` have no interp-engine point.

On GPT-2 small (158 addresses) and SmolLM2-135M (392 addresses), in float32 on CPU with
interp-engine's eager backend, every address above that TorchLens has a site for was
captured by interp-engine and matched the TorchLens tensor exactly. Attention scores match
including the masked entries.

## Worked example: checking nnsight's hooks against TorchLens

nnsight (`pip install nnsight`) reads module inputs and outputs inside a `trace` context.
The example reads five sites per layer plus the logits through nnsight, captures the same
model with TorchLens, resolves the nnterp names, and compares every pair. nnsight reads
modules in execution order, so each layer is read input first and output last.

```python
# migration-test: tool=nnsight expected=(11, [])
import torch
import transformers
from nnsight import NNsight

import torchlens as tl
import torchlens.mechinterp as mi

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
hf_model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])

# The engine side: hook values read through nnsight, keyed by nnterp name.
engine = NNsight(hf_model)
engine_values = {}
with engine.trace(ids):
    for i, block in enumerate(engine.transformer.h):
        engine_values[f"layers_input[{i}]"] = block.input.save()
        engine_values[f"attentions_output[{i}]"] = block.attn.output[0].save()
        engine_values[f"mlps_input[{i}]"] = block.mlp.input.save()
        engine_values[f"mlps_output[{i}]"] = block.mlp.output.save()
        engine_values[f"layers_output[{i}]"] = block.output.save()
    engine_values["lm_head.output"] = engine.lm_head.output.save()

# The reference side: one TorchLens capture, sites found by the same names.
trace = tl.trace(hf_model, ids, save="all")
mismatched = []
for name, engine_value in engine_values.items():
    hit = mi.resolve_alias(trace, name)
    reference = trace.modules[hit.module_address].facets[hit.facet]
    if not torch.allclose(torch.as_tensor(engine_value), reference, atol=1e-6, rtol=1e-5):
        mismatched.append(name)

RESULT = (len(engine_values), mismatched)
```

An empty `mismatched` list means every engine hook read the tensor its name promises. For
a serving engine, do the same with its hooks on a small model it can load, in eval mode
and with the dtype and device it serves in, and set the tolerance to the precision the
engine runs at; a mismatch names the site to investigate before you trust that hook at
scale.
