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
TransformerLens names):

```python
# migration-test: tool=torchlens expected=(19, True, 'transformer.h.1', 'resid_post')
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
live = all(row.status in ("real", "reconstructed_read_only") for row in report)
site = mi.resolve_alias(trace, "layers_output[1]")
RESULT = (len(report), live, site.module_address, site.facet)
```

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
