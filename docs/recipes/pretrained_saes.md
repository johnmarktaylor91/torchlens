# Pretrained SAEs: SAELens loads, TorchLens runs

SAELens keeps the registry of published sparse autoencoders, Gemma Scope 1 and
Gemma Scope 2 among them, and loads any of them with `SAE.from_pretrained`.
TorchLens runs the model, finds the site the SAE was trained on, splices the
SAE's reconstruction in with `torchlens.bridge.sae.splice` and reports what the
splice did: reconstruction fidelity, the SAE latents, the change in next-token
cross-entropy and the KL divergence from the clean run. TorchLens ships no SAE
loader; the division of labour is SAELens for the SAE, TorchLens for the run.

```text
pip install torchlens transformers
pip install "sae-lens==6.54.0"
```

## Real use: Gemma Scope

This block is not run here: the Gemma weights are gated on the Hugging Face
Hub (accept the licence and log in first), and the SAEs download from it.

```text
import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchlens as tl
from torchlens.bridge.sae import splice
from torchlens.mechinterp import resolve_alias

# Gemma Scope 1, on Gemma 2 2B: the residual stream after decoder layer 12.
sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical")
model_id = "google/gemma-2-2b"
# Gemma Scope 2, on Gemma 3 270M: the same kind of site after layer 12.
# sae = SAE.from_pretrained("gemma-scope-2-270m-pt-res", "layer_12_width_16k_l0_medium")
# model_id = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, attn_implementation="eager"
).eval()
ids = tokenizer("The capital of France is Paris.", return_tensors="pt").input_ids

trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
site = resolve_alias(trace, sae.cfg.metadata.hook_name).module_address  # "model.layers.12"
result = splice(trace, site=site, sae=sae, tokens=ids, exclude_bos=True)
result.fraction_variance_explained_pooled, result.ce_loss_recovered, result.kl_divergence
```

`exclude_bos=True` follows the Gemma Scope evaluation: the BOS position keeps
its clean value and is left out of every metric, because its activation norm
is far larger than any other position's.

## Site names: SAELens hooks to TorchLens sites

An SAELens SAE records the site it was trained on in
`sae.cfg.metadata.hook_name`, as a TransformerLens hook name.
`torchlens.mechinterp.resolve_alias(trace, hook_name)` translates that name
against the trace in front of you; its `module_address` is the splice site,
and its `status` must be `"real"` (an op TorchLens captured). On a Hugging Face
Gemma or GPT-2 model the common Gemma Scope hooks land here:

| SAELens `hook_name` | Gemma Scope releases | TorchLens site |
| --- | --- | --- |
| `blocks.{L}.hook_resid_post` | `gemma-scope-2b-pt-res`, `gemma-scope-2-*-pt-res` | output of the decoder layer, `model.layers.{L}` (GPT-2: `transformer.h.{L}`) |
| `blocks.{L}.hook_mlp_out` | `gemma-scope-2b-pt-mlp`, `gemma-scope-2-*-pt-mlp` | the MLP output facet that `resolve_alias` names; on Gemma 2 and 3 the MLP output passes through `post_feedforward_layernorm` before it joins the residual stream, so check which tensor the release was trained on |
| `blocks.{L}.attn.hook_z` | `gemma-scope-2b-pt-att` | the concatenated head outputs before the output projection (`resolve_alias` reports whether that tensor is a captured op in your attention implementation) |
| a transcoder's input and output hooks | `gemma-scope-2-*-pt-transcoders` | the MLP module as a whole, spliced with `tl.splice_module(transcoder, input_route="in")`; see the transcoder section below |

## The same steps on a stand-in

The rest of the page runs offline: a three-layer GPT-2 with random weights
built from a config, and a small SAELens `StandardSAE` with seeded weights, saved in
the SAELens format and loaded back with `SAE.load_from_disk`, as a downloaded
SAE would be.

```python
# recipe-test: tool=sae_lens expected=('StandardSAE', 'blocks.1.hook_resid_post')
import tempfile

import torch
from sae_lens import SAE, StandardSAE, StandardSAEConfig
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

torch.manual_seed(0)
config = GPT2Config(n_layer=3, n_head=2, n_embd=16, vocab_size=64, n_positions=32, use_cache=False)
model = GPT2LMHeadModel(config).eval()
ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(0))

cfg = StandardSAEConfig(d_in=16, d_sae=48)
cfg.metadata.hook_name = "blocks.1.hook_resid_post"
built = StandardSAE(cfg)
with torch.no_grad():
    for parameter in built.parameters():
        parameter.copy_(0.3 * torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1)))
folder = tempfile.mkdtemp()
built.save_model(folder)
sae = SAE.load_from_disk(folder)
RESULT = (type(sae).__name__, sae.cfg.metadata.hook_name)
```

## Map the hook name and splice

`resolve_alias` maps the hook name to the block output. `tokens=ids` turns on
the next-token metrics: the logits are the model output shaped like `ids` plus
a vocabulary axis.

```python
# recipe-test: tool=sae_lens expected=('real', 'transformer.h.1', True, True, False)
from torchlens.bridge.sae import splice
from torchlens.mechinterp import resolve_alias

trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
alias = resolve_alias(trace, sae.cfg.metadata.hook_name)
result = splice(trace, site=alias.module_address, sae=sae, tokens=ids, exclude_bos=True)

with torch.no_grad():
    latents = sae.encode(result.clean_site_out)
RESULT = (
    alias.status,
    alias.module_address,
    bool(torch.equal(result.latents, latents)),
    all(value is not None for value in (result.ce_loss_recovered, result.kl_divergence)),
    result.zero_variance,
)
```

## Check the numbers against SAELens's own computation

The site TorchLens spliced is Hugging Face's `hidden_states[L + 1]` (for any
block but the last, whose entry comes after the final norm), and the metrics are the ones SAELens's evals and the Gemma Scope paper report: the
fraction of variance unexplained with per-dimension centring, and the
cross-entropy loss recovered against a zero-ablated site. Recomputed here
directly, without the splice:

```python
# recipe-test: tool=sae_lens expected=(True, True, True)
with torch.no_grad():
    hidden = model(ids, output_hidden_states=True).hidden_states[2]
    x = hidden[:, 1:].reshape(-1, 16).double()
    recon = sae.decode(sae.encode(hidden[:, 1:])).reshape(-1, 16).double()
fvu = float(((recon - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

def next_token_ce(logits):
    log_probs = torch.log_softmax(logits.double(), -1)[:, 1:-1]
    return float(-log_probs.gather(-1, ids[:, 2:, None]).mean())

ce_clean = next_token_ce(result.clean_outputs[0])
ce_spliced = next_token_ce(result.spliced_outputs[0])
recovered = (result.ce_loss_zero_ablated - ce_spliced) / (result.ce_loss_zero_ablated - ce_clean)
RESULT = (
    bool(torch.allclose(result.clean_site_out, hidden)),
    abs((1 - result.fraction_variance_explained_pooled) - fvu) < 1e-5,
    abs(result.ce_loss_recovered - recovered) < 1e-9,
)
```

## Transcoders

A transcoder maps a block's MLP input to its MLP output, so it replaces the
MLP call as a whole: `tl.splice_module(transcoder, input_route="in")` on the MLP
module runs the transcoder on the arguments the MLP was called with (here the
`ln_2` output) and uses its result as the MLP's output. With `input="out"` it
would read the MLP's own output instead, which is the SAE case. A loaded
SAELens transcoder goes in the same place; the stand-in below is an SAELens
`Transcoder` with seeded weights, a map from the MLP's 16-wide input to its
16-wide output.

```python
# recipe-test: tool=sae_lens expected=(True, True)
from sae_lens import Transcoder, TranscoderConfig

transcoder = Transcoder(TranscoderConfig(d_in=16, d_sae=48, d_out=16, apply_b_dec_to_input=False))
with torch.no_grad():
    for parameter in transcoder.parameters():
        parameter.copy_(0.3 * torch.randn(parameter.shape, generator=torch.Generator().manual_seed(2)))
clean = tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))
spliced = tl.trace(
    model,
    ids,
    capture=tl.options.CaptureOptions(
        intervention_ready=True,
        hooks={tl.module("transformer.h.1.mlp"): tl.splice_module(transcoder, input_route="in")},
    ),
)
with torch.no_grad():
    expected = transcoder(clean["transformer.h.1.ln_2"].out)
RESULT = (
    bool(torch.allclose(spliced["transformer.h.1.mlp"].out, expected, atol=1e-6)),
    not torch.allclose(spliced.output_ops[0].out, clean.output_ops[0].out),
)
```

## Visualise a latent

An SAE latent is also an objective: optimise an input until the latent at its
site responds strongly. [The feature-visualisation recipe](feature_visualisation.md)
reads the site through a capture-free binding, passes it through the SAE's
`encode` and checks the result with `splice`.
