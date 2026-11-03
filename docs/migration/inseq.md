# Coming from Inseq

This page is for people who attribute a sequence-to-sequence or language model's generation
to its inputs with Inseq and want to know which parts TorchLens covers. The bridge below was
tested on inseq 0.7.1 (the `torchlens[inseq]` extra pins `inseq~=0.6`). Inseq downloads model
weights, so its calls are described in prose; the TorchLens side runs on a tiny model built in the
fence. Checked 2026-11-03.

## The same task in both

Attribute the model's next-token prediction after "the cat sat on" to the prompt tokens.

### In Inseq

Inseq loads a model with an attribution method and attributes each generated step. You call
`inseq.load_model("gpt2", "saliency")` to get an attribution model, then
`attr_model.attribute("the cat sat on", generation_args={"max_new_tokens": 1})` to attribute
one generated token, and `.show()` on the result to display it. The model downloads from the
Hugging Face Hub, so this is described here rather than run; check the calls against your inseq
version.

### In TorchLens

`tl.attribution.text` attributes one target token to the prompt tokens with integrated
gradients over the input embeddings. The example builds a tiny GPT-2 and a six-word tokenizer
locally; without `target=` it explains the model's own argmax prediction and says so.

```python
# migration-test: tool=torchlens expected=(True, True)
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
import torchlens as tl

torch.manual_seed(0)
vocab = {"<pad>": 0, "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}
raw = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
raw.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>")
config = transformers.GPT2Config(
    n_layer=2, n_embd=32, n_head=2, vocab_size=len(vocab), n_positions=16
)
model = transformers.GPT2LMHeadModel(config).eval()
result = tl.attribution.text(model, tokenizer, "the cat sat on", num_steps=16)
tokens = result.payload().display_tokens
RESULT = (len(result.scores) == len(tokens), torch.isfinite(result.scores).all().item())
```

## Translation table

| Inseq operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attribute generation outputs | `tl.attribution.text(model, tokenizer, text, target=...)` attributes one target token to the prompt tokens (integrated gradients over input embeddings); a whole generation run can be captured with episode capture (`episode=tl.options.EpisodeOptions(...)`, documented-unstable) | Partial; per-step attribution across every generated token, with step scores, stays in Inseq. |
| Step-wise hidden-state access | `log.find_sites(...)` over repeated loop sites | Equivalent when generation loop is visible. |
| Contrastive inputs | Capture clean/corrupted logs and compare with `Bundle` | Equivalent comparison pattern. |
| Attribution method selection | `tl.attribution` ships saliency, input x gradient, integrated gradients, SmoothGrad, gradient SHAP, occlusion, layer attribution methods and more; `tl.attribution.text` uses integrated gradients | Partial; Inseq's generation-aware method catalog (including DeepLift, LIME and attention-weight methods) stays in Inseq. |
| Intervention during generation | `tl.when(site, action).bind(model).generate(ids, ...)`, or `.bind(model)` alone for many forwards | Equivalent for local PyTorch generation code. |
| Token-position patching | Tensor-shaped replacements or custom hooks | Equivalent if target tensors expose position dimensions. |
| Save analysis setup | `.tlspec/` saves intervention recipe; metric code remains external | Partial. |
| Visual attribution reports | The `tl.attribution.text` result renders per-token scores (`.show()`, `.to_html()`, `.to_text()`) | Per-token report for one target; no per-step generation heatmap. |
| Fused attention internals | Eager-attention load: the softmax and score ops are captured ops you read directly. Default fused SDPA load: `tl.trace(model, x, reconstruction_ready=True)` serves read-only reconstructed attention facets (`scores`, `pattern`, `z`, checked against the fused output; see `docs/facets.md`) | Head-level semantics are readable on fused models; the raw kernel intermediates are not captured ops, and editing them needs an eager recapture. |
| Run inseq itself from a TorchLens session | `tl.bridge.inseq.attribute(model_or_id, text, method=..., generated_texts=...)` (extra `torchlens[inseq]`, tested on inseq 0.7.1) loads inseq's attribution model and returns inseq's own result (its `.show()` renderer works). `method` takes inseq's names or the TorchLens spelling (`input_x_grad`). Forced decoding is checked: a result that attributes any text other than `generated_texts` refuses `bridge_inseq_forced_decoding_ignored` instead of returning free-generation attribution. | Equivalent for one call; the per-step product stays in inseq. |
| HuggingFace convenience wrappers | `torchlens.bridge.transformers.trace_text(model, text)` traces a Hugging Face language model from raw text; `tl.attribution.text` takes the model and its tokenizer | Shipped for capture and single-target token attribution. |

## What TorchLens adds

- Intervention during generation on the model's own `generate()`: `tl.when(site, action)
  .bind(model).generate(ids, ...)` applies the edit at every decoding step, KV cache included.
- Clean and corrupted runs compared in one container, `tl.Bundle`, with an explicit metric.
- A convergence report on the integral (`result.converged`) and a printed concrete baseline.
- General attribution methods on any model under `tl.attribution`, and the activations behind
  a score from the same model with `tl.trace`.
- A tested bridge that runs Inseq itself from a TorchLens session (`tl.bridge.inseq.attribute`).

## What Inseq does that TorchLens does not

Inseq is the better fit when the deliverable is per-step attribution across a generated sequence,
with its step scores, method catalog and generation heatmaps. TorchLens attributes a single target
token to the prompt (`tl.attribution.text`), ships general attribution methods under
`tl.attribution`, captures visible PyTorch generation loops, compares clean and corrupted runs, and
exposes gradients and activations for custom analysis.

- Inseq's generation-aware method catalog (including DeepLift, LIME and attention-weight
  methods) stays in Inseq; for text DeepLift and LIME, use Captum and Inseq. See
  [where to go instead](where_to_go_instead.md).

## Using both

### Inseq feeds TorchLens

An Inseq attribution model wraps a transformers model, which is an `nn.Module`; trace the
underlying model with `tl.trace`. Inseq's result objects are not TorchLens traces.

### TorchLens feeds Inseq

`tl.bridge.inseq.attribute(model_or_id, text, method=..., generated_texts=...)` (extra
`torchlens[inseq]`, tested on inseq 0.7.1) loads Inseq's attribution model and returns Inseq's
own result, whose `.show()` renderer works. `method` takes Inseq's names or the TorchLens
spelling (`input_x_grad`). Forced decoding is checked: a result that attributes any text other
than `generated_texts` refuses `bridge_inseq_forced_decoding_ignored` instead of returning
free-generation attribution.

## Semantic differences

- **One target, not every step.** `tl.attribution.text` attributes one target token to the
  prompt; Inseq attributes every generated step and keeps the step scores.
- **Default target.** Without `target=`, TorchLens explains the model's own argmax prediction
  and says so.
- **Method spellings.** TorchLens uses `input_x_grad`; the bridge accepts both spellings.
- **Fused attention.** Head-level semantics are readable on fused models through
  reconstructed facets (`tl.trace(model, x, reconstruction_ready=True)`, `docs/facets.md`); the
  raw kernel intermediates are not captured ops.
