# Coming from BertViz or Ecco

This page is for people who look at attention with BertViz's views, or at saliency, the logit
lens and neuron factors with Ecco, and want the TorchLens equivalents. Neither BertViz nor
Ecco is installed in TorchLens's test environment (Ecco targets transformers 4), so their calls
are shown as text and the inputs they consume are produced by an executed example. The pictures
live in `torchlens.transformer` (`import torchlens.transformer as tviz`); their spellings are
documented-unstable. Saving a picture needs matplotlib (`pip install "torchlens[viz]"`).
Checked 2026-11-03.

## The same task in both

Show the attention of a two-block GPT-2, one panel per head, for the tokens "The cat sat down".

### In BertViz

BertViz takes the `output_attentions` tuple of a transformers model and the token strings:

```text
from bertviz import head_view

head_view(attention, tokens)   # attention = model(ids, output_attentions=True).attentions
```

The executed part is the input BertViz consumes, built with transformers alone:

```python
# migration-test: tool=torchlens expected=(2, (1, 2, 4, 4))
import torch
import transformers

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
config._attn_implementation = "eager"
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    attention = model(ids, output_attentions=True).attentions
RESULT = (len(attention), tuple(attention[0].shape))
```

### In TorchLens

The TorchLens side of both tables, on the same model: BertViz's head view, then Ecco's layer
predictions as a logit-lens ribbon.

```python
# migration-test: tool=torchlens expected=(2, (2, 4, 4), True, True)
import os
import tempfile

import torch
import transformers
import torchlens as tl
import torchlens.transformer as tviz
from torchlens._semantic.logit_lens import logit_lens_predictions

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
config._attn_implementation = "eager"
model = transformers.GPT2LMHeadModel(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])
tokens = ["The", " cat", " sat", " down"]
trace = tl.trace(model, ids)

# BertViz's head view: one view per attention layer, then a per-head grid.
views = tviz.attention_views(trace, tokens=tokens)
outdir = tempfile.mkdtemp()
attention_pdf = tl.viz.attention(views[0], os.path.join(outdir, "layer0.pdf"))

# Ecco's layer predictions: the logit lens, drawn as a ribbon.
predictions = logit_lens_predictions(trace, k=3)
trajectory = tviz.prediction_trajectory(predictions)
ribbon = tviz.render_prediction_ribbon(trajectory, os.path.join(outdir, "ribbon.pdf"))

RESULT = (
    len(views),
    tuple(views[0].pattern.shape[-3:]),
    all(os.path.exists(path) for path in attention_pdf.paths),
    all(os.path.exists(path) for path in ribbon.paths),
)
```

## Translation table

### BertViz

| BertViz | TorchLens | Notes |
| --- | --- | --- |
| `model(..., output_attentions=True).attentions`, then `head_view(attention, tokens)` | `views = tviz.attention_views(tl.trace(model, ids), tokens=tokens)`, then `tl.viz.attention(views[0], "layer0.pdf")` | One `AttentionView` per attention layer; the render is a per-head grid. |
| `model_view(attention, tokens)` (every layer and head) | `tl.viz.attention_atlas(views, "atlas.pdf")` | |
| Notebook HTML views | `tviz.circuitsvis_attention(views)` | The CircuitsVis bridge, in a notebook. |
| Keep using BertViz's own HTML views | `head_view(tviz.bertviz_tuple(views), tokens)` | `bertviz_tuple` returns the `output_attentions` tuple layout. It is a recipe, not a supported bridge. |
| `neuron_view` (query and key vectors behind each score) | `tviz.ScoreDecomposition` records | GPT-2 and BERT families; rotary-embedding families refuse. |
| Requires the eager attention backend for `output_attentions` | Capture with the eager attention backend too; under fused SDPA, read the reconstructed attention facets (see `docs/facets.md`) | |

Attention weights show where each query position reads from. They are routing measurements,
not causal importance, and every TorchLens attention picture says so in its footer.

### Ecco

| Ecco | TorchLens | Notes |
| --- | --- | --- |
| `lm = ecco.from_pretrained(name)` | Load the model with transformers, then `tl.trace(model, ids)` | Any PyTorch model, not a fixed list. |
| `output = lm.generate(text, generate=n)` | `model.generate(...)`, then trace the prompt plus the tokens you want to explain | Episode capture records a whole generation run (`docs/reference/episode_capture.md`). |
| `output.saliency()` (input saliency for a generated token) | `tl.attribution.text(model, tokenizer, text, target=" Paris").show()` | Integrated gradients for one target token. |
| `output.layer_predictions(position=...)` (logit lens) | `torchlens._semantic.logit_lens.logit_lens_predictions(trace)`, then `tviz.prediction_trajectory(...)` and `tviz.render_prediction_ribbon(...)` | |
| `output.run_nmf(...)` (neuron factors) | The NMF recipe in `docs/reference/tviz.md`: factorize a named activation matrix with scikit-learn and draw it with TorchLens's token strips | NMF stays your code by design. |

## What TorchLens adds

- One capture feeds every picture: `tl.trace(model, ids)` once, then
  `tviz.attention_views(trace, ...)`, the logit-lens ribbon and any op's saved value.
- Attention under the default fused backend: reconstructed facets (`docs/facets.md`) where
  `output_attentions=True` returns nothing.
- Any PyTorch model, not a fixed list of pretrained names.
- Pictures that state what they measure: the attention footer says routing, not causal
  importance.
- `tl.attribution.text` reports whether its integral converged (`result.converged`), which
  saliency pictures from other tools do not.

## What BertViz and Ecco do that TorchLens does not

- BertViz's interactive HTML views (`head_view`, `model_view`, `neuron_view`) run in a
  notebook with hover and selection. TorchLens writes static pictures (PDF, SVG); its
  notebook route is the CircuitsVis bridge, `tviz.circuitsvis_attention(views)`.
- Ecco ships its own model wrapper that records activations as you generate; TorchLens traces
  the prompt plus the tokens you want to explain, or uses episode capture.
- Ecco's neuron-factor view bundles the NMF fit. In TorchLens the fit stays your code.
- For other attribution and visualization tools see [where to go instead](where_to_go_instead.md).

## Using both

### BertViz or Ecco feeds TorchLens

There is no bridge. Both take a transformers model, which TorchLens traces the same way. An
Ecco `lm` wraps a model; trace the underlying transformers model instead.

### TorchLens feeds BertViz or Ecco

For BertViz, `tviz.bertviz_tuple(views)` returns the `output_attentions` tuple layout, so
`head_view(tviz.bertviz_tuple(views), tokens)` works from a TorchLens capture. It is a recipe,
not a supported bridge. Ecco has no such route.

## Semantic differences

- **Eager attention.** `output_attentions` needs the eager backend; TorchLens reads a captured
  softmax under eager attention and a reconstructed facet under fused SDPA.
- **Routing, not importance.** Attention weights are routing measurements; the footer of
  every TorchLens attention picture says so.
- **Saliency method.** `tl.attribution.text` is integrated gradients for one target token,
  with a convergence report; Ecco's saliency offers several gradient variants.
- **Documented-unstable spellings.** The `torchlens.transformer` picture functions may be
  renamed.
