# Coming from BertViz or Ecco

This page maps BertViz's attention views and Ecco's saliency, logit-lens and
neuron-factor views to TorchLens: their spelling on the left, the TorchLens spelling on
the right. The pictures live in `torchlens.tviz` (`import torchlens.tviz as tviz`); their
spellings are documented-unstable. Saving a picture needs matplotlib
(`pip install "torchlens[viz]"`).

## BertViz

| BertViz | TorchLens | Notes |
| --- | --- | --- |
| `model(..., output_attentions=True).attentions`, then `head_view(attention, tokens)` | `views = tviz.attention_views(tl.trace(model, ids), tokens=tokens)`, then `tviz.render_attention(views[0], "layer0.pdf")` | One `AttentionView` per attention layer; the render is a per-head grid. |
| `model_view(attention, tokens)` (every layer and head) | `tviz.render_attention_atlas(views, "atlas.pdf")` | |
| Notebook HTML views | `tviz.circuitsvis_attention(views)` | The CircuitsVis bridge, in a notebook. |
| Keep using BertViz's own HTML views | `head_view(tviz.bertviz_tuple(views), tokens)` | `bertviz_tuple` returns the `output_attentions` tuple layout. It is a recipe, not a supported bridge. |
| `neuron_view` (query and key vectors behind each score) | `tviz.ScoreDecomposition` records | GPT-2 and BERT families; rotary-embedding families refuse. |
| Requires the eager attention backend for `output_attentions` | Capture with the eager attention backend too; under fused SDPA, read the reconstructed attention facets (see `docs/facets.md`) | |

Attention weights show where each query position reads from. They are routing
measurements, not causal importance, and every TorchLens attention picture says so in
its footer.

## Ecco

| Ecco | TorchLens | Notes |
| --- | --- | --- |
| `lm = ecco.from_pretrained(name)` | Load the model with transformers, then `tl.trace(model, ids)` | Any PyTorch model, not a fixed list. |
| `output = lm.generate(text, generate=n)` | `model.generate(...)`, then trace the prompt plus the tokens you want to explain | Episode capture records a whole generation run (`docs/reference/episode_capture.md`). |
| `output.saliency()` (input saliency for a generated token) | `tl.attribution.text(model, tokenizer, text, target=" Paris").show()` | Integrated gradients for one target token. |
| `output.layer_predictions(position=...)` (logit lens) | `torchlens.semantic.logit_lens.logit_lens_predictions(trace)`, then `tviz.prediction_trajectory(...)` and `tviz.render_prediction_ribbon(...)` | |
| `output.run_nmf(...)` (neuron factors) | The NMF recipe in `docs/reference/tviz.md`: factorize a named activation matrix with scikit-learn and draw it with TorchLens's token strips | NMF stays your code by design. |

Neither BertViz nor Ecco is executed here (Ecco targets transformers 4). The TorchLens
side of both tables, on a two-block GPT-2 built from a config:

```python
# migration-test: tool=torchlens expected=(2, (2, 4, 4), True, True)
import os
import tempfile

import torch
import transformers
import torchlens as tl
import torchlens.tviz as tviz
from torchlens.semantic.logit_lens import logit_lens_predictions

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
attention_pdf = tviz.render_attention(views[0], os.path.join(outdir, "layer0.pdf"))

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
