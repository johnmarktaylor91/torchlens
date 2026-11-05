# Coming from transformers-interpret

This page is for people who explain a transformers classifier or language model with
transformers-interpret's explainers and want token attributions from TorchLens. TorchLens's
test environment does not pin transformers-interpret; its example skips where the package is
not installed. Checked 2026-11-03.

Scope: TorchLens covers text classification and language-model attribution. It does not ship
transformers-interpret's question-answering, token-classification (NER) or zero-shot task
wrappers; for those tasks, write the scalar you want to explain as a callable `target=` (a
function from the model's logits to one number).

## The same task in both

Attribute the class-1 score of a tiny BERT classifier to the tokens of "the film was good".
The example builds the classifier and a ten-word tokenizer locally, so it needs no download;
with a real checkpoint, load both with `from_pretrained` instead.

### In transformers-interpret

```python
# migration-test: tool=transformers_interpret expected=True
import os
import tempfile

import torch
import transformers
from transformers_interpret import SequenceClassificationExplainer

torch.manual_seed(0)
words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "film", "was", "good", "bad"]
vocab_path = os.path.join(tempfile.mkdtemp(), "vocab.txt")
with open(vocab_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(words))
tokenizer = transformers.BertTokenizer(vocab_path)
config = transformers.BertConfig(
    vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
    num_attention_heads=2, intermediate_size=64, num_labels=2,
)
model = transformers.BertForSequenceClassification(config).eval()
explainer = SequenceClassificationExplainer(model, tokenizer)
attributions = explainer("the film was good", index=1)
RESULT = all(isinstance(score, float) for _, score in attributions)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=(True, True)
import os
import tempfile

import torch
import transformers
import torchlens as tl

torch.manual_seed(0)
words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "film", "was", "good", "bad"]
vocab_path = os.path.join(tempfile.mkdtemp(), "vocab.txt")
with open(vocab_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(words))
tokenizer = transformers.BertTokenizer(vocab_path)
config = transformers.BertConfig(
    vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
    num_attention_heads=2, intermediate_size=64, num_labels=2,
)
model = transformers.BertForSequenceClassification(config).eval()
result = tl.attribution.text(model, tokenizer, "the film was good", target=1, num_steps=32)
tokens = result.payload().display_tokens
RESULT = (len(result.scores) == len(tokens), torch.isfinite(result.scores).all().item())
```

## Translation table

| transformers-interpret | TorchLens | Notes |
| --- | --- | --- |
| `SequenceClassificationExplainer(model, tokenizer)(text)` | `tl.attribution.text(model, tokenizer, text)` | Integrated gradients over the input embeddings, one score per token. |
| `class_name="POSITIVE"` / `index=1` | `target=1` (the class index) | Without `target`, TorchLens explains the predicted class and says so. |
| `explainer.word_attributions` (a list of `(token, score)`) | `result.scores` (one score per token), with tokens in `result.payload().display_tokens` | |
| `explainer.visualize()` | `result.show()`, or `result.payload().to_html()` | |
| `n_steps=50` | `num_steps=50` | TorchLens also reports whether the integral converged (`result.converged`). |
| `internal_batch_size=` | `steps_per_batch=` | |
| Baseline of pad tokens | `baseline="auto"` (the default), or your own baseline | |
| `MultiLabelClassificationExplainer`, `ImageClassificationExplainer` | `tl.attribution` methods on any model | |
| `QuestionAnsweringExplainer`, `TokenClassificationExplainer`, `ZeroShotClassificationExplainer` | Not shipped; use a callable `target=` | |

## What TorchLens adds

- A convergence report for the integral (`result.converged`) and a baseline that is always
  resolved to a printed concrete choice (`baseline="auto"`).
- Attribution beyond integrated gradients on the same model: `tl.attribution` ships saliency,
  input x gradient, SmoothGrad, gradient SHAP, occlusion and layer attribution methods
  (see `docs/reference/attribution.md`).
- The same model can be traced afterwards with `tl.trace(model, ids)` to look at the
  activations behind a score, without changing tools.
- A contrastive target (`target` and a foil) and a callable `target=` for any scalar of the
  logits.

## What transformers-interpret does that TorchLens does not

- Its question-answering, token-classification (NER) and zero-shot wrappers have no
  TorchLens counterpart; you write the scalar yourself as a callable `target=`.
- It ships a one-call explainer class per task, with the class name and label logic in the
  wrapper. TorchLens takes the model, tokenizer, text and a target.
- For attribution methods TorchLens does not ship: TCAV and influence functions live in
  Captum, and DeepLift and LIME for text live in Captum and Inseq (see also
  [where to go instead](where_to_go_instead.md)).

## Using both

### transformers-interpret feeds TorchLens

There is no bridge. Both take the same transformers model and tokenizer, so load them once and
pass them to either tool.

### TorchLens feeds transformers-interpret

There is no bridge. A TorchLens `result.scores` tensor is a plain tensor; line it up with
`result.payload().display_tokens` to compare against `word_attributions`. The two use
different baselines and step counts, so expect different numbers.

## Semantic differences

- **Predicted class by default.** Without `target`, TorchLens explains the model's predicted
  class and says so; transformers-interpret explains the predicted class unless you pass
  `index` or `class_name`.
- **Step count keyword.** The integration steps keyword is `num_steps` in TorchLens, `n_steps`
  in transformers-interpret.
- **Special tokens.** TorchLens keeps special tokens at their true embeddings under a pad
  baseline (`keep_special_tokens=True`), and its result tokens include them.
- **Convergence.** TorchLens reports whether the integral converged (`result.converged`);
  transformers-interpret returns the scores without that check.
