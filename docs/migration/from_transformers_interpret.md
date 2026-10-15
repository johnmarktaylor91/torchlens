# Coming from transformers-interpret

This page maps transformers-interpret's explainers to TorchLens: their spelling on the
left, the TorchLens spelling on the right.

Scope: TorchLens covers text classification and language-model attribution. It does not
ship transformers-interpret's question-answering, token-classification (NER) or
zero-shot task wrappers; for those tasks, write the scalar you want to explain as a
callable `target=` (a function from the model's logits to one number).

| transformers-interpret | TorchLens | Notes |
| --- | --- | --- |
| `SequenceClassificationExplainer(model, tokenizer)(text)` | `tl.attribution.text(model, tokenizer, text)` | Integrated gradients over the input embeddings, one score per token. |
| `class_name="POSITIVE"` / `index=1` | `target=1` (the class index) | Without `target`, TorchLens explains the predicted class and says so. |
| `explainer.word_attributions` (a list of `(token, score)`) | `result.scores` (one score per token), with tokens in `result.payload().display_tokens` | |
| `explainer.visualize()` | `result.show()`, or `result.payload().to_html()` | |
| `n_steps=50` | `n_steps=50` | TorchLens also reports whether the integral converged (`result.converged`). |
| `internal_batch_size=` | `steps_per_batch=` | |
| Baseline of pad tokens | `baseline="auto"` (the default), or your own baseline | |
| `MultiLabelClassificationExplainer`, `ImageClassificationExplainer` | `tl.attribution` methods on any model | |
| `QuestionAnsweringExplainer`, `TokenClassificationExplainer`, `ZeroShotClassificationExplainer` | Not shipped; use a callable `target=` | |

The example builds a tiny BERT classifier and a ten-word tokenizer locally, so it needs
no download; with a real checkpoint, load both with `from_pretrained` instead.

Their construct:

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

TorchLens equivalent:

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
result = tl.attribution.text(model, tokenizer, "the film was good", target=1, n_steps=32)
tokens = result.payload().display_tokens
RESULT = (len(result.scores) == len(tokens), torch.isfinite(result.scores).all().item())
```
