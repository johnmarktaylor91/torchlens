# LIT bridge (`tl.bridge.lit`) — EXPERIMENTAL

TorchLens bridges INTO [LIT](https://pair-code.github.io/lit/) (Google PAIR's
Language Interpretability Tool, PyPI `lit-nlp`): a served browser dashboard for
behavioral analysis — counterfactual editing, embedding projection, salience,
metrics, side-by-side model comparison. The bridge wraps a **live** torch model
as a genuine `lit_nlp.api.model.Model`, so LIT's panels drive it over real
HTTP. The differentiation: LIT users hand-write per-model wrappers exposing one
or two fields; TorchLens generates the wrapper for any torch model exposing
**any traced site**, with no `output_hidden_states` plumbing and no model-code
changes.

Every spelling on this page is DOCUMENTED-UNSTABLE pending the naming sprint.

```python
import torchlens as tl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
net = AutoModelForSequenceClassification.from_pretrained(name).eval()

wrapper = tl.bridge.lit.model(
    net, tok,
    task="classification",     # or "causal_lm"
    sites="blocks",            # gate-passed preset; or explicit ("distilbert.transformer.layer.5", ...)
    pooling="mean_masked",     # or "first_token" / "last_unmasked" / callable
    salience=True,             # opt-in model-provided TokenSalience (signed, autorun=False)
)
ds = tl.bridge.lit.dataset(["a great movie", "a terrible movie"],
                           labels=("NEGATIVE", "POSITIVE"))

# The bridge ships no serve()/launch() verb: call LIT's server directly.
from lit_nlp import dev_server
dev_server.Server({"tl": wrapper}, {"sst": ds},
                  layouts={"torchlens": tl.bridge.lit.layout()}).serve()
```

## What flows in

- **Predictions and counterfactual re-prediction**: every `predict()` call
  tokenizes LIT's (possibly edited) text and runs ONE padded traced forward.
- **Activations as `Embeddings`** (projector/PCA): explicitly selected sites,
  mask-aware pooled. `sites="blocks"` selects the outputs of the model's
  repeated block modules (measured correct on DistilBERT/BERT/RoBERTa/GPT-2);
  explicit module addresses (optionally pass-qualified `"address:N"`) select
  anything else; `sites=()` means no embedding fields. There is **no default**:
  the factory refuses and lists candidates rather than guessing.
- **Opt-in `TokenSalience`**: signed activation-times-gradient at the
  input-embedding site toward the predicted class (classifier) or the argmax
  next token (LM). Rendered verbatim by LIT's Model-provided-salience panel.
  It is NOT integrated gradients and is never called that.
- **Causal LM**: deterministic generation (`generated_text`, byte-identical to
  unbatched HF `generate`), per-token `TokenTopKPreds`, tokens.

## What deliberately does not

- **No `AttentionHeads`**: LIT deleted its attention panel upstream
  (2024-06-20); the type serializes into a void. `attention=` refuses, citing
  the removal (`lit_attention_unsupported`).
- **No LIT Integrated Gradients / HotFlip triggers, no TCAV**: the trigger
  fields 500 or fail LIT's own validator upstream; with this spec LIT
  correctly hides them, so nothing advertised can fail.
- **No vision in v1**: LIT's own image path is broken on matplotlib >= 3.9
  (unbounded upstream pin) and image models get zero counterfactual
  generators. If you try LIT's built-in image saliency, that breakage is
  LIT's, not the bridge's.
- **No custom TypeScript panel**: `layout()` is a pure-Python
  `LitCanonicalLayout`; LIT's compiled 19.5 MB client is never rebuilt.
- **A finished `Trace` refuses** (`lit_trace_not_executable`): LIT calls
  `predict()` on NEW inputs, which a historical trace cannot execute. Pass the
  live model.

## Site identity (why edits and ragged batches are safe)

Padding shifts every torchlens layer LABEL (labels count ops; padding inserts
ops) — 0/6 and 0/12 label survivors were measured across architectures. The
bridge therefore pins `(field_name, module_address, site_key, pass_index)` at
construction on an unpadded probe trace and re-resolves each request's trace
through the structural `site_key` index, refusing (`lit_site_key_drift`) when
a forward does not reproduce a pinned site. `site_key` is relative to the
traced root: a different wrapper class or a module-tree rename correctly
refuses rather than silently rebinding — that refusal is the feature working.

On decoder-only models the final block's site is PRE-final-norm (GPT-2's block
11 output is pre-`ln_f`; blocks 0-10 match HF `output_hidden_states`
bit-exactly). Comparing the last block against `hidden_states[-1]` will differ
by design.

## Padding conventions (one coherent set per path)

Classification right-pads throughout. The causal-LM adapter uses TWO
tokenizations: LEFT-padded for one batched `generate()` (right-padded batched
generation fluently echoes the prompt) and RIGHT-padded with `use_cache=False`
for the one traced forward. Mixing halves is silent wrongness (worst measured
case cosine 0.40 at HTTP 200), which is why the bridge's gates assert exact
equality, never "close enough". The tokenizer is never mutated: padding is
applied manually with `pad_token_id`, falling back to `eos_token_id`.

## Operational caveats (each earned by a measurement)

- Installing `[lit]` moves numpy 2.5.2 -> 1.26.4 and adds ~83 packages: use a
  separate environment.
- Python 3.13 source-builds numpy and shap (~100 s cold, compiler required);
  there is deliberately no version marker — the stack works.
- Upstream is dormant externally (no release since 2024-12-20; user issues
  unanswered since then). The bridge is experimental; its stability rests on
  LIT's frozen typed Model API.
- `[lit]` and `[shap]` cannot co-resolve against lit-nlp 1.3.1 (`shap<0.46`
  vs torchlens `shap~=0.46`). The bridge uses torchlens-native salience and
  does not need the shap bridge.
- Latency: a single-example CPU trace is ~0.3-2.5 s for a 6-12-block
  transformer — that is the interactive promise; batched `warm_start` is far
  cheaper per example; GPU is the real lever.
- Benign per-trace warnings you may see and should not fear: the transformers
  masking `ScalarEscapeWarning` and the functorch/vmap boundary note on causal
  masks (mask-construction ops only).
- LIT's Salience Clustering clusters LIME's salience, not the model-provided
  field (upstream constructs it from gradient-map interpreters only).
- `supports_concurrent_predictions` is `False`: torchlens tracing is not
  re-entrant; two browser tabs serialize.
- Rank-4 `[B, C, H, W]` site payloads pool by spatial mean (disclosed here);
  rank-3 payloads pool mask-aware over tokens; anything else refuses.
- `LitWidget` takes the wrapper as-is (it is a valid `Model`); nothing extra
  to build.

## Refusal codes

The bridge's stable codes (`lit_*`) are documented in the
[error-refusal contract](error_refusal_contract.md): dependency, task,
tokenizer, sites/preset/resolution, pooling, labels, salience, attention,
model-output, and capture-outcome refusals, each with a remedy.
