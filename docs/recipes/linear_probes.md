# Linear probes: leakage-safe splits, an obfuscation check and a causal check

A linear probe asks whether a property can be read off one layer by a linear
classifier: extract the layer's activation for every token, label each token,
train a logistic regression, and score it on held-out data. A probe score is a
claim about what the layer carries, and three things undermine it: a split that
leaks the test set into training, a perturbation that hides the property from
the probe while the model behaves the same, and a direction the model does not
use. This recipe trains the probe with scikit-learn and covers each of those,
then streams the probe's score over every token of a generation. TorchLens
supplies the activations, the steering and the per-token scores; scikit-learn
supplies the probe and the metrics.

```text
pip install torchlens transformers scikit-learn
```

## Real use: a trained model and a concept you label

With a trained model, load it with `from_pretrained`, label each token or each
episode with the concept from data you trust (a truthfulness label, a flagged
episode, a refusal), and keep the rest of the page. This is not run here:

```text
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
SITE = "transformer.h.6.ln_1"  # the residual stream entering block 6
```

The rest of the page runs on a two-layer GPT-2 with random weights, built from a
config with no download, so it shows the mechanics only. A random network has
learned no concepts, so the probe reads one it carries anyway: whether the
current token's id is even, which the residual stream holds because it contains
the token's embedding. With a trained model and a real concept, expect lower and
noisier scores, and expect the causal check to be able to fail.

## 1. Extract one activation per token

Every row is one token: its activation at `SITE`, its label, and the id of the
prompt it came from. `SITE` is the first layer norm of block 1, the normalised
residual stream entering that block; a module whose output is a tensor, not a
block, whose output is a tuple. One `tl.extract` call reads the whole batch of
prompts, which here use 32 token ids, 16 of them even.

```python
# recipe-test: tool=sklearn expected=((384, 64), 48, True)
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

torch.manual_seed(0)
config = GPT2Config(
    n_layer=2,
    n_head=2,
    n_embd=64,
    vocab_size=512,
    n_positions=64,
    bos_token_id=0,
    eos_token_id=0,
    attn_implementation="eager",
    use_cache=False,
)
model = GPT2LMHeadModel(config).eval().requires_grad_(False)
SITE = "transformer.h.1.ln_1"

N_PROMPTS, LENGTH = 48, 8
generator = torch.Generator().manual_seed(0)
prompts = torch.randint(100, 132, (N_PROMPTS, LENGTH), generator=generator)
acts = tl.extract(model, prompts, {"act": SITE})["act"]  # [prompt, position, width]

X = acts.reshape(-1, config.n_embd).numpy()
tokens = prompts.reshape(-1).numpy()
y = (tokens % 2 == 0).astype(int)  # the concept: the current token's id is even
prompt_id = np.repeat(np.arange(N_PROMPTS), LENGTH)
RESULT = (X.shape, len(set(prompt_id)), bool(0.4 < y.mean() < 0.6))
```

## 2. Train with scikit-learn and report the standard metrics

Hold out whole prompts, train a seeded `LogisticRegression` on the rest, and
report the five metrics a probe result should carry. Accuracy and F1 read the
thresholded prediction; ROC AUC and average precision read the score and do not
depend on the threshold. Token identity is trivially readable from a random
network's residual stream, so every metric is 1.0 here; a real concept is not.

```python
# recipe-test: tool=sklearn expected=True
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
train, test = next(split.split(X, y, groups=prompt_id))
clf = LogisticRegression(max_iter=1000, random_state=0).fit(X[train], y[train])
pred, score = clf.predict(X[test]), clf.decision_function(X[test])
metrics = {
    "accuracy": accuracy_score(y[test], pred),
    "balanced_accuracy": balanced_accuracy_score(y[test], pred),
    "f1": f1_score(y[test], pred),
    "roc_auc": roc_auc_score(y[test], score),
    "average_precision": average_precision_score(y[test], score),
}
print({name: round(value, 3) for name, value in metrics.items()})
RESULT = all(value > 0.9 for value in metrics.values())
```

## 3. Leakage-safe splits

The rule: never put tokens of one prompt or episode in both train and test, and
treat repeated copies of a prompt as one prompt. Tokens of a prompt share its
context, and a repeated prompt yields identical rows, so a token-level split
lets the probe memorise prompts instead of learning the concept. The classic
case is a dataset where each prompt was logged several times (resampled
episodes). Below, twelve prompts are logged four times each and carry an
episode label that is a coin flip per prompt, so there is nothing to learn: the
honest score is chance. A token-level `KFold` reports about 0.8 anyway, because
every test row has an exact copy in training; `GroupKFold` by prompt id reports
chance. On the real concept the two splits agree,
because that signal does not depend on seeing the prompt.

```python
# recipe-test: tool=sklearn expected=(True, True, True)
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


def cv_accuracy(features, labels, cv, groups=None):
    probe = LogisticRegression(max_iter=1000, random_state=0)
    scores = cross_val_score(probe, features, labels, cv=cv, groups=groups)
    return scores.mean()


token_split = KFold(n_splits=4, shuffle=True, random_state=0)
prompt_split = GroupKFold(n_splits=4)

logged = np.tile(np.flatnonzero(prompt_id < 12), 4)  # each of 12 prompts, four times
coin = np.random.default_rng(0).permutation([0, 1] * 6)  # one flip per prompt
X_log, groups_log = X[logged], prompt_id[logged]
y_log = coin[groups_log]
leaky = cv_accuracy(X_log, y_log, token_split)
honest = cv_accuracy(X_log, y_log, prompt_split, groups_log)
real_leaky = cv_accuracy(X, y, token_split)
real_honest = cv_accuracy(X, y, prompt_split, prompt_id)
print(round(leaky, 2), round(honest, 2), round(real_leaky, 2), round(real_honest, 2))
RESULT = (leaky > 0.75, honest < 0.6, abs(real_leaky - real_honest) < 0.05)
```

## 4. The few-positive case

A concept with few positives, here "the token is one of two ids" (about 5
percent of tokens), breaks accuracy as a metric: a probe that never fires is
right about 95 percent of the time. Weight the classes with
`class_weight="balanced"`, stratify the folds so every fold holds positives
(`StratifiedGroupKFold` keeps the prompt grouping), and report precision, recall
and average precision, whose chance level is the positive rate, not one half.

```python
# recipe-test: tool=sklearn expected=(True, True, True)
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

y_rare = np.isin(tokens, (105, 112)).astype(int)
rare_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
weighted = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
rare_score = cross_val_predict(
    weighted, X, y_rare, cv=rare_split, groups=prompt_id, method="decision_function"
)
rare_pred = (rare_score > 0).astype(int)
never_fires = np.zeros_like(y_rare)
report = {
    "positive_rate": y_rare.mean(),
    "never_fires_accuracy": accuracy_score(y_rare, never_fires),
    "never_fires_recall": recall_score(y_rare, never_fires),
    "precision": precision_score(y_rare, rare_pred),
    "recall": recall_score(y_rare, rare_pred),
    "average_precision": average_precision_score(y_rare, rare_score),
}
print({name: round(value, 3) for name, value in report.items()})
RESULT = (
    report["never_fires_accuracy"] > 0.9 and report["never_fires_recall"] == 0.0,
    report["recall"] > 0.8,
    report["average_precision"] > 5 * report["positive_rate"],
)
```

## 5. Obfuscation check

The obfuscated-activations attack (Bailey et al., 2024) asks whether an input
can hide the concept from the probe while the model's behaviour stays the same.
Optimise a small perturbation of the input embeddings that pushes every flagged
token's probe score below the threshold, while a KL penalty keeps the
next-token distribution close to the clean one. One bound spec does both
halves: a rule at `transformer.wte` adds the perturbation, and a rule at `SITE`
reads the activation the probe sees, gradients included.

```python
# recipe-test: tool=sklearn expected=(True, True, True)
import torch.nn.functional as F

w = torch.tensor(clf.coef_[0], dtype=torch.float32)
b = float(clf.intercept_[0])
held_out = np.unique(prompt_id[test])
victims = prompts[torch.as_tensor(held_out[:4])]
delta = torch.zeros(len(victims), LENGTH, config.n_embd, requires_grad=True)
seen = {}


def perturb(activation, *, ctx):
    return activation + delta


def read_site(activation, *, ctx):
    seen["act"] = activation
    return activation


attacked = (
    tl.when(tl.module("transformer.wte"), perturb)
    .merge(tl.when(tl.module(SITE), read_site))
    .bind(model)
)


def kl_from(clean_logp, logits):
    """Mean KL(clean || perturbed) of the next-token distribution over positions."""
    return (clean_logp.exp() * (clean_logp - F.log_softmax(logits, -1))).sum(-1).mean()


with torch.no_grad():
    clean_logp = F.log_softmax(model(victims).logits, -1)
    attacked(victims)
    flagged = seen["act"] @ w + b > 0  # tokens the probe flags before the attack
    embed_norm = model.transformer.wte(victims).norm()

optimizer = torch.optim.Adam([delta], lr=3e-4)
for _ in range(40):
    optimizer.zero_grad()
    logits = attacked(victims).logits
    evade = F.relu(seen["act"] @ w + b + 1.0)[flagged].mean()
    (evade + 10.0 * kl_from(clean_logp, logits)).backward()
    optimizer.step()

with torch.no_grad():
    kl = kl_from(clean_logp, attacked(victims).logits).item()
    evaded = (seen["act"] @ w + b)[flagged] < 0
relative_norm = (delta.norm() / embed_norm).item()
print(f"evaded {evaded.float().mean().item():.2f}, KL {kl:.2e}, relative norm {relative_norm:.2f}")
RESULT = (bool(evaded.all()), kl < 1e-2, relative_norm < 0.5)
```

Read the relative norm (about 0.3 here) as the cost of evasion. A probe that a perturbation small
next to the input flips while the output barely moves is fragile as a monitor:
an adversary, or ordinary optimisation pressure against the probe, can hide the
concept from it. On a random network the KL bound is easy to meet, because the
output depends weakly on the input; on a trained model the KL term is what makes
the attack hard, and its size at a given evasion rate is the number to report.

## 6. Causal check through the intervention bundle

A probe direction is evidence the layer carries the concept, not that the model
uses it. Steer the site along the probe's unit direction with growing strength
and read two things from one `tl.record` per strength: the model output, whose
KL from the unsteered next-token distribution must grow with the strength, and
the steered site, whose probe score must move with it (the check that the edit
landed where the probe reads).

```python
# recipe-test: tool=sklearn expected=(True, True, True)
direction = w / w.norm()
site = tl.module(SITE)
with torch.no_grad():
    base_logp = F.log_softmax(model(victims).logits, -1)

kls, site_scores = [], []
for alpha in (0.0, 1.0, 2.0, 4.0, 8.0):
    spec = tl.when(site, tl.steer(direction, strength=alpha, dim=-1))
    out, rec = tl.record(model, victims, save=site, intervene=spec, return_output=True)
    steered = rec.to_trace().find_sites(site).first().out
    kls.append(kl_from(base_logp, out.logits).item())
    site_scores.append((steered @ w + b).mean().item())


def increasing(values):
    return all(a < b for a, b in zip(values, values[1:]))


print([f"{value:.2e}" for value in kls])
RESULT = (kls[0] < 1e-6, increasing(kls), increasing(site_scores))
```

This passes: the output moves further from the unsteered distribution at every
step of the strength. It would fail if the KL stayed at zero as the strength
grew: then the direction is a correlate the layer carries and nothing downstream
reads, and the probe says nothing about what the model does. On a trained model,
also compare against random directions of the same norm and check that the
output moves toward the behaviour the concept predicts, not just away from the
clean output.

## 7. Streaming the probe over a generation

To watch the concept token by token, give the fitted probe to the recorder as a
reducer: each forward of a cached `generate()` replaces the site's activation
with the probe's decision function at the positions that forward computed, and
the full activation is released. The series holds the prompt positions first
(the prefill), then one score per generated token. The check below compares the
prefill scores with `clf.decision_function` on activations extracted separately.

```python
# recipe-test: tool=sklearn expected=(True, True, 16)
from torchlens.recording import GenerationOptions, reducers

prompt = prompts[torch.as_tensor(held_out[:1])]
P = prompt.shape[1]
gen = tl.record(
    model,
    prompt,
    generate_kwargs={"max_new_tokens": 16, "do_sample": False, "use_cache": True},
    generation=GenerationOptions(reducers={SITE: reducers.probe(clf)}),
)
series = gen.series(SITE)  # TokenSeries: .positions and .values, prompt first

prefill = tl.extract(model, prompt, {"act": SITE})["act"][0].numpy()
expected = torch.from_numpy(clf.decision_function(prefill)).float()
RESULT = (
    bool(torch.allclose(series.values[:P].float(), expected, atol=1e-4)),
    list(series.positions) == list(range(len(series.values))),
    len(series.values) - P,
)
```

## Limits

- A probe finds what is linearly readable at one site; a failed probe does not
  show the concept is absent, and a passed one does not show the model uses it
  (that is what step 6 is for).
- Choose the site, the regularisation and the split before looking at test
  scores; sweeping them against the test folds is another leak.
- The obfuscation check here perturbs embeddings with a white-box gradient. An
  attacker restricted to discrete tokens is weaker; report which threat model
  you tested.
- Streaming scores come from one prompt per generation; batched prompts, beams
  and `intervene=` refuse in the generation recorder.
