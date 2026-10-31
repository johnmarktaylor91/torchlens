# Conditional steering with a per-token audit

Conditional steering applies a steer only when a condition holds: a condition
direction is read at one site, its score is compared with a threshold, and the
behaviour steer at a later site is added only on the forwards where the score
passed. TorchLens has no `when=` option for this; the condition is a custom edit
written in a few lines. What TorchLens adds is the record: every forward of a
steered generation leaves a fire record, and each record says whether the edit
replaced the activation. Joined with the condition scores, that gives an audit
of every prompt and every generated token: the score, whether it passed the
threshold, and whether the steer was applied. A false trigger, a prompt that
should not have been steered but was, shows up as a row you can inspect.

The example uses a two-layer Llama with random weights and token-id prompts, so
it runs offline; with a real checkpoint, load it with `from_pretrained` and keep
the rest.

## 1. Fit the condition direction and a threshold

The condition here is "the newest token is token 7". `fit_direction`
fits it from prompts that end in that token against prompts that do not, at
the first MLP. The threshold is the midpoint between the two groups' mean
scores on fresh prompts.

```python
# recipe-test: tool=transformers expected=True
import torch
import torch.nn.functional as F
import transformers

import torchlens as tl
from torchlens.intervention import fit_direction

torch.manual_seed(0)
config = transformers.LlamaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)
model = transformers.LlamaForCausalLM(config).eval()
model.config.use_cache = False  # traces read plain forwards; generation turns the cache on
CONDITION_SITE, STEER_SITE, TRIGGER = "model.layers.0.mlp", "model.layers.1.mlp", 7


def prompts(n, triggered, seed):
    """Token-id prompts of length 6; the triggered ones end in TRIGGER."""
    ids = torch.randint(10, 64, (n, 6), generator=torch.Generator().manual_seed(seed))
    if triggered:
        ids[:, -1] = TRIGGER
    return ids


capture = tl.options.CaptureOptions(layers_to_save="all")
pos = tl.trace(model, prompts(8, True, 1), capture=capture)
neg = tl.trace(model, prompts(8, False, 2), capture=capture)
condition = fit_direction(pos, neg, CONDITION_SITE, method="pca")


def score(activation):
    """Cosine similarity of the newest position with the condition direction."""
    return float(F.cosine_similarity(activation[0, -1].detach(), condition.vector, dim=0))


def calibration_scores(ids):
    trace = tl.trace(model, ids, capture=capture)
    out = trace[CONDITION_SITE].out
    return [score(out[i : i + 1]) for i in range(ids.shape[0])]


on_scores = calibration_scores(prompts(6, True, 3))
off_scores = calibration_scores(prompts(6, False, 4))
threshold = (sum(on_scores) / 6 + sum(off_scores) / 6) / 2
RESULT = min(on_scores) > threshold > max(off_scores)
```

## 2. The conditional edit, and one steered generation per prompt

Two rules in one spec. The first reads the condition at `CONDITION_SITE`, keeps
the score, and leaves the activation untouched. It puts its decision in
`hook.run_ctx`, a dictionary that every hook of one bound call shares and that
starts empty for the next call, so no decision leaks from one prompt into the
next. The second rule adds the behaviour direction at `STEER_SITE` only when
that decision passed; otherwise it returns the activation object unchanged.

Each prompt runs in its own bound `generate` call, so one fire report belongs
to one prompt and one forward. The binding derives `replaced` from object
identity, not from anything the hook reports, which is why it is the evidence
of whether the steer was applied.

```python
# recipe-test: tool=transformers expected=True
behaviour = torch.randn(32, generator=torch.Generator().manual_seed(5))
MAGNITUDE = 8.0
scores = []


def read_condition(activation, *, ctx):
    value = score(activation)
    scores.append(value)
    ctx.run_ctx["condition_passed"] = value > threshold
    return activation


def gated_steer(activation, *, ctx):
    if not ctx.run_ctx.get("condition_passed", False):
        return activation
    return activation + MAGNITUDE * behaviour


spec = tl.when(tl.module(CONDITION_SITE), read_condition).merge(
    tl.when(tl.module(STEER_SITE), gated_steer)
)
gen_kwargs = {"max_new_tokens": 4, "do_sample": False, "use_cache": True, "pad_token_id": 0}
eval_prompts = [("trigger", ids) for ids in prompts(3, True, 6)]
eval_prompts += [("neutral", ids) for ids in prompts(3, False, 7)]

bound = spec.bind(model, on_zero_fire="error")
audit = []
for prompt_index, (label, ids) in enumerate(eval_prompts):
    scores.clear()
    outputs = bound.generate(ids[None], **gen_kwargs)
    report = bound.last_report  # this call's fire records
    steer_fires = [fire for fire in report.fires if fire["site_key"] == STEER_SITE]
    assert len(steer_fires) == len(scores)  # one condition read and one steer per forward
    new_tokens = outputs[0, ids.shape[0] :].tolist()
    for step, (value, fire) in enumerate(zip(scores, steer_fires)):
        audit.append(
            {
                "prompt": prompt_index,
                "label": label,
                "step": step,
                "token": new_tokens[step],
                "score": round(value, 3),
                "passed": value > threshold,
                "applied": fire["replaced"],
            }
        )

RESULT = all(row["applied"] == row["passed"] for row in audit)
```

Step `k` is the `k`-th forward of the generation: step 0 is the prompt
(prefill), and step `k` reads the token generated at step `k - 1`. Its score is
read at the newest input position, and the steer it gates shapes the token
generated at step `k`, shown in `token`.

## 3. Read the audit

Every row is one prompt and one generated token. On the prompt itself (step 0),
the condition passes exactly for the prompts that end in the trigger:

```python
# recipe-test: tool=transformers expected=True
for row in audit:
    print(row)

prefill = [row for row in audit if row["step"] == 0]
RESULT = all(row["passed"] == (row["label"] == "trigger") for row in prefill)
```

The rows a safety review starts from are the ones where the steer was applied
to a prompt that should not have been steered, and the ones where a prompt that
should have been steered was not:

```python
# recipe-test: tool=transformers expected=True
false_triggers = [row for row in audit if row["applied"] and row["label"] == "neutral"]
missed = [r for r in prefill if r["label"] == "trigger" and not r["applied"]]
RESULT = sorted(audit[0]) == ["applied", "label", "passed", "prompt", "score", "step", "token"]
```

## 4. A rule that never ran is an error, not a quiet audit

Because the condition lives inside the hook, the steer rule fires on every
forward and its records carry `replaced=False` when the condition failed.
Zero fires therefore means the rule never ran at all, which is a wiring
mistake, never "the condition did not pass". The recipe passes
`on_zero_fire="error"` so such a run refuses instead of producing an audit with
no applied rows. The binding's default, `"warn"`, also records the
fact on the report and warns. Here the reader is meant to run on the last
generated token only, but the step was counted from 1: `at_step(4)`. Steps
count from 0, so a four-token generation (steps 0 to 3) never reaches step 4:

```python
# recipe-test: tool=transformers expected='bind_zero_fire'
from torchlens import at_step

late = tl.when(tl.module(CONDITION_SITE) & at_step(4), read_condition).merge(
    tl.when(tl.module(STEER_SITE), gated_steer)
)
try:
    late.bind(model, on_zero_fire="error").generate(eval_prompts[0][1][None], **gen_kwargs)
except Exception as error:
    RESULT = error.fields["code"]
```

## Limits

- One prompt per call. The condition decides per forward, and a fire record
  covers the whole batch at that site; with a batch, gate each row inside the
  hook and keep your own per-row record of which rows were steered.
- The score is read at the newest position only, and the steer is added at
  every position of that forward (all prompt positions at step 0, the one new
  position after that).
- The condition reader and the steer are opaque callables: the spec is
  audit-only and does not travel in a portable artifact.
