# Unsupervised steering with an entropy vector

Most steering vectors come from contrast pairs: activations on prompts with a behaviour minus
activations on prompts without it. An entropy vector needs no labels and no pairs. It follows
the gradient of the model's own output entropy with respect to one residual-stream site,
averaged over the model's sampled continuations, and keeps only the part of that gradient
that lies in a middle band of the site's activation covariance. Added at generated positions,
it raises the entropy of the next-token distribution; subtracted, it lowers it. Its author
reports that the effect looks more like a temperature change than a rescaling of the logits.

Source: LuxiaSL, entropy-vector (MIT license), https://github.com/LuxiaSL/entropy-vector,
checked 2026-11-03. The repository gives no paper; this page follows its README: the
gradient of mean generated-token entropy at the residual stream entering one decoder layer,
projected onto the covariance eigenvectors of ranks 16 to 255, normalised, and added at
generated positions with a coefficient times the median residual norm. The repository samples
20 continuations for the gradient and 60 for the covariance with up to 512 new tokens; this
page uses fewer and shorter ones so it runs on a CPU. For contrast-pair steering see
[coming from RepE](../migration/repe.md) and the [conditional steering](conditional_steering.md)
page; for steering at serving speed, see EasySteer and EasyEdit2 in the
[landscape](../migration/landscape.md).

The recipe downloads GPT-2 small from the Hugging Face Hub (about 500 MB) on first use and
runs on a CPU in about a minute.

## Minimal example on the running model

This page uses [the decoder on the IOI
prompt](running_examples.md#the-decoder-on-the-ioi-prompt). The entropy vector starts from
the gradient of the output entropy at one site. With the model's weights frozen nothing in the
forward pass needs a gradient, so a hook bound with `tl.when(site, hook).bind(model)` hands the
rest of the model a detached copy of the site that requires one; after one backward pass the
copy holds the gradient. The recipe below runs the method on GPT-2 small.

```python
# recipe-test: tool=transformers expected=True
import torch
import transformers

import torchlens as tl

PROMPT = "When Mary and John went to the store, John gave a drink to"
words = PROMPT.replace(",", " ,").split()
vocab = sorted(set(words))
x = torch.tensor([[vocab.index(word) for word in words]])
torch.manual_seed(0)
config = transformers.GPT2Config(
    vocab_size=len(vocab),
    n_positions=32,
    n_embd=64,
    n_layer=2,
    n_head=4,
    use_cache=False,
    attn_implementation="eager",
)
model = transformers.GPT2LMHeadModel(config).eval()

trace = tl.trace(model, x)

for parameter in model.parameters():
    parameter.requires_grad_(False)
site = {}


def as_leaf(activation, *, ctx):
    site["value"] = activation.detach().requires_grad_()
    return site["value"]


probs = tl.when(tl.module("transformer.h.0"), as_leaf).bind(model)(x).logits.softmax(-1)
(-(probs * probs.log()).sum(-1).mean()).backward()
RESULT = site["value"].grad.shape == (1, len(words), 64)
```

## 1. Sample continuations

The model writes its own data: short sampled continuations of a few neutral openings, at
temperature 0.7 and top-p 0.9 as in the source repository. Each continuation has exactly 24
new tokens, so the continuations of one opening stack into one batch.

```python
# recipe-test: tool=transformers expected=True
import torch
import transformers

import torchlens as tl

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
for parameter in model.parameters():
    parameter.requires_grad_(False)

OPENINGS = [
    "The old lighthouse keeper",
    "In the morning the market",
    "Our team spent the week",
    "The letter from my sister",
    "At the edge of the forest",
    "The new library downtown",
]
NEW = 24


def continuations(opening, n, seed):
    """``n`` sampled continuations of one opening: ids ``[n, P + NEW]`` and ``P``."""
    ids = tokenizer(opening, return_tensors="pt").input_ids
    torch.manual_seed(seed)
    out = model.generate(
        ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        min_new_tokens=NEW,
        max_new_tokens=NEW,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out, ids.shape[1]


gradient_set = [continuations(opening, 2, seed) for seed, opening in enumerate(OPENINGS)]
covariance_set = [continuations(opening, 4, 100 + seed) for seed, opening in enumerate(OPENINGS)]
RESULT = all(ids.shape[1] == prompt + NEW for ids, prompt in gradient_set + covariance_set)
```

## 2. The entropy gradient at the site

The site is the residual stream entering block 6, which is the output of block 5,
`transformer.h.5`. For each continuation the score is the mean entropy of the next-token
distribution over its generated positions. One forward pass per opening through the leaf hook
of the opening example, and one backward pass of the summed scores, gives the gradient at
every position of the site; rows are independent, so each continuation's gradient is its own. The vector's raw direction
is the gradient averaged over generated positions and then over continuations.

```python
# recipe-test: tool=transformers expected=True
SITE = "transformer.h.5"


def entropy(logits):
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


leaf = {}


def as_leaf(activation, *, ctx):
    leaf["value"] = activation.detach().requires_grad_()
    return leaf["value"]


graded = tl.when(tl.module(SITE), as_leaf).bind(model)
gradients = []
for ids, prompt in gradient_set:
    logits = graded(ids).logits
    # logits at position t predict token t + 1, so generated tokens are predicted from prompt - 1 on
    scores = entropy(logits[:, prompt - 1 : -1]).mean(-1)
    scores.sum().backward()
    gradients.append(leaf["value"].grad[:, prompt:].mean(1))
g = torch.cat(gradients).mean(0)
RESULT = g.shape == (768,) and bool(g.norm() > 0)
```

## 3. Project onto the covariance band

The covariance of the site's activations comes from the other continuations' generated
positions. Its leading eigenvectors carry the few huge directions every token shares, and its
tail is noise; the source repository keeps the eigenvectors of ranks 16 to 255 and projects
the gradient onto them. The unit result is the entropy vector. The steering scale is the
median norm of the residual at the site.

```python
# recipe-test: tool=transformers expected=True
acts = torch.cat(
    [tl.extract(model, ids, [SITE])[SITE][:, prompt:].reshape(-1, 768) for ids, prompt in covariance_set]
)
centered = acts - acts.mean(0)
covariance = centered.T @ centered / (len(acts) - 1)
eigenvalues, eigenvectors = torch.linalg.eigh(covariance)  # ascending
band = eigenvectors.flip(-1)[:, 16:256]  # ranks 16 to 255 in descending order
projected = band @ (band.T @ g)
entropy_vector = projected / projected.norm()
median_norm = acts.norm(dim=-1).median()
print(f"{len(acts)} activations, gradient share in the band {projected.norm() / g.norm():.2f}")
RESULT = entropy_vector.shape == (768,) and len(acts) > 256
```

## 4. Steer generation and measure entropy

The edit adds `alpha * median_norm * entropy_vector` to the site only on decode steps, where
the KV cache gives the site one new position; the prompt's own positions are left alone, as in
the source repository. `tl.when(site, hook).bind(model)` runs it at the cost of a forward hook
and its `.generate(...)` keeps the cache on. Greedy decoding on held-out openings returns the
scores of every step, and the measure is their mean entropy. A random direction from the same
band, at the same scale, is the control.

```python
# recipe-test: tool=transformers expected=True
HELD_OUT = ["The museum guide said", "Late last night the station", "My grandfather often told us"]
steer = {}


def add_on_decode(activation, *, ctx):
    if activation.shape[1] != 1:
        return activation
    return activation + steer["alpha"] * median_norm * steer["direction"]


bound = tl.when(tl.module(SITE), add_on_decode).bind(model)


def generated_entropy(direction, alpha):
    steer.update(direction=direction, alpha=alpha)
    values = []
    for opening in HELD_OUT:
        ids = tokenizer(opening, return_tensors="pt").input_ids
        with torch.no_grad():
            out = bound.generate(
                ids,
                do_sample=False,
                max_new_tokens=NEW,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        values.append(entropy(torch.cat(out.scores[1:])).mean())  # step 0 is the prompt
    return torch.stack(values).mean().item()


random_vector = band @ torch.randn(240, generator=torch.Generator().manual_seed(0))
random_vector = random_vector / random_vector.norm()
ALPHA = 0.25
measured = {alpha: generated_entropy(entropy_vector, alpha) for alpha in (-ALPHA, 0.0, ALPHA)}
random_shift = abs(generated_entropy(random_vector, ALPHA) - measured[0.0])
for alpha, value in measured.items():
    print(f"alpha {alpha:+.2f}: mean next-token entropy {value:.2f} nats")
print(f"random band direction at alpha {ALPHA:+.2f} moves it by {random_shift:.2f}")
RESULT = (
    measured[-ALPHA] < measured[0.0] < measured[ALPHA]
    and measured[ALPHA] - measured[-ALPHA] > 2 * random_shift
)
```

## Limits

- The vector is fitted on the model's own samples, so it is specific to the model, the site
  and the sampling settings. The source repository chooses the site per model, about halfway
  up (layer 14 of 28 and layer 16 of 32 for the two models its README uses).
- Entropy is not quality. A higher-entropy greedy continuation can repeat less or wander
  more; read the text as well as the number.
- With 24 continuations of 24 tokens the covariance has 576 samples for 768 dimensions, so
  its band is a rough estimate. The source repository uses 60 continuations of up to 512
  tokens.
