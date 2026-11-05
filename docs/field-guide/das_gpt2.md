# Fit a causal basis on GPT-2 small (distributed alignment search)

Distributed alignment search (DAS) learns an orthogonal rotation of one hidden layer such that
swapping a few rotated coordinates between two inputs changes the network's output exactly as
swapping one variable of a high-level causal model would. The share of swaps that match is the
interchange intervention accuracy (IIA); when it is near 1, the learned rows are a causal basis
for the variable. TorchLens applies a learned basis with `tl.subspace(..., method="das")`
([a rotation learned elsewhere](../intervention_explainers.md#a-rotation-learned-elsewhere)) and
ships no trainer. This page fits the basis itself, with a plain PyTorch loop over activations
TorchLens captured, then applies it with the builtin subspace edit.

Paper: Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard and Noah D. Goodman,
"Finding Alignments Between Interpretable Causal Variables and Distributed Neural
Representations", CLeaR 2024, https://arxiv.org/abs/2303.02536. The task is the
indirect-object prompt of Wang et al., "Interpretability in the Wild: a Circuit for Indirect
Object Identification in GPT-2 small", ICLR 2023, https://arxiv.org/abs/2211.00593. The same
method on a small trained MLP, with a layer comparison, is the notebook
`examples/methods/das_causal_basis_recipe.ipynb`.

The recipe downloads GPT-2 small from the Hugging Face Hub (about 500 MB) on first use and
runs on a CPU in a few minutes, most of them in the training loop.

## Minimal example on the running model

This page uses [the decoder on the IOI
prompt](running_examples.md#the-decoder-on-the-ioi-prompt). A subspace edit writes a source
run's component along a few orthonormal rows into the captured run and replays the rest of
the model; here the rows are random. The recipe below fits the rows on GPT-2 small.

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

rows = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(1)))[0].T[:2]
source = tl.trace(model, x.flip(-1))
fork = trace.fork()
fork.do(
    tl.subspace(tl.module("transformer.h.0"), rows, origin="random rows", method="das"),
    tl.patch_from(source),
)
RESULT = not torch.allclose(fork[fork.output_layers[0]].out, trace[trace.output_layers[0]].out)
```

## 1. GPT-2 small solves the task

Every prompt has the form "When A and B went to the store, S gave a drink to", with
single-token names. The subject `S` repeats one of the two names and the answer is the other
one, the indirect object. The causal variable is `IO_FIRST`: whether the indirect object is
the first name (`S == B`) or the second (`S == A`). Every prompt has the same number of
tokens, so a batch needs no padding. The model answers right when its logit for the indirect
object beats its logit for the subject.

```python
# recipe-test: tool=transformers expected=True
import torch
import transformers

import torchlens as tl

model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
for parameter in model.parameters():
    parameter.requires_grad_(False)

NAMES = [" Mary", " John", " Alice", " Bob", " Tom", " Sarah", " James", " Anna", " David", " Paul"]
NAME_IDS = torch.tensor([tokenizer(name).input_ids[0] for name in NAMES])
TEMPLATE = "When{a} and{b} went to the store,{s} gave a drink to"


def sample(n, generator):
    """Token ids ``[n, 14]``, each prompt's two name ids ``[n, 2]`` and ``IO_FIRST`` ``[n]``."""
    pairs = torch.stack([torch.randperm(len(NAMES), generator=generator)[:2] for _ in range(n)])
    io_first = torch.rand(n, generator=generator) < 0.5
    texts = [
        TEMPLATE.format(a=NAMES[a], b=NAMES[b], s=NAMES[b] if first else NAMES[a])
        for (a, b), first in zip(pairs.tolist(), io_first.tolist())
    ]
    return tokenizer(texts, return_tensors="pt").input_ids, NAME_IDS[pairs], io_first


def name_logits(logits, names):
    """The last position's logits for the prompt's first and second name, ``[n, 2]``."""
    return logits[:, -1].gather(1, names)


ids, names, io_first = sample(64, torch.Generator().manual_seed(1))
with torch.no_grad():
    says_first = name_logits(model(ids).logits, names).argmax(1) == 0
accuracy = (says_first == io_first).float().mean().item()
print(f"GPT-2 small names the indirect object on {accuracy:.0%} of 64 prompts")
RESULT = ids.shape[1] == 14 and accuracy > 0.9
```

## 2. Fit the basis with a plain PyTorch loop

An interchange runs a base prompt with `IO_FIRST` taken from a source prompt with other
names. If the model reads `IO_FIRST` from a direction at the site, writing the source's
component along that direction into the base makes the model name the base's first name
when the source's indirect object came first, and the second name otherwise. About half of
these targets differ from the base's own answer, so a basis that does nothing scores about
0.5.

The site is the residual stream after block 8, `transformer.h.8`. The source activations
come from TorchLens once (`tl.extract`, in chunks); the training loop is ordinary PyTorch.
The learned rotation's first `K` rows are the basis; a QR decomposition of a trainable
`[K, 768]` matrix keeps them orthonormal, so only those rows are parameters. The matrix starts
at unit norm: Adam's steps have a fixed size, so a start with entries of size 1 (norm about
28) would barely turn in the 32 steps below.
`tl.when(site, hook).bind(model)` runs GPT-2 with the hook and no capture, and gradients
flow through the hook into the rows while the model's own weights stay frozen.

```python
# recipe-test: tool=transformers expected=True
SITE, K = "transformer.h.8", 1

generator = torch.Generator().manual_seed(2)
train_base, train_names, _ = sample(512, generator)
train_source, _, train_source_first = sample(512, generator)
train_target = (~train_source_first).long()  # 0: the base's first name, 1: its second
source_acts = torch.cat(
    [tl.extract(model, chunk, [SITE])[SITE] for chunk in train_source.split(64)]
)

start = torch.randn(K, 768, generator=torch.Generator().manual_seed(3))
weight = torch.nn.Parameter(start / start.norm(dim=1, keepdim=True))
batch = {}


def basis():
    """``K`` orthonormal rows: the first rows of the learned rotation."""
    return torch.linalg.qr(weight.T).Q.T


def interchange(activation, *, ctx):
    rows = basis()
    return activation + (batch["source"] - activation) @ rows.T @ rows


bound = tl.when(tl.module(SITE), interchange).bind(model)
optimizer = torch.optim.Adam([weight], lr=0.01)
for epoch in range(2):
    order = torch.randperm(512, generator=torch.Generator().manual_seed(epoch))
    for rows in order.split(32):
        batch["source"] = source_acts[rows]
        logits = name_logits(bound(train_base[rows]).logits, train_names[rows])
        loss = torch.nn.functional.cross_entropy(logits, train_target[rows])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

learned = basis().detach()
print(f"last training loss {loss.item():.3f}")
RESULT = learned.shape == (K, 768) and torch.allclose(learned @ learned.T, torch.eye(K), atol=1e-5)
```

## 3. Apply it with the builtin subspace edit, against two controls

The test uses TorchLens's own edit, not the training hook. On a captured trace of held-out
base prompts, `fork.do(tl.subspace(site, rows, origin=..., method="das"),
tl.patch_from(source))` writes the source trace's component along `rows` into the base and
replays the rest of GPT-2. `origin=` and `method=` go into the edit's audit record with the
rows' digest, so a saved portable spec carries where the basis came from. The controls are a
random direction of the same size and the whole site (`tl.patch_from` with no subspace),
which also carries the source's names.

```python
# recipe-test: tool=transformers expected=True
held_out = torch.Generator().manual_seed(4)
test_base, test_names, _ = sample(32, held_out)
test_source, _, test_source_first = sample(32, held_out)
test_target = (~test_source_first).long()
source = tl.trace(model, test_source)
target = tl.trace(model, test_base)


def interchange_accuracy(selection):
    fork = target.fork()
    fork.do(selection, tl.patch_from(source))
    logits = name_logits(fork[fork.output_layers[0]].out, test_names)
    return (logits.argmax(1) == test_target).float().mean().item()


random_row = torch.randn(1, 768, generator=torch.Generator().manual_seed(5))
scores = {
    "learned basis": interchange_accuracy(
        tl.subspace(tl.module(SITE), learned, origin="DAS on IO_FIRST, GPT-2 small", method="das")
    ),
    "random direction": interchange_accuracy(
        tl.subspace(tl.module(SITE), random_row / random_row.norm(), origin="random control")
    ),
    "whole site": interchange_accuracy(tl.module(SITE)),
}
for name, score in scores.items():
    print(f"{name:>16}: IIA {score:.2f}")
RESULT = scores["learned basis"] > 0.8 and max(scores["random direction"], scores["whole site"]) < 0.7
```

The learned direction moves the model's answer as the causal model predicts on held-out
names; a random direction leaves the answer where it was, and the whole residual stream
carries the source's names along with `IO_FIRST`, so the model stops choosing between the
base's two names in the predicted way.

## Limits

- `K = 1` is the smallest basis. Raise `K` for a variable that needs more dimensions; the QR
  step keeps any number of rows orthonormal.
- The interchange replaces the component at every position of the site. The training hook
  and the test edit do the same, so the two agree; a position-specific basis needs a hook
  that slices the position.
- IIA on 32 held-out pairs is a coarse number. Report it with more pairs, and with several
  seeds for the rows, before reading small differences between sites.
- To keep the basis, save the edited fork with `tl.io.save_intervention(fork, path,
  level="portable")` after a `tl.replace_with(value)` interchange; `tl.patch_from` saves only
  at the audit level ([a rotation learned
  elsewhere](../intervention_explainers.md#a-rotation-learned-elsewhere)).
