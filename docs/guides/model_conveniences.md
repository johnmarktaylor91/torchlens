# Adding conveniences for your model

TorchLens already knows how to feed text to a Hugging Face model, how to name
ImageNet classes and where the attention heads of common transformer blocks
live. Three registries let you teach it the same things about your own model:

| You want | Register with | Every later `tl.trace` of your model |
|----------|---------------|--------------------------------------|
| Pass raw input (a string, a record) straight to `tl.trace` | `tl.inputs.register` | runs your preprocessor, keeps the raw value as `trace.input` |
| Read class labels instead of logit indices | `tl.outputs.register` | decodes the output into labelled top-k rows |
| Address a named piece of a module, such as attention heads | `tl.facets.register` | serves it as `trace.modules[...].facets[name]` |

All three take a `class_name=` that limits the convenience to models (or, for
facets, modules) of that class, and all three affect future traces only: a
trace made before you register keeps what it had.

Each example below runs on its own.

## A custom input preprocessor

The function receives the model and the raw input and returns what `forward`
takes: a tensor, a tuple of positional arguments, or a dict of keyword
arguments. Return `None` for input it does not handle, and TorchLens carries on
as if it were not registered.

```python
import torch
from torch import nn

import torchlens as tl

VOCAB = "abcdefghijklmnopqrstuvwxyz "


class CharModel(nn.Module):
    """Embeds lowercase characters and averages them into one vector."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), 8)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.embed(codes).mean(dim=1)


@tl.inputs.register(name="char_codes", class_name="CharModel")
def char_codes(model, raw):
    """Turn a string into a batch of one row of character codes."""
    if not isinstance(raw, str):
        return None  # tensors and other input take the usual path
    return torch.tensor([[VOCAB.index(ch) for ch in raw.lower() if ch in VOCAB]])


trace = tl.trace(CharModel(), "hello world")
print(trace.input)  # hello world
print(trace.provenance.input_preprocessor.identifier)  # char_codes

tl.inputs.unregister("char_codes")
```

Preprocessors run in ascending `priority=` order (default 100); the built-in
Hugging Face text, image and multimodal routes sit at 10 to 30. Passing an
explicit `capture=tl.options.CaptureOptions(input_transform=...)` bypasses every
registered preprocessor for that call.

## A custom output decoder

The function receives the model's output and returns the class labels in logit
order: a list of strings, or a dict from class index to label. TorchLens decodes
the output once at the end of each capture.

```python
import torch
from torch import nn

import torchlens as tl


class PetClassifier(nn.Module):
    """Scores a 4-number feature vector as one of three pets."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@tl.outputs.register(name="pet_labels", class_name="PetClassifier")
def pet_labels(outputs):
    """Name the three logits."""
    return ["cat", "dog", "rabbit"]


trace = tl.trace(PetClassifier(), torch.randn(2, 4))
for row in trace.decode_output(k=1)["rows"]:
    print(row["batch_item"], row["label"], round(row["prob"], 3))
print(trace.provenance.output_postprocessor.identifier)  # pet_labels

tl.outputs.unregister("pet_labels")
```

The built-in decoders (a Hugging Face `config.id2label`, verified ImageNet-1k
widths, Hugging Face text) sit at priorities 10 to 30, so a decoder at the
default priority 100 runs only when none of them claims the output; pass a
priority below 10 to go first. A registered decoder also answers an on-demand
`trace.decode_output(style="pet_labels")`, which reads the registry as it is
when you call it.

## A custom facet

A facet recipe receives the TorchLens record of a module of the registered class
and returns named `FacetSpec`s: where a value lives (here, the output op of the
query projection) plus a chain of views over it (here, split into heads).

```python
import torch
from torch import nn

import torchlens as tl
from torchlens.facets import FacetSpec

N_HEADS, D_HEAD = 2, 4


class TinyAttention(nn.Module):
    """Single-layer self-attention with two heads of width four."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(N_HEADS * D_HEAD, N_HEADS * D_HEAD)
        self.k = nn.Linear(N_HEADS * D_HEAD, N_HEADS * D_HEAD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.q(x) @ self.k(x).transpose(-2, -1)
        return scores.softmax(dim=-1) @ x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = TinyAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


@tl.facets.register(class_name="TinyAttention", target_scope="module", facets=("q_heads",))
def tiny_attention_heads(module):
    """Expose the query projection split into heads."""
    prefix = "" if module.address == "self" else f"{module.address}."
    q_proj = module.trace.modules[f"{prefix}q"]
    q_out = module.trace.ops[q_proj.calls[0].output_ops[0]]
    return {"q_heads": FacetSpec.from_home(q_out).heads(N_HEADS, D_HEAD)}


trace = tl.trace(Encoder(), torch.randn(1, 5, N_HEADS * D_HEAD))
print(trace.modules["attn"].facets["q_heads"].shape)

tl.facets.reset()  # back to the built-in recipes
```

See [Facets](../facets.md) for the full recipe contract, including reading head
counts from module config.

## All three on one model

```python
import torch
from torch import nn

import torchlens as tl
from torchlens.facets import FacetSpec

VOCAB = "abcdefghijklmnopqrstuvwxyz "
N_HEADS, D_HEAD = 2, 4


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(N_HEADS * D_HEAD, N_HEADS * D_HEAD)
        self.k = nn.Linear(N_HEADS * D_HEAD, N_HEADS * D_HEAD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.q(x) @ self.k(x).transpose(-2, -1)
        return scores.softmax(dim=-1) @ x


class MoodClassifier(nn.Module):
    """Reads a lowercase string and scores it as one of three moods."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), N_HEADS * D_HEAD)
        self.attn = TinyAttention()
        self.head = nn.Linear(N_HEADS * D_HEAD, 3)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(self.embed(codes)).mean(dim=1))


@tl.inputs.register(name="mood_text", class_name="MoodClassifier")
def mood_text(model, raw):
    if not isinstance(raw, str):
        return None
    return torch.tensor([[VOCAB.index(ch) for ch in raw.lower() if ch in VOCAB]])


@tl.outputs.register(name="mood_labels", class_name="MoodClassifier")
def mood_labels(outputs):
    return {0: "calm", 1: "cheerful", 2: "grumpy"}


@tl.facets.register(class_name="TinyAttention", target_scope="module", facets=("q_heads",))
def mood_heads(module):
    q_proj = module.trace.modules[f"{module.address}.q"]
    q_out = module.trace.ops[q_proj.calls[0].output_ops[0]]
    return {"q_heads": FacetSpec.from_home(q_out).heads(N_HEADS, D_HEAD)}


trace = tl.trace(MoodClassifier(), "what a lovely day")
print(trace.input)  # what a lovely day
print(trace.decode_output(k=1)["rows"][0]["label"])  # one of calm, cheerful, grumpy
print(trace.modules["attn"].facets["q_heads"].shape)

tl.inputs.unregister("mood_text")
tl.outputs.unregister("mood_labels")
tl.facets.reset()
```

## Where to go next

- [Semantic input and output](../semantic_io.md): the built-in routes, read-time
  `decode_output(style=, head=)` and output provenance.
- [Facets](../facets.md): specs, gradients, intervention and patching on facets.
