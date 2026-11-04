# Adding conveniences for your model

TorchLens already knows how to feed text to a Hugging Face model, how to name
ImageNet classes and where the attention heads of common transformer blocks
live. Three registries let you teach it the same things about your own model:

| You want | Register with | Every later `tl.trace` of your model |
|----------|---------------|--------------------------------------|
| Pass raw input (a string, a record) straight to `tl.trace` | `tl.preprocessing.register` | converts the raw input before it reaches `forward` |
| Read class labels instead of logit indices | `tl.postprocessing.register` | decodes the output into labelled top-k rows |
| Address a named piece of a module, such as attention heads | `tl.facets.register` | serves it as `trace.modules[...].facets[name]` |

Each registers a plain function and affects future traces only: a trace made
before you register keeps what it had. `unregister(name)` (or `tl.facets.reset()`)
removes it again. The two processing namespaces load on import:
`import torchlens.preprocessing` and `import torchlens.postprocessing`.

Each example below runs on its own.

## A custom input preprocessor

An input detector receives the model, the raw input and the rest of the
`tl.trace` call as keyword arguments. It returns `None` for input it does not
handle, and TorchLens carries on as if it were not registered. To claim the
input, it converts it and calls `tl.trace` again with the converted value,
passing `**kwargs` through so the caller's `capture=`, `save=` and other options
still apply. The converted value is a tensor the detector itself declines, so
the second call takes the ordinary path.

```python
import torch
from torch import nn

import torchlens as tl
import torchlens.preprocessing

VOCAB = "abcdefghijklmnopqrstuvwxyz "


class CharModel(nn.Module):
    """Embeds lowercase characters and averages them into one vector."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), 8)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.embed(codes).mean(dim=1)


def encode_chars(text):
    """Turn a string into a batch of one row of character codes."""
    return torch.tensor([[VOCAB.index(ch) for ch in text.lower() if ch in VOCAB]])


@tl.preprocessing.register(name="char_codes", priority=5)
def char_codes(model, raw, **kwargs):
    """Trace a CharModel called on a string."""
    if not isinstance(model, CharModel) or not isinstance(raw, str):
        return None  # tensors and other models take the usual path
    return tl.trace(model, encode_chars(raw), **kwargs)


trace = tl.trace(CharModel(), "hello world")
print(trace.input_layers)  # one input layer: the character codes of "hello world"

tl.preprocessing.unregister("char_codes")
```

Detectors run in ascending `priority=` order (default 100); the built-in Hugging
Face text, multimodal and image routes sit at 10, 20 and 30, so a priority
below 10 goes first. A call that already chooses its own input handling skips
every detector: an explicit `capture=tl.options.CaptureOptions(input_transform=...)`,
a `backend=` or a `chunk_size=`.

## A custom output decoder

An output detector receives the model's output and a `meta` dict and returns a
`ResolvedPostprocessing` record or `None`. For class labels the record names
the style `"classification"` and carries the labels in `config["id2label"]`.
TorchLens runs the detectors once at the end of each capture (`meta["style"]`
is `None` then) and again whenever you call
`trace.decode_output(style=...)` with a style no built-in decoder handles.
`meta["model_class_name"]` is the class name of the traced model.

```python
import torch
from torch import nn

import torchlens as tl
import torchlens.postprocessing
from torchlens.postprocessing import ResolvedPostprocessing

PET_LABELS = {0: "cat", 1: "dog", 2: "rabbit"}


class PetClassifier(nn.Module):
    """Scores a 4-number feature vector as one of three pets."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@tl.postprocessing.register(name="pet_labels", priority=5)
def pet_labels(outputs, meta):
    """Name the three logits of a PetClassifier."""
    if meta.get("model_class_name") != "PetClassifier":
        return None
    if meta.get("style") not in (None, "pet_labels"):
        return None
    return ResolvedPostprocessing(
        source="project",
        identifier="pet_labels",
        verified=True,
        config={"num_labels": len(PET_LABELS), "id2label": PET_LABELS},
        description="pet label bank",
        style="classification",
        selected_output_head=meta.get("head"),
        label_source="pet_labels",
        label_source_version="v1",
        confidence=1.0,
    )


trace = tl.trace(PetClassifier(), torch.randn(2, 4))
for row in trace.decode_output(k=1)["rows"]:
    print(row["batch_item"], row["label"], round(row["prob"], 3))
print(trace.provenance.output_postprocessor.identifier)  # pet_labels

tl.postprocessing.unregister("pet_labels")
```

The built-in decoders (a Hugging Face `config.id2label`, verified ImageNet-1k
widths, Hugging Face text) sit at priorities 10 to 30, so a decoder at the
default priority 100 runs only when none of them claims the output; pass a
priority below 10 to go first. An on-demand `trace.decode_output(style="pet_labels")`
reads the registry as it is when you call it.

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
import torchlens.postprocessing
import torchlens.preprocessing
from torchlens.facets import FacetSpec
from torchlens.postprocessing import ResolvedPostprocessing

VOCAB = "abcdefghijklmnopqrstuvwxyz "
MOODS = {0: "calm", 1: "cheerful", 2: "grumpy"}
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


@tl.preprocessing.register(name="mood_text", priority=5)
def mood_text(model, raw, **kwargs):
    if not isinstance(model, MoodClassifier) or not isinstance(raw, str):
        return None
    codes = torch.tensor([[VOCAB.index(ch) for ch in raw.lower() if ch in VOCAB]])
    return tl.trace(model, codes, **kwargs)


@tl.postprocessing.register(name="mood_labels", priority=5)
def mood_labels(outputs, meta):
    if meta.get("model_class_name") != "MoodClassifier" or meta.get("style") is not None:
        return None
    return ResolvedPostprocessing(
        source="project",
        identifier="mood_labels",
        verified=True,
        config={"num_labels": len(MOODS), "id2label": MOODS},
        description="mood label bank",
        style="classification",
        selected_output_head=meta.get("head"),
    )


@tl.facets.register(class_name="TinyAttention", target_scope="module", facets=("q_heads",))
def mood_heads(module):
    q_proj = module.trace.modules[f"{module.address}.q"]
    q_out = module.trace.ops[q_proj.calls[0].output_ops[0]]
    return {"q_heads": FacetSpec.from_home(q_out).heads(N_HEADS, D_HEAD)}


trace = tl.trace(MoodClassifier(), "what a lovely day")
print(trace.decode_output(k=1)["rows"][0]["label"])  # one of calm, cheerful, grumpy
print(trace.modules["attn"].facets["q_heads"].shape)

tl.preprocessing.unregister("mood_text")
tl.postprocessing.unregister("mood_labels")
tl.facets.reset()
```

## Where to go next

- [Semantic input and output](../semantic_io.md): the built-in routes, read-time
  `decode_output(style=, head=)` and output provenance.
- [Facets](../facets.md): specs, gradients, intervention and patching on facets.
