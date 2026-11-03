# The two running examples

Every Field guide page starts from one of two small models and the same line,
`trace = tl.trace(model, x)`, then adds one verb. The decoder is for language-model work
(the indirect-object prompt from Wang et al. 2022); the CNN is for vision and NeuroAI work
(ResNet-18 on one image). Both run on a CPU with nothing downloaded.

<!-- JMT review: plain factual prose, written for review. -->

## The decoder on the IOI prompt

A two-layer GPT-2-shaped decoder with random weights and a word-level vocabulary built from
the prompt. The structure (embeddings, attention blocks, MLPs, the unembedding) is GPT-2's;
the weights are not trained, so the predictions mean nothing. The swap to the real GPT-2 small
is shown below the fence.

```python
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
```

The same page with the trained model, which downloads GPT-2 small from the Hugging Face Hub:

```text
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
x = tokenizer(PROMPT, return_tensors="pt").input_ids

trace = tl.trace(model, x)
```

## ResNet-18 on one image

torchvision's ResNet-18 with random weights on one seeded image-shaped tensor. Swap in
`weights="IMAGENET1K_V1"` and a real preprocessed image for meaningful activations; nothing
else on any page changes.

```python
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.rand(1, 3, 224, 224)

trace = tl.trace(model, x)
```

## What both fences give you

`trace` holds every operation the forward executed, in order, with its output tensor, the
module that ran it and the arguments it was called with. The capture was checked against the
real forward by replay unless the trace says otherwise. Run after the ResNet fence:

```python
assert torch.equal(trace.output, model(x))
print(len(trace.ops), "operations recorded")
print(trace.summary())
```

## Pages that use them

| Task | Running example | Page |
| --- | --- | --- |
| See the graph and inspect any operation | both | [see_and_inspect.md](see_and_inspect.md) |
| Extract features for a stimulus set (NeuroAI) | ResNet-18 | [extract_features.md](extract_features.md) |
| Intervene on a site and measure the effect | the decoder | [intervene.md](intervene.md) |
