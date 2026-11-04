# <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/logo.png" width=8% align="absmiddle"> TorchLens

[![PyPI version](https://img.shields.io/pypi/v/torchlens.svg)](https://pypi.org/project/torchlens/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchlens.svg)](https://pypi.org/project/torchlens/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Nightly](https://github.com/johnmarktaylor91/torchlens/actions/workflows/nightly.yml/badge.svg)](https://github.com/johnmarktaylor91/torchlens/actions/workflows/nightly.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Inspect, visualize and intervene on any operation in any neural network.**

Understanding and controlling the computations inside neural networks is [the most important scientific challenge of our time](https://darioamodei.com/post/the-urgency-of-interpretability) (at least). Progress on this problem, known as *mechanistic interpretability*, would help [make AI more beneficial for society](https://www.goodfire.com/research/interpretability-for-alzheimers-detection), [illuminate how our brains work](https://doi.org/10.1146/annurev-vision-082114-035447), and guard against undesirable behaviors like [deception](https://arxiv.org/abs/2412.14093) or [killing us all](https://ai-2027.com/).

Some aspects of mechanistic interpretability are deep and open-ended. We need to understand dauntingly complex systems that may soon outsmart us, with new frontier models arriving faster than we can study the current ones. It's not even clear what a "solution" would look like, or how many obstacles remain. We'll surely need help from the AI agents themselves.

Other aspects are just plumbing, and while tedious, they are finite: the same small set of tasks for every model, like extracting and editing its activations and gradients. For instance, PyTorch has long lacked a built-in way to extract every intermediate activation from arbitrary networks (an official docstring admits this can take ["some massaging"](https://github.com/pytorch/vision/blob/e334535aaf10e3327b618e1f9f069e09d18ebb8e/torchvision/models/feature_extraction.py#L391-L405)). But all networks are built from the same basic parts, so a few well-made tools should work on all of them.

The modest goal of TorchLens is to fix these plumbing problems once and for all, so you and your agents can spend your time and tokens on the deep questions, and maybe even solve some of them.

## What TorchLens does

Almost everything you do in TorchLens starts with one function, `tl.trace`. You don't have to set anything up. Pass it any PyTorch model and its input exactly as they are, with no hooks, wrappers or changes to your code:

```python
import torchlens as tl
from transformers import GPT2LMHeadModel

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()
text = "The capital of France is"

trace = tl.trace(gpt2, text)              # run model as-is, logging every operation
op = trace["tanh_6"]                      # fetch the 6th tanh operation
op.out.shape                              # torch.Size([1, 5, 3072])
op.parents, op.children                   # the operations before and after tanh_6
mlp = trace.modules["transformer.h.5.mlp"]  # look up a module's record by its address
mlp.total_flops_forward                     # MLP's forward FLOPs (one of 800+ metadata fields in TorchLens)
trace.draw(module="transformer.h.5.mlp.act")  # draw the graph of the MLP's GELU activation module
```

<picture>
  <img alt="TorchLens drawing of every operation inside the GELU module of GPT-2 block 5's MLP" src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/readme_gelu_v3.svg" width="400">
</picture>

This figure zooms in on the MLP's GELU module; here is [the whole GPT-2 graph](https://modelmenagerie.ai/models/gpt2-huggingface/). Each node is one tensor operation. For instance, `tanh_6_266` is the 6th `tanh` operation and the 266th operation overall. The function name and call number are enough on their own, so `trace["tanh_6"]` and `trace["tanh_6_266"]` return the same operation. A trace can also be indexed by a module's address (`trace["transformer.h.5.mlp"]`) or by a selector (below).

That one `tl.trace` call runs your model as usual and logs the result of every operation it executes, not just module outputs. Because the `Trace` object returned by this function holds every step of the forward pass, it can answer questions that require the whole computational graph: which units influence which other units (receptive and projective fields), whether replaying each step reproduces its saved result (`tl.validate`), or how results change downstream when you edit one activation. A trace with gaps (e.g., module outputs only) cannot fully answer such questions.

Since the trace holds everything, TorchLens doesn't need a separate tool for every question. A small "basis set" of composable tools covers just about anything you'd want to do when inspecting or intervening on a network:

**Select** activations or gradients by any criteria, from simple ("every LayerNorm in the `transformer.h.5` module") to sophisticated ("every unit in `layer1` and `layer2` of a ResNet affected by a specific set of units in `conv1`"). Any set of units you can describe by their kind, location and causal links to other units can be picked out with a single selector like those below ([provably](docs/reference/selection_completeness.md)).

```python
trace[tl.func("layer_norm") & tl.in_module("transformer.h.5")]           # by function and module
trace["conv1"].projective_field.of(units, within=["layer1", "layer2"])   # by downstream influence (ResNet-18)
```

**Extract** activations at whatever speed or level of detail you need: save them all, stop the forward pass once you have the ones you want, or quickly run a dynamic model many times while saving only those matching your criteria (e.g., every ReLU after a convolution). Saved activations can stay connected to the network if you want, so you can train it on any loss computed from them. You can also apply a function to each activation as it is saved (e.g., averaging across token positions or moving it to the CPU), and even go below Python function calls to the results of PyTorch's low-level ATen operations.

```python
trace = tl.trace(
    gpt2, text,
    save=tl.func("tanh"),                             # keep only the tanh ops
    stop_after="transformer.h.5",                     # stop forward pass after block 5
    capture=tl.options.CaptureOptions(
        aten_ops="values",                            # save ATen results too
        activation_transform=lambda x: x.mean(1),     # also save each activation averaged across token positions
    ),
)
trace["tanh_6"].out.norm().backward()                 # train on any function of saved activations
trace["tanh_6"].transformed_out                       # the saved average across token positions, shape (1, 3072)
trace["tanh_6"].aten_ops[0].out                       # output of the first ATen call beneath tanh_6
```

Gradients work the same way: ask for them when you call `tl.trace`, run a backward pass from any loss you want, and read the gradient at any operation, or at the `grad_fn` nodes PyTorch runs between operations.

```python
trace = tl.trace(
    gpt2, text,
    capture=tl.options.CaptureOptions(save_grads=True),  # record gradients when a backward pass runs
)
loss = trace["lm_head"].out[0, -1].max()                 # any scalar can serve as the loss; here, the top next-word score
trace.backward(loss)                                     # run the backward pass from the loss
trace["tanh_6"].grad.shape                               # torch.Size([1, 5, 3072]): the gradient at tanh_6's output
trace["tanh_6"].grad_fn.calls[0].grad_inputs[0].shape    # torch.Size([1, 5, 3072]): the gradient passed back to tanh_6's input
```

**Inspect** every part of a network in exhaustive detail. Every operation, module, parameter, buffer, and `grad_fn` node gets its own data structure, as does the trace itself, with [800+ metadata attributes](docs/reference/glossary.md) in all, covering everything from each operation's output shape and FLOPs to the line of code that ran it, whether it was inside an `if` branch, and the random number state when it ran. You can summarize this metadata in a readable table, or export it to pandas (one row per operation, module, parameter, etc.).

```python
print(trace["tanh_6"])                       # everything TorchLens knows about one operation
# Layer tanh_6: float32 [1, 5, 3072] on cpu, min -1.000 max 0.9937 mean -0.4892 sd 0.4728
#   position  operation 266 of 541
#   module    transformer.h.5.mlp.act
#   output    15,360 values, 60.0 KiB
#   values    min -1.000  max 0.9937  mean -0.4892  sd 0.4728
#   graph     mul_23 -> tanh_6 -> add_25 (reaches 13 of 25 model outputs)
#   function  tanh, gradient TanhBackward0, 151 us
#   source    transformers/activations.py:66 in forward
#   more      .ops  .parents  .children  .to_pandas()
trace["tanh_6"].code_context[-1].call_line   # the exact line of source code that called tanh_6
# 'return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))'
trace.params["lm_head.weight"].all_module_addresses   # GPT-2 reuses its embedding weights as its output layer
# ['transformer.wte', 'lm_head']
```

**Visualize** a model's computational graph using one-line defaults, premade styles for common tasks (e.g., debugging, profiling, seeing what an intervention changed), or full customization. TorchLens can show recurrent networks "rolled" or "unrolled", collapse nested modules to any depth (or pick a readable depth automatically), and draw the backward pass. Each node (one operation) can also show its own content, like input images or text, output category labels, or MDS plots of intermediate layer representations. Any visual attribute, such as node size or color, can be mapped to any metadata you like (e.g., memory use or execution time). The [visualization guide](docs/guides/visualization.md) explains every part of a drawing and every option.

```python
trace.draw(collapse="auto")                      # fold nested modules into boxes as needed to keep the graph readable
trace.draw(rolled=True)                          # draw a loop's layers once each, with an arrow back for the repeats
trace.draw(color_by="time", size_by="bytes")     # color nodes by execution time, size them by memory use
```

**Intervene** on any activation or gradient (using simple syntax based on Judea Pearl's [do-calculus](https://www.amazon.com/dp/1541608984)) to ablate, steer, patch in activations from another run (20+ built-in edits), or apply your own function. Edits can propagate through the saved graph without rerunning the model, or be applied to a fresh run.

```python
site = tl.module("transformer.h.5.mlp")                   # select block 5's MLP to edit
ablated = trace.fork().do(site, tl.zero_ablate())         # copy the trace and zero the MLP in the copy
italy = tl.trace(gpt2, "The capital of Italy is")         # trace the same model on an Italy prompt
patched = italy.fork().do(site, tl.patch_from(trace))     # patch in the MLP's activations from the France run
```

**Compare** any set of traces (e.g., clean vs. patched, different inputs, different checkpoints) operation by operation by grouping them into a `Bundle` object: compute metrics across them at any operation, find where they diverge most, and show the differences on the graph.

```python
runs = tl.Bundle({"italy": italy, "patched": patched})         # group two Traces into one Bundle
runs.rank_changes("italy", k=10)      # the 10 operations the patch changed most
runs.draw(graph="diff")               # draw the model with those changes highlighted
```

**Debug** models with over 20 convenient functions for finicky tasks like locating the first operation that produces a NaN, finding dead units and exploding gradients, and inferring what input shape a model expects. Each result names the operation and the line of code that ran it.

```python
tl.debug.find_nan(model, x)                    # finds the first operation that produced a NaN and its line of code
tl.debug.dead_neurons(trace)                   # dead units in each layer
tl.debug.grad_health(trace)                    # vanishing and exploding gradients, after trace.backward(loss)
tl.debug.guess_input_shape(model).shape        # (1, 16): input shape inferred from the model alone
```

Every tool above is designed to work on [*any* PyTorch model](#model-menagerie), including ones with loops, conditional branches, and other dynamic behavior. For example, this small model has a conditional branch and a loop. Setting `rolled=True` draws each operation in the loop as one node, no matter how many times it runs; the default (unrolled) gives each pass its own node:

```python
class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        if x.sum() < 0:     # branch
            x = -x
        for _ in range(2):  # 2 passes
            x = self.fc2(torch.relu(self.fc1(x)))
        return x

loop = Loop()
x = torch.randn(1, 4)
trace = tl.trace(loop, x)
trace.draw(rolled=True, direction="leftright")  # rolled (first picture)
trace.draw(direction="leftright")               # unrolled (second picture)
```

<p align="center">
<img alt="The loop model drawn rolled, left to right: the input splits to a sum test, labelled IF, that lands on a yellow IF: TRUE node, and to a negation, labelled THEN, that feeds a row of first Linear, ReLU and second Linear drawn once each and marked x2; a short arc under the row returns from the second Linear to the first, labelled with the passes it carries, and the second Linear's last pass leaves to the output on the right." src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/readme_loop_v6.png" width="100%">
<br><br>
<img alt="The same loop model unrolled, left to right: after the IF test and the THEN negation, the two passes of first Linear, ReLU and second Linear are laid out as six separate nodes in one row, each named with its pass number, ending at the output." src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/readme_loop_v6_unrolled.png" width="100%">
</p>

TorchLens also works at every scale, from single ATen calls up to agent harnesses that call the model many times. For autoregressive language models, TorchLens can trace the whole generation loop. Give it `gpt2.generate`, and each new token gets its own pass through the model, so you can pull out the activations for any token ([guide](docs/guides/kv_cache.md#trace-a-generation-loop)):

```python
gen = tl.trace(gpt2.generate, "The capital of France is", input_kwargs={"max_new_tokens": 4})
gen["transformer.h.5.mlp"].num_calls     # 4: one call per generated token
gen["transformer.h.5.mlp:1"].out.shape   # torch.Size([1, 5, 768]): the first call reads the whole prompt
gen["transformer.h.5.mlp:4"].out.shape   # torch.Size([1, 1, 768]): later calls read only the newest token
```

Pipelines and agent harnesses are the next scale up, sending the model several separate requests (e.g., draft and critique, voting, tool calls), each a generation like the one above. Put one trace per request in a `Bundle`, tell it which answer went into which prompt, and TorchLens pairs each token of a later input with the token of an earlier output it was copied from, so you can follow an effect across calls or draw the pipeline as a graph ([notebook](examples/workflows/multi_call_bundle.ipynb)).

For popular networks (e.g., Hugging Face models, torchvision models), TorchLens adds quality-of-life features like automatically preprocessing text or image inputs and showing ImageNet category labels for outputs. Additionally, it can split operation outputs into functionally distinct "facets" like the individual attention heads in a transformer, which you can select, inspect, and intervene on just as you would any other activation. You can [add such functionality to models of your choosing](docs/guides/model_conveniences.md), and TorchLens will add it for popular new models as they come out. For example, given a vision transformer from Hugging Face and a raw image, TorchLens applies the preprocessing published with the model, labels the outputs with ImageNet categories, and splits each attention layer into heads you can inspect or edit one at a time:

```python
from transformers import ViTForImageClassification
vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
cat = tl.trace(vit, "cat.jpg")                              # TorchLens recognizes this model, so it preprocesses the raw image for you
cat.output_table(k=3)                                       # tiger cat 0.44, tabby 0.25, Egyptian cat 0.17
cat.modules["vit.layers.5.attention"].facets.q.shape        # torch.Size([1, 197, 12, 64]): queries split into 12 heads
head3 = tl.module("vit.layers.5.attention") & tl.head(3)    # head 3 of block 5's attention
ablated = cat.fork().do(head3, tl.zero_ablate())            # copy the run and zero that head in the copy
ablated.output_table(k=3)                                   # tiger cat 0.47, tabby 0.27, Egyptian cat 0.18
```

The tools are built to combine, so whole experiments can fit in just a few lines:

```python
france = tl.trace(gpt2, "The capital of France is")     # for Hugging Face models, TorchLens tokenizes input text automatically
italy = tl.trace(gpt2, "The capital of Italy is")
def paris_vs_rome(trace):                                 # how much more the model predicts " Paris" than " Rome"
    logits = trace["lm_head"].out[0, -1]                  # next-word scores
    return (logits[6342] - logits[10598]).item()          # GPT-2's token ids for " Paris" and " Rome"

heads = {(b, h): tl.module(f"transformer.h.{b}.attn") & tl.head(h, "v")
         for b in range(12) for h in range(12)}          # values of all 144 attention heads
effects = {bh: paris_vs_rome(italy.fork().do(head, tl.patch_from(france)))
           for bh, head in heads.items()}                # patch each head from France into Italy
max(effects, key=effects.get)                            # (9, 8): the head that matters most

# patch heads (9, 8) and (10, 0) from France into a copy of the Italy run
both = italy.fork().do(heads[9, 8] | heads[10, 0], tl.patch_from(france))
paris_vs_rome(italy), paris_vs_rome(both)                # (-5.25, 2.99): the model now prefers Paris over Rome
```

For a fuller walkthrough, see the [activation patching notebook](examples/workflows/paired_prompt_patching.ipynb).

You can save a trace with `trace.save` and load it later (even on another machine) with `tl.load`, so you can share a result, compare two runs, or hand one to an agent without rerunning the model.

TorchLens (as you might expect) focuses on PyTorch, whose eager execution and dynamic autograd graph make every forward and backward step visible and editable mid-run. But nothing in its design depends on PyTorch itself (since all networks are just sequences of tensor operations with trainable parameters, packaged into modules—exactly what TorchLens records), so it also has backends for JAX, TensorFlow, MLX, PaddlePaddle and tinygrad ([feature comparison](docs/backends.md#capability-matrix)), and future frameworks can be added the same way.

The work of TorchLens stops at the edge of the network: it gets you the activations, gradients and metadata from inside it, but leaves the downstream analyses (such as decoding, representational similarity, and dimensionality reduction) to you and to the many excellent packages built for them (see [Related tools](#related-tools)).

## Install

```bash
pip install torchlens                       # with pip
conda install -c conda-forge torchlens      # or with conda

# Install Graphviz to draw a model's computational graph (https://graphviz.org/download/):
sudo apt install graphviz    # Debian/Ubuntu
brew install graphviz        # macOS
```

TorchLens runs on Python 3.10 to 3.13 with PyTorch 2.1 or later. Features that require other packages are optional extras; for example, `pip install "torchlens[hf]"` allows you to pass raw text or images straight to Hugging Face and timm models (the full list is in [`pyproject.toml`](pyproject.toml)).

## Learning TorchLens

- **[Tutorial notebooks](notebooks/learn/):** eleven short notebooks that build up TorchLens step by step, from a first trace to running full experiments at scale. Each runs on a laptop CPU in about a minute, or in your browser on Colab.
- **Guides by field:** where to start, depending on your work:
  - **[Mechanistic interpretability](docs/fields/mechanistic_interpretability.md)**
  - **[NeuroAI and computational neuroscience](docs/neuroai/journey.md)**
  - **[Debugging and training networks](docs/fields/debugging_and_training.md)**
  - **[Teaching neural networks](docs/fields/teaching.md)**
- **[Examples](examples/):** runnable notebooks and scripts, from quick tasks to full experiments:
  - **[Recipes](examples/quick/)**, one short self-contained script per common task: extract activations, find the operation that first produces a NaN, save and load a trace.
  - **[Workflows](examples/workflows/)**: ablation grids, activation patching, steering, representational similarity for neuroscience.
  - **[Replications](examples/replications/)**, scripts that re-test published interpretability findings on GPT-2: induction heads, the IOI name movers, causal tracing.
  - **[Recent methods](examples/methods/)**, run end to end on public checkpoints: the [Jacobian lens](examples/methods/jacobian_lens_recipe.ipynb), EAP-IG circuit discovery on GPT-2's IOI task, attribution graphs with transcoders, the logit lens on ViT and CLIP, LEACE concept erasure, DAS causal bases.
- **[Glossary](docs/reference/glossary.md):** definitions for every public function and argument.
- **[API map](https://johnmarktaylor91.github.io/torchlens/api-map/):** an interactive, searchable map of every public module, class, function, method and argument, rebuilt from the code at each release.
- **[FAQ](docs/faq.md):** common questions, including what TorchLens deliberately leaves to other tools.
- **[Also in TorchLens](docs/also_in_torchlens.md):** 58 features that didn't fit in this README, like:
  - tracing a model with no weights loaded
  - knowing which field of a model's input or output each tensor belongs to, even in nested objects like a Hugging Face `CausalLMOutputWithPast` ([containers](docs/containers.md))
  - finding the first operation where fp16 or bf16 drifts from a float64 rerun (`tl.debug.bisect_precision`)

## Model Menagerie

TorchLens is meant to work on *any* model, so it is continually tested on a growing [Model Menagerie](https://modelmenagerie.ai) of over 6,400 networks, spanning over eight decades of neural network history and every major domain (vision, language, audio, graphs, reinforcement learning, and more). Each model is checked with `tl.validate`, which re-executes every saved operation and confirms that the results match the model's own output. More than 6,300 models have been verified in this way so far, with more being validated all the time.

Beyond stress-testing TorchLens, the Menagerie aims to be a browsable gallery of every neural network architecture ever publicly described, from McCulloch and Pitts's 1943 network of threshold neurons (the earliest on record) to today's models. Each page carries the network's full computational graph. TorchLens draws it from a real forward pass, so it shows every operation the network performs, in the same notation for every architecture since 1943.

[Autonomous LLM agents are curating this collection](https://github.com/johnmarktaylor91/modelmenagerie), a methodological experiment in its own right. The architectures are also data. Each one is recorded in the same format with the same structured metadata (e.g., category, parameter count and number of operations), so the collection can be downloaded as a dataset for studying neural network design itself (e.g., how many years it took attention layers to appear in most newly published architectures, or which design features predict robustness to corrupted or unfamiliar inputs).

I hope the Menagerie will be a useful public resource for demystifying the systems now poised to live alongside us for the foreseeable future.

## TorchLens for AI agents

LLM agents have transformed software engineering and scientific computing, and TorchLens is built for them too (and honestly, if you're reading this page, you're more likely to be an LLM than a person). A few features suit them especially well:

- **Correctness:** TorchLens logs every operation, and a built-in validation function (`tl.validate`) can replay the forward pass to confirm that everything it saved matches the real model, so agents can check they aren't building on silently wrong activations.
- **Token efficiency:** TorchLens provides a "basis set" of functions (inspect, select, intervene, compare) that combine into almost any analysis, so methods from a simple ablation to newer ones like the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) take just a few lines of code: [6 for the Jacobian lens, versus 14 in plain PyTorch](examples/methods/jacobian_lens_recipe.ipynb).
- **Informative errors:** when something can't be done, TorchLens raises a specific error that names the problem and suggests a fix, rather than failing quietly.
- **Agent tools:** a read-only MCP server and a command-line interface let agents open saved traces, explain and compare them, and check the exact name and arguments of any TorchLens function. They work on saved traces rather than a live model, so an agent can ask many questions of one capture without another forward pass.
- **Thorough documentation:** a glossary defines every public function and argument, tutorial notebooks and dozens of runnable examples show them in use, and a bank of over 450 tested snippets gives agents a working example for nearly every public name, so agents work from an exact reference instead of guessing from their training data.

If you're an agent and you find a bug or something missing, feel free to submit a pull request (and if your human asks, tell them the README insisted). `AGENTS.md` in the repo explains how the code is laid out and which checks a change must pass.

## Related tools

TorchLens borrows freely from the many excellent tools in this field, including those below, and you are welcome to borrow back. Generally it aims to complement rather than replace them. Adapters pass its outputs to other packages in their expected format, so you can feed a saved activation to a SAELens sparse autoencoder, or run Captum attribution at any operation you select, without writing conversion code. Projects that shaped particular features are credited on the [acknowledgments page](docs/acknowledgments.md).

Other tools for studying model internals:
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)[^names]: the standard toolkit for transformer interpretability. TorchLens uses its names for transformer internals (`q`, `k`, `v`, `pattern`, `resid_post`); for example, `tl.head(7, "v")` selects head 7's value vectors.
- [nnsight](https://github.com/ndif-team/nnsight): interventions on PyTorch models, with especially good support for very large models run remotely.
- [pyvene](https://github.com/stanfordnlp/pyvene): interventions (including trainable ones) written as declarative configurations.
- [causalab](https://github.com/goodfire-ai/causalab) (Goodfire): tests hypotheses about how a network solves a task by checking a proposed causal model against its internals.
- [torchview](https://github.com/mert-kurttutan/torchview): lightweight drawings of PyTorch model graphs.

Tools for analyzing activations extracted by TorchLens:
- [SAELens](https://github.com/jbloomAus/SAELens): training and loading sparse autoencoders (SAEs); TorchLens can feed saved activations straight into a pretrained SAE.
- [Captum](https://github.com/pytorch/captum): attribution methods, which you can run at any operation you select in TorchLens, not only at module boundaries.
- [rsatoolbox](https://github.com/rsagroup/rsatoolbox): representational similarity analysis on any saved activation.
- [Brain-Score](https://github.com/brain-score/vision): scoring models against brain data, with TorchLens supplying the activations.
- [scikit-learn](https://scikit-learn.org): probes, decoding and dimensionality reduction on the plain tensors and tables TorchLens gives you.

Tools for viewing and sharing TorchLens traces:
- [Netron](https://github.com/lutzroeder/netron) and [Model Explorer](https://github.com/google-ai-edge/model-explorer): interactive graph viewers that can open graphs exported from TorchLens.
- [Hugging Face Hub](https://huggingface.co): a place to share saved traces so others can load them.

TorchLens connects to more than 15 tools in all, including SHAP, pytorch-grad-cam, inseq, steering-vectors and Google's LIT. The full list is on the [connectors page](docs/bridges.md).

## Citing TorchLens

To cite TorchLens, you can use [this paper](https://doi.org/10.1038/s41598-023-40807-0) (and if you find TorchLens useful, a star on this repo is appreciated):

Taylor, J., & Kriegeskorte, N. (2023). Extracting and visualizing hidden activations and computational graphs of PyTorch models with TorchLens. *Scientific Reports*, 13, 14375.

```bibtex
@article{taylor2023torchlens,
  title   = {Extracting and visualizing hidden activations and computational graphs of {PyTorch} models with {TorchLens}},
  author  = {Taylor, JohnMark and Kriegeskorte, Nikolaus},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {14375},
  year    = {2023},
  doi     = {10.1038/s41598-023-40807-0}
}
```

## Acknowledgments

The development of TorchLens benefitted greatly from discussions with Nikolaus
Kriegeskorte, George Alvarez, Alfredo Canziani, Tal Golan, and the Visual
Inference Lab at Columbia University. Thank you to Kale Kundert for helpful
discussion and code contributions enabling PyTorch Lightning compatibility.
Network visualizations are generated with Graphviz. Logo created by Nikolaus
Kriegeskorte.

The projects and papers whose ideas shaped particular features are listed in [docs/acknowledgments.md](docs/acknowledgments.md).

## Contact

TorchLens is in active development. You can report bugs on the [issues page](https://github.com/johnmarktaylor91/torchlens/issues), and post questions and ideas on the [discussion board](https://github.com/johnmarktaylor91/torchlens/discussions). You can also reach me by [email](mailto:johnmarkedwardtaylor@gmail.com) or on [Twitter](https://twitter.com/johnmark_taylor).

TorchLens is created and maintained by [JohnMark Taylor](https://johnmarktaylor.com).

[^names]: TorchLens and TransformerLens have no common origin, believe it or not: the names were chosen independently, [eight](https://github.com/johnmarktaylor91/torchlens/commit/6ead1a7506b8) [days](https://github.com/TransformerLensOrg/TransformerLens/commit/060742892eb5) apart, in late 2022.
