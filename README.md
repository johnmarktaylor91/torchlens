# <img src="images/logo.png" width=8% height=8%> TorchLens

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Quick Links**

- [Paper introducing TorchLens](https://www.nature.com/articles/s41598-023-40807-0)
- [CoLab tutorial](https://colab.research.google.com/drive/1ORJLGZPifvdsVPFqq1LYT3t5hV560SoW?usp=sharing)
- [Facets tutorial notebook](notebooks/facets_tutorial.ipynb)
- [5-minute notebook gallery](examples/5min/README.md)
- [50-minute workflow notebook gallery](examples/50min/README.md)
- [Performance guide](docs/performance.md) and [AI-agent quick reference](docs/for-ai-agents.md)
- [\"Menagerie\" of model visualizations](https://drive.google.com/drive/u/0/folders/1BsM6WPf3eB79-CRNgZejMxjg38rN6VCb)
- [Metadata provided by TorchLens](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-023-40807-0/MediaObjects/41598_2023_40807_MOESM1_ESM.pdf)
- License: [Apache 2.0](LICENSE); functional docs: [limitations](LIMITATIONS.md),
  [roadmap](ROADMAP.md), and [migration tables](docs/migration/).

## Overview

*TorchLens* is a package for doing exactly two things:

1) Easily extracting the activations from every single intermediate operation in a PyTorch model, with no
   modifications needed, in one line of code. "Every operation" means every operation; "one line" means one line.
2) Understanding the model's computational structure via an intuitive automatic visualization and extensive
   metadata ([partial list here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-023-40807-0/MediaObjects/41598_2023_40807_MOESM1_ESM.pdf))
   about the network's computational graph.

Backend status: PyTorch is the full-feature backend and the default
(`tl.trace(..., backend=None)`). TorchLens 2.x also ships **technical-preview
backends for MLX, JAX, tinygrad, Paddle, and TensorFlow** behind one `CaptureBackend`
protocol - select with `backend="jax"` / `backend="tinygrad"` /
`backend="mlx"` / `backend="paddle"` / `backend="tf"`. Unsupported or mismatched selections
raise typed backend errors, and `tl.validate(..., backend=...)` follows the
same resolution path. See
[What's new in 2.x](#whats-new-in-2x) for the capability matrix.

Here it is in action for a very simple recurrent model; as you can see, you just define the model like normal and pass
it in, and *TorchLens* returns a full log of the forward pass along with a visualization:

```python
class SimpleRecurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        for r in range(4):
            x = self.fc(x)
            x = x + 1
            x = x * 2
        return x


simple_recurrent = SimpleRecurrent()
model_history = tl.log_forward_pass(simple_recurrent, x,
                                    layers_to_save='all',
                                    vis_mode='rolled')
print(model_history['linear_1_1:2'].activation)  # second pass of first linear layer

'''
tensor([[-0.0690, -1.3957, -0.3231, -0.1980,  0.7197],
        [-0.1083, -1.5051, -0.2570, -0.2024,  0.8248],
        [ 0.1031, -1.4315, -0.5999, -0.4017,  0.7580],
        [-0.0396, -1.3813, -0.3523, -0.2008,  0.6654],
        [ 0.0980, -1.4073, -0.5934, -0.3866,  0.7371],
        [-0.1106, -1.2909, -0.3393, -0.2439,  0.7345]])
'''
```

<img src="images/simple_recurrent.png" width=30% height=30%>

And here it is for a very complex transformer model ([swin_v2_b](https://arxiv.org/abs/2103.14030)) with 1932 operations
in its forward pass; you can grab the saved outputs of every last one:

<img src="images/swin_v2_b_demo.jpg" width="70%" height="70%">

The goal of *TorchLens* is to do this for any PyTorch model whatsoever. You can see a bunch of example model
visualizations in this [model menagerie](https://drive.google.com/drive/u/0/folders/1BsM6WPf3eB79-CRNgZejMxjg38rN6VCb).

## What's new in 2.x

TorchLens has grown from "log every activation" into a configurable capture
substrate. The same one-line capture now drives selective saving, interventions,
backward-pass capture, and — new in 2.x — capture across multiple tensor
frameworks. Everything below is one `tl.trace(...)` call with different options;
there is a single capture path underneath, so you pay only for what you ask for.

**Multiple backends, one API.** Every interpretability tool in this space is
PyTorch-only. TorchLens captures eager-mode execution through a small backend
protocol, so the same trace/inspect/validate workflow now runs on other
frameworks:

| Capability | PyTorch | JAX (preview) | tinygrad (preview) | MLX (preview) | Paddle (preview) | TensorFlow (preview) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Forward capture + graph/metadata | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Module hierarchy | `torch_module` | Equinox/Flax NNX `pytree_module`; raw `function_root` | object `object_module`; raw `function_root` | object `object_module`; raw `function_root` | object `object_module`; raw `function_root` | Keras/`tf.Module` `object_module`; raw `function_root` |
| Control-flow unroll | eager Python | `lax.scan`/`cond`/`while_loop` | lazy UOp graph | limited | dygraph/eager Python only | eager Python control flow |
| Static-label `save=` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Portable array `.tlspec` payloads | full | forward/derived arrays | forward/derived arrays | forward/derived arrays | forward/derived arrays | forward arrays |
| Gradients | full backward graph | leaf-level + zero-tap T1 intermediate derived | leaf-level + T1 intermediate derived | leaf-level + custom-VJP-tap T1 intermediate derived | leaf-level + T1 intermediate derived | deferred |
| Interventions / halt / fastlog | ✅ | — | — | — | — | — |

The JAX backend is **jaxpr-first**: it captures the lowered jaxpr, so any
spelling (operators, array methods, `jnp`/`lax` calls) is captured by
construction, and captured ops are replay-validated in live traces. Equinox and
Flax NNX modules use `module_identity_mode="pytree_module"`, while raw functions
use `function_root`. The tinygrad backend captures UOp graphs, supports
callable-object `object_module` hierarchy, and exposes T1 intermediate derived
gradients through `trace.intermediate_derived_grads` / `op.derived_grad`.
The JAX backend exposes the same intermediate surface when
`tl.backends.jax.GradOptions(intermediate_grads=True, max_intermediate_grads=...)`
passes its zero-tap producer and per-boundary oracle.
MLX `mlx.nn.Module` roots use object-discovered `object_module` hierarchy, with
`function_root` available as an explicit root-only fallback, and expose leaf plus
custom-VJP-tap intermediate derived gradients through `tl.backends.mlx.GradOptions`.
Paddle is a dygraph/eager technical preview through `backend="paddle"` and
`tl.backends.paddle.GradOptions`; it rejects in-place mutation, RNG, user-level
tensor-to-Python scalar escapes, active stochastic/training composites, and true
backward capture. Same-object no-ops are captured as alias annotations rather
than value-copying ops. TensorFlow is a Keras-3 / TF>=2.16 technical preview
through `backend="tf"` or `backend="tensorflow"`. Its primary path is live eager
`op_callbacks` capture with real values, real taken-branch control flow, op-level
records, and Keras/`tf.Module` module stacks when `keras.backend.backend() ==
"tensorflow"`. Graph-only FuncGraph fallback is the static-mode design for compiled
or SavedModel-style entries; interventions, true backward capture, and T1 derived
gradients are deferred like the sibling preview gaps. JAX/tinygrad/MLX/Paddle/TF portable saves use
`payload_policy="array_payloads"` and loaded traces report replay as unavailable
rather than as a false pass. PyTorch remains the full-feature backend: true
backward capture, value-dependent predicates, `intervene=`, `halt=`, and fastlog
are torch-only for now. (Preview backends are pinned and documented in
[`docs/`](docs/); they are not yet drop-in for arbitrary models.)
When future importer-owned regions are not per-op replayable, replay status can be
`unverified`: the trace is materialized and replayable checks passed, but the status is
not a pass and raises on boolean coercion.

MLX preview traces support static-label `save=` for `tl.func`, `tl.label`,
`tl.module`, `tl.in_module`, `tl.contains`, and boolean composites of those.
Portable MLX saves round-trip saved forward and derived payloads as
`mlx.core.array` values when the MLX runtime is installed.
JAX portable saves round-trip typed PRNG keys and fully addressable single-host
sharded arrays by value. Their `codec_metadata` includes a reconstructible
JSON-primitive `jax_named_sharding` contract with mesh axis names, mesh shape,
and `PartitionSpec`; default load remains value-only, and callers must pass
`PayloadLoadHints(jax=JaxPayloadLoadHint(...))` to explicitly re-shard. The
`map_location` API remains single-device only. Multi-host or unaddressable
sharded arrays fail closed instead of silently losing topology.
The Op `__slots__` retained-memory baseline is tracked in
[`benchmarks/perf/slots_baseline.md`](benchmarks/perf/slots_baseline.md) and
shows roughly 10-15% lower retained trace memory on the measured fixtures.

```python
log = tl.trace(torch_model, x)                  # PyTorch (default)
log = tl.trace(jax_fn,  inputs, backend="jax")       # JAX preview
log = tl.trace(tg_fn,   inputs, backend="tinygrad")  # tinygrad preview
log = tl.trace(paddle_model, x, backend="paddle")     # Paddle preview
log = tl.trace(tf_model, x, backend="tf")             # TensorFlow preview
```

(`backend="tf"` / `backend="tensorflow"` capture eager TensorFlow forwards; JAX/tinygrad capture
pure functions; MLX and Paddle gradients run second guarded AD passes over module params and
selected inputs; TensorFlow gradients are deferred. Gradient options, audit caveats, and exact
calling conventions are documented per backend in
[`docs/`](docs/).)

**Backward-pass capture.** Capture the backward graph the same way you capture
the forward — per-op gradients, first-class `BackwardPass` records, and
higher-order support:

```python
x = torch.randn(2, 8, requires_grad=True)
log = tl.trace(model, x, save_grads=True)
log.log_backward(log[log.output_layers[0]].out.sum())
grad = log["relu_1_1"].grad        # gradient flowing through that op
```

**Interventions in one line.** Ablate, steer, patch, or swap activations during
the forward pass, then propagate the edit with replay or rerun (full API in
[Interventions](#interventions-v20) below):

```python
ablated = tl.trace(model, x, save=tl.func("relu"),
                   intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
```

**Pay only for what you use.** Selective, windowed, and early-halting capture
keep big models cheap — and `halt=` can finish *faster than a plain forward pass*
when you only need early layers:

```python
# capture conv2d only where it feeds a relu, keeping a 4-op lookback window
conv_before_relu = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
log = tl.trace(model, x, save=conv_before_relu,
               lookback=4, lookback_payload_policy="detached_raw")

# stop the forward pass as soon as the target is captured
log = tl.trace(model, x, save=tl.func("layer3"), halt=tl.func("layer3"))

# zero-copy reference saving for read-only inspection of large activations
log = tl.trace(model, x, save=tl.in_module("encoder"), save_mode="reference")
```

**Semantic access (facets).** Address named sub-views of an op's output — e.g.
attention `q`/`k`/`v` from a fused projection — for read, gradient, and
intervention, without hand-indexing tensors. See the facets tutorial in
[`notebooks/`](notebooks/).

## Compatibility report

Before filing a bug for a model-specific failure, run the runtime compatibility
report on the exact model wrapper and example input you plan to log:

```python
import torchlens as tl

compat = tl.compat.report(model, x)
print(compat.show())
print(compat.to_markdown())  # useful in issues and README snippets
```

`tl.compat.report(model, x)` does not execute the model. It inspects the model
wrapper, modules, parameter sharing, input tensors, CUDA visibility, and common
framework markers, then reports each row as `pass`, `known_broken`, `scope`, or
`not_tested`. Known-broken and out-of-scope rows are cross-referenced in
[`LIMITATIONS.md`](LIMITATIONS.md).

## Interventions (v2.0+)

TorchLens can also capture an intervention-ready log, mutate a fork of that log,
and propagate the edit with replay or rerun:

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)

clean = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
site = clean.find_sites(tl.func("relu")).first()

edited = clean.fork("zero_relu")
edited.attach_hooks(tl.label(site.layer_label), tl.zero_ablate())
edited.replay()

assert not torch.allclose(clean.layer_list[-1].activation, edited.layer_list[-1].activation)
```

Key intervention entry points:

- Select sites with `tl.label`, `tl.func`, `tl.module`, `tl.contains`, `tl.where`,
  and `tl.in_module`; selectors can be composed with `&` and `|` for discovery.
- Use helpers such as `tl.zero_ablate`, `tl.mean_ablate`, `tl.resample_ablate`,
  `tl.steer`, `tl.scale`, `tl.clamp`, `tl.noise`, `tl.project_onto`,
  `tl.project_off`, `tl.swap_with`, and `tl.splice_module`.
- Use `log.fork()` for branched experiments, then `log.set(...)`,
  `log.attach_hooks(...)`, `log.do(...)`, or top-level `tl.do(log, ...)`.
- Use `log.replay()` for graph-stable post-hoc edits and `log.rerun(model, x)`
  when the model should execute again.
- Raw PyTorch `nn.Module.register_forward_hook` replacements are also tolerated
  during active captures; replacement layer passes are marked with
  `intervention_replaced=True`.
- Compare related logs with `tl.bundle(...)`.
- Publish recipes with `log.save_intervention(path, level="portable")`; the
  saved `.tlspec/` directory contains JSON metadata plus tensor sidecars.

Worked examples live in [`examples/intervention/`](examples/intervention/README.md).
Additional docs cover [visibility classes](docs/visibility.md),
[intervention explainers](docs/intervention_explainers.md), the
[intervention API](docs/intervention_api.md), and cohort migration tables in
[`docs/migration/`](docs/migration/).

## Installation

To install *TorchLens*, first install graphviz if you haven't already (required to generate the network visualizations),
and then install *TorchLens* using pip:

```bash
sudo apt install graphviz
pip install torchlens
```

*TorchLens* is compatible with versions 1.8.0+ of PyTorch.

## How-To Guide

Below is a quick demo of how to use it; for an interactive demonstration, see
the [CoLab walkthrough](https://colab.research.google.com/drive/1ORJLGZPifvdsVPFqq1LYT3t5hV560SoW?usp=sharing).

The main function of *TorchLens* is `log_forward_pass`: when called on a model and input, it runs a
forward pass on the model and returns a ModelHistory object containing the intermediate layer activations and
accompanying metadata, along with a visual representation of every operation that occurred during the forward pass:

### Quick capture

For text models, TorchLens can now use duck-typed preprocessing methods attached to the model. Tensor inputs still
work exactly as before, but common interpreter workflows can pass raw text directly:

```python
# Before (still works)
import torchlens as tl
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
log = tl.log_forward_pass(model, tokenizer("Hello world", return_tensors="pt").input_ids)

# After (with model.tokenizer attached, or HookedTransformer)
model.tokenizer = tokenizer
log = tl.log_forward_pass(model, "Hello world")
```

```python
import torch
import torchvision
import torchlens as tl

alexnet = torchvision.models.alexnet()
x = torch.rand(1, 3, 224, 224)
model_history = tl.log_forward_pass(alexnet, x, layers_to_save='all', vis_mode='unrolled')
print(model_history)

'''
Log of AlexNet forward pass:
	Model structure: purely feedforward, without branching; 23 total modules.
	24 tensors (4.8 MB) computed in forward pass; 24 tensors (4.8 MB) saved.
	16 parameter operations (61100840 params total; 248.7 MB).
	Random seed: 3210097511
	Time elapsed: 0.288s
	Module Hierarchy:
		features:
		    features.0, features.1, features.2, features.3, features.4, features.5, features.6, features.7,
		    features.8, features.9, features.10, features.11, features.12
		avgpool
		classifier:
		    classifier.0, classifier.1, classifier.2, classifier.3, classifier.4, classifier.5, classifier.6
	Layers:
		0: input_1_0
		1: conv2d_1_1
		2: relu_1_2
		3: maxpool2d_1_3
		4: conv2d_2_4
		5: relu_2_5
		6: maxpool2d_2_6
		7: conv2d_3_7
		8: relu_3_8
		9: conv2d_4_9
		10: relu_4_10
		11: conv2d_5_11
		12: relu_5_12
		13: maxpool2d_3_13
		14: adaptiveavgpool2d_1_14
		15: flatten_1_15
		16: dropout_1_16
		17: linear_1_17
		18: relu_6_18
		19: dropout_2_19
		20: linear_2_20
		21: relu_7_21
		22: linear_3_22
		23: output_1_23
'''
```

<img src="images/alexnet.png" width=30% height=30%>

You can pull out information about a given layer, including its activations and helpful metadata, by indexing
the ModelHistory object in any of these equivalent ways:

1) the name of a layer (with the convention that 'conv2d_3_7' is the 3rd convolutional layer, and the 7th layer overall)
2) the name of a module (e.g., 'features' or 'classifier.3') for which that layer is an output, or
3) the ordinal position of the layer (e.g., 2 for the 2nd layer, -5 for the fifth-to-last; inputs and outputs count as
   layers here).

To quickly figure out these names, you can look at the graph visualization, or at the output of printing the
ModelHistory object (both shown above). Here are some examples of how to pull out information about a
particular layer, and also how to pull out the actual activations from that layer:

```python
print(model_history['conv2d_3_7'])  # pulling out layer by its name
# The following commented lines pull out the same layer:
# model_history['conv2d_3'] you can omit the second number (since strictly speaking it's redundant)
# model_history['conv2d_3_7:1'] colon indicates the pass of a layer (here just one)
# model_history['features.6'] can grab a layer by the module for which it is an output
# model_history[7] the 7th layer overall
# model_history[-17] the 17th-to-last layer
'''
Layer conv2d_3_7, operation 8/24:
	Output tensor: shape=(1, 384, 13, 13), dype=torch.float32, size=253.5 KB
		tensor([[ 0.0503, -0.1089, -0.1210, -0.1034, -0.1254],
        [ 0.0789, -0.0752, -0.0581, -0.0372, -0.0181],
        [ 0.0949, -0.0780, -0.0401, -0.0209, -0.0095],
        [ 0.0929, -0.0353, -0.0220, -0.0324, -0.0295],
        [ 0.1100, -0.0337, -0.0330, -0.0479, -0.0235]])...
	Params: Computed from params with shape (384,), (384, 192, 3, 3); 663936 params total (2.5 MB)
	Parent Layers: maxpool2d_2_6
	Child Layers: relu_3_8
	Function: conv2d (grad_fn=ConvolutionBackward0)
	Computed inside module: features.6
	Time elapsed:  5.670E-04s
	Output of modules: features.6
	Output of bottom-level module: features.6
	Lookup keys: -17, 7, conv2d_3_7, conv2d_3_7:1, features.6, features.6:1
'''

# You can pull out the actual output activations from a layer with the activation field:
print(model_history['conv2d_3_7'].activation)
'''
tensor([[[[-0.0867, -0.0787, -0.0817,  ..., -0.0820, -0.0655, -0.0195],
          [-0.1213, -0.1130, -0.1386,  ..., -0.1331, -0.1118, -0.0520],
          [-0.0959, -0.0973, -0.1078,  ..., -0.1103, -0.1091, -0.0760],
          ...,
          [-0.0906, -0.1146, -0.1308,  ..., -0.1076, -0.1129, -0.0689],
          [-0.1017, -0.1256, -0.1100,  ..., -0.1160, -0.1035, -0.0801],
          [-0.1006, -0.0941, -0.1204,  ..., -0.1146, -0.1065, -0.0631]]...
'''
```

If you do not wish to save the activations for all layers (e.g., to save memory), you can specify which layers to save
with the `layers_to_save` argument when calling `log_forward_pass`; you can either indicate layers in the same way
as indexing them above, or by passing in a desired substring for filtering the layers (e.g., 'conv'
will pull out all conv layers):

```python
# Pull out conv2d_3_7, the output of the 'features' module, the fifth-to-last layer, and all linear (i.e., fc) layers:
model_history = tl.log_forward_pass(alexnet, x, vis_mode='unrolled',
                                    layers_to_save=['conv2d_3_7', 'features', -5, 'linear'])
print(model_history.layer_labels)
'''
['conv2d_3_7', 'maxpool2d_3_13', 'linear_1_17', 'dropout_2_19', 'linear_2_20', 'linear_3_22']
'''
```

You can also keep raw activations while saving a transformed copy for analysis. For example, this stores each model
output in `activation` (also available as `tensor`) and stores a channel-averaged copy in `transformed_activation`:

```python
model_history = tl.log_forward_pass(
    alexnet,
    x,
    layers_to_save="all",
    activation_postfunc=lambda t: t.mean(dim=(2, 3)) if t.ndim == 4 else t,
)

layer = model_history["conv2d_3_7"]
print(layer.activation.shape)                # raw model output
print(layer.transformed_activation.shape)    # postfunc output
print(layer.tensor_shape)                    # metadata for the raw output
print(layer.transformed_activation_shape)    # metadata for the transformed output
```

By default TorchLens stores both tensors when `activation_postfunc` is set. To keep only the transformed tensor while
still retaining raw shape/dtype/memory metadata, pass `save_raw_activation=False`.

### Saving and Loading

```python
import torch
import torch.nn as nn
import torchlens as tl

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
x = torch.randn(2, 4)
model_log = tl.log_forward_pass(model, x, layers_to_save="all")
tl.save(model_log, "demo_bundle")

lazy_log = tl.load("demo_bundle", lazy=True)
activation = lazy_log["linear_1_1"].materialize_activation()
print(activation.shape)
```

You can also stream activations directly to disk during capture:

```python
streamed_log = tl.log_forward_pass(
    model,
    x,
    layers_to_save="all",
    save_activations_to="stream_bundle",
    keep_activations_in_memory=False,
)
```

## Fast activation recording (`tl.fastlog`)

Use `tl.fastlog` when you already know the events you want and do not need a full
`ModelLog`. `log_forward_pass()` remains the exhaustive path for graph metadata,
visualization, validation, and faithful reconstruction of the forward pass. Fastlog is
the lighter path for predicate-selected activations across one pass or many repeated
rollouts. Fastlog remains a PyTorch-only backend surface in the current backend
registry.

```python
keep_op = lambda ctx: ctx.kind == "op" and ctx.layer_type == "relu"
recording = tl.fastlog.record(model, x, keep_op=keep_op)
print(recording.summary())
```

Predicates receive `RecordContext` objects with operation/module fields and bounded
recent history, so they can express rules such as "ReLUs after convolutions" or "outputs
of every `Linear` module." Captures can stay in RAM, stream synchronously to a fastlog
directory bundle, or mirror to both. RAM mode is training-compatible via
`CaptureSpec(keep_grad=True)`.

The tutorial notebook at `notebooks/fastlog_tutorial.ipynb` walks through one-shot
recording, many-rollout `Recorder` sessions, disk load/recovery, graph previews,
`dry_run()` predicate iteration, downcasting with `CaptureSpec`, and the v1 support
boundaries.

## Training from saved activations

Use `train_mode=True` when a saved activation is part of a loss that will feed
`backward()`. TorchLens keeps saved floating tensors graph-connected in RAM, so you can
train with auxiliary losses, frozen-backbone probes, multi-tap losses, or
teacher-student activation distillation.

```python
model_log = tl.log_forward_pass(
    model,
    x,
    layers_to_save=["block1"],
    train_mode=True,
)
aux_loss = model_log["block1"].activation.square().mean()
aux_loss.backward()
```

The same knob is available on `ModelLog.save_new_activations()` for same-graph replay
and on `tl.fastlog.record()` / `tl.fastlog.Recorder()` for predicate-selected RAM
captures. Slow/replay training capture rejects disk activation saves because detached
disk payloads cannot carry autograd. For inspection persistence, capture in RAM first
and then call `tl.save(model_log, path)` after the training use. Existing code that
explicitly uses `detach_saved_tensors=False` continues to work, but `train_mode=True`
adds training guardrails: frozen parameter settings are preserved, disk/inference
misconfigurations fail early, and fastlog defaults can be promoted to keep-grad capture.

See `notebooks/training_tutorial.ipynb` for executable examples of the supported
training patterns and gotchas.

## Security

Portable bundles contain a pickle file in `metadata.pkl`. Only load bundles
from trusted sources. Loading an untrusted bundle with `tl.load()` or
`ModelLog.load()` can execute arbitrary code.

Export options:
- `to_pandas()` on model, layer, module, parameter, and buffer surfaces
- `to_csv(...)`, `to_parquet(...)`, and `to_json(...)` on those same surfaces
- `to_parquet(...)` requires `pyarrow`

Buffer support:
- Registered buffers are tracked as versioned model state, including recurrent
  reassignment, explicit in-place writes, and BatchNorm/InstanceNorm running
  statistics. See [docs/buffers.md](docs/buffers.md).

Caveats:
- Portable bundles do not support `validate_forward_pass()` or replay-oriented validation.
- Streaming capture is always strict; unsupported tensors abort the save instead of being skipped.
- Portable bundle save/load requires `safetensors`.

The main function of *TorchLens* is `log_forward_pass`; the remaining functions are:

1) `get_model_metadata`, to retrieve all model metadata without saving any activations (e.g., to figure out which
   layers you wish to save; note that this is the same as calling `log_forward_pass` with `layers_to_save=None`)
2) `show_model_graph`, which visualizes the model graph without saving any activations
3) `validate_model_activations`, which runs a procedure to check that the activations are correct: specifically,
   it runs a forward pass and saves all intermediate activations, re-runs the forward pass from each intermediate
   layer, and checks that the resulting output matches the ground-truth output. It also checks that swapping in
   random nonsense activations instead of the saved activations generates the wrong output. **If this function ever
   returns False (i.e., the saved activations are wrong), please contact me via email (johnmarkedwardtaylor@gmail.com)
   or on this GitHub page with a description of the problem, and I will update TorchLens to fix the problem.**

And that's it. *TorchLens* remains in active development, and the goal is for it to work with any PyTorch model
whatosever without exception. As of the time of this writing, it has been tested with over 700
image, video, auditory, multimodal, and language models, including feedforward, recurrent, transformer,
and graph neural networks.

## Miscellaneous Features

- You can visualize models at different levels of nesting depth using the `vis_nesting_depth` argument
  to `log_forward_pass`; for example, here you can see one of GoogLeNet's "inception" modules at different levels of
  nesting depth:

<img src="images/nested_modules_example.png" width=80% height=80%>

- An experimental feature is to extract not just the activations from all of a model's operations,
  but also the gradients from a backward pass (which you can compute based on any intermediate layer, not just the
  model's
  output),
  and also visualize the path taken by the backward pass (shown with blue arrows below). See the CoLab tutorial for
  instructions on how to do this.

<img src="images/gradients.png" width=30% height=30%>

- You can see the literal code that was used to run the model with the func_call_stack field.
  Each entry is a `FuncCallLocation` object with a clean repr and source context:

```python
print(model_history['conv2d_3'].func_call_stack[0])
'''
FuncCallLocation:
  file: /usr/local/lib/python3.10/dist-packages/torchvision/models/alexnet.py
  line: 48
  function: forward
  code:
          x = self.features(x)
          x = self.avgpool(x)
    --->  x = self.classifier(x)
          return x
'''
```

## Known limitations / unsupported contexts

TorchLens is not compatible with `torch.compile`'d models, TorchScript, or
`torch.export` (the forward pass doesn't run as ordinary Python, so our
wrappers never see the ops). It also has specific behaviors around FSDP,
sparse tensors, meta tensors, quantization, and `torch.func.vmap`. In
each case `log_forward_pass` either raises a clear error or emits a
targeted warning so you never get silent wrong results.

See **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** for the full matrix:
what fails, what works, and the recommended workaround for each context.

## Planned Features

1) In the further future, I am considering adding functionality to not just save activations,
   but counterfactually intervene on them (e.g., how would the output have changed if these parameters
   were different or if a different nonlinearity were used). Let me know if you'd find this useful
   and if so, what specific kind of functionality you'd want.
2) I am planning to add an option to only visualize a single submodule of a model rather than the full graph at once.

## Other Packages You Should Check Out

The goal is for *TorchLens* to completely solve the problem of extracting activations and metadata
from deep neural networks and visualizing their structure so that nobody has to think about this stuff ever again, but
it intentionally leaves out certain functionality: for example, it has no functions for loading models or stimuli, or
for analyzing the extracted activations. This is in part because it's impossible to predict all the things you might
want to do with the activations, or all the possible models you might want to look at, but also because there are
already outstanding packages for doing these things. Here are a few-let me know if I've missed any!

- [Cerbrec](cerbrec.com): Program for interactively visualizing and debugging deep neural networks (uses TorchLens under
  the hood for extracting the graphs of PyTorch models!)
- [ThingsVision](https://github.com/ViCCo-Group/thingsvision): has excellent functionality for loading vision models,
  loading stimuli, and analyzing the extracted activations
- [Net2Brain](https://github.com/cvai-roig-lab/Net2Brain): similar excellent end-to-end functionality to ThingsVision,
  along with functionality for comparing extracted activations to neural data.
- [surgeon-pytorch](https://github.com/archinetai/surgeon-pytorch): easy-to-use functionality for extracting activations
  from models, along with functionality for training a model using loss functions based on intermediate layer
  activations
- [deepdive](https://github.com/ColinConwell/DeepDive): has outstanding functionality for loading and benchmarking
  many different models
- [torchvision feature_extraction module](https://pytorch.org/vision/stable/feature_extraction.html): can extract
  activations from models with static computational graphs
- [rsatoolbox3](https://github.com/rsagroup/rsatoolbox): total solution for performing representational similarity
  analysis on DNN activations and brain data

## Acknowledgments

The development of *TorchLens* benefitted greatly from discussions with Nikolaus Kriegeskorte, George Alvarez,
Alfredo Canziani, Tal Golan, and the Visual Inference Lab at Columbia University. Thank you to Kale Kundert
for helpful discussion and for his code contributions enabling PyTorch Lightning compatibility.
All network visualizations were created with graphviz. Logo created by Nikolaus Kriegeskorte.

## Citing Torchlens

To cite *TorchLens*, you can
cite [this paper describing the package](https://www.nature.com/articles/s41598-023-40807-0) (and consider adding a star
to this repo if you find *TorchLens* useful):

Taylor, J., Kriegeskorte, N. Extracting and visualizing hidden activations and computational graphs of PyTorch models
with *TorchLens*. Sci Rep 13, 14375 (2023). https://doi.org/10.1038/s41598-023-40807-0

## Contact

As *TorchLens* is still in active development, I would love your feedback. Please contact
johnmarkedwardtaylor@gmail.com,
contact me via [twitter](https://twitter.com/johnmark_taylor), or post on
the [issues](https://github.com/johnmarktaylor91/torchlens/issues)
or [discussion](https://github.com/johnmarktaylor91/torchlens/discussions) page for this GitHub
repository, if you have any questions, comments, or suggestions (or if you'd be interested in collaborating!).
