# <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/logo.png" width=8% height=8%> TorchLens

[![PyPI version](https://img.shields.io/pypi/v/torchlens.svg)](https://pypi.org/project/torchlens/)
[![Python versions](https://img.shields.io/pypi/pyversions/torchlens.svg)](https://pypi.org/project/torchlens/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Release](https://github.com/johnmarktaylor91/torchlens/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/johnmarktaylor91/torchlens/actions/workflows/release.yml)
[![Nightly](https://github.com/johnmarktaylor91/torchlens/actions/workflows/nightly.yml/badge.svg)](https://github.com/johnmarktaylor91/torchlens/actions/workflows/nightly.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**See, save, and steer any PyTorch model.** TorchLens captures every activation
and gradient -- across the forward and backward pass -- auto-visualizes the full
computational graph, exposes rich per-op metadata, and lets you intervene on the
network as it runs. Any architecture, even dynamic and recurrent ones.

> **[Explore the Model Menagerie](https://modelmenagerie.ai)** -- a live, browsable atlas of
> **thousands of cataloged neural-network architecture entries** (8,500+ catalog entries
> across ~3,600 architecture families, measured 2026-08-16) captured with TorchLens, from
> McCulloch & Pitts (1943) to today's frontier models. *(Early preview.)*

Run across **thousands of cataloged entries** in the Model Menagerie (image, video, audio,
multimodal, language; feedforward, recurrent, transformer, GNN, MoE, diffusion) --
**verified on thousands of models across all major architecture families**: each verified
capture is replayed op-by-op against its own forward pass, with metadata-invariant
tripwires over the graph, so faithful capture is **proven, not assumed**. TorchLens also
records **180+ metadata fields per operation**, and
**550+ fields in total** across every record type — operations, modules, parameters,
buffers, gradients, and the model itself.

```python
import torch, torchvision.models as models, torchlens as tl

model = models.resnet18(weights="IMAGENET1K_V1").eval()  # .eval(): pretrained models arrive
# in train mode, and a train-mode forward silently updates BatchNorm running statistics
x = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))

log = tl.trace(model, x)      # one call -- full graph + all activations
print(log.summary())          # module table, op count, FLOPs
print(log["conv2d_1_1"].out.shape)  # grab any activation by name ...
print(log["layer1.0"].out.shape)    # ... or by module path
print(log[7].func_name)             # ... or by ordinal
log.draw()                    # PDF of the computational graph
```

Every verb climbs the same three-rung **input ladder** ([full guide](docs/quickstart.md)):

```python
# API sketch; `model` is your model and `lm` any HuggingFace language model.
log = tl.trace(model, x)                             # best: your real input
log = tl.trace(model, input_size=(1, 3, 224, 224))   # your shape, random values; disclosed
log = tl.trace(model)                                # inferred shape + random values; disclosed, or a teach
log = tl.trace(lm, "The quick brown fox")            # HF models: a string is a real input
```

<img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/swin_v2_b_demo.jpg" width="70%" height="70%">

**Quick Links**

- [Paper](https://www.nature.com/articles/s41598-023-40807-0) |
  [10-minute tutorial notebook](notebooks/torchlens_in_10_minutes.ipynb) |
  [Facets tutorial](notebooks/facets_tutorial.ipynb) |
  [5-minute gallery](examples/5min/README.md) |
  [50-minute gallery](examples/50min/README.md)
- [Choosing a fast path](docs/guides/fast_paths.md) |
  [Performance guide](docs/performance.md) |
  [Receptive & projective fields](docs/receptive_projective_fields.md) |
  [AI-agent quick reference](docs/for-ai-agents.md) |
  [Limitations and remedies](docs/reference/limitations.md) |
  [Migration tables](docs/migration/)


## Validated across thousands of catalog entries

TorchLens is not just smoke-tested on example models. The Model Menagerie validation
campaign runs the same adversarial check across the cataloged architecture
entries (**8,500+ rows across ~3,600 families** in the menagerie catalog, measured
2026-08-16; the live site carries additional locally-validated entries): capture the model with TorchLens, forward-replay the
captured DAG, compare replayed outputs against the original forward pass, and
run metadata-invariant tripwires over the resulting graph. If replay or an
invariant fails, the capture is treated as genuinely wrong and caught
automatically, not waved through because the model "ran."

```text
model forward -> TorchLens capture -> DAG replay -> output parity + metadata invariants
```

Today the campaign has algorithmically verified **thousands of models across all major
architecture families**, and the count is climbing; the exact per-row verification status
lives in the menagerie project's append-only ledger rather than in this README, so the public
claim stays hedged until the campaign closes. Inside this repository, a coverage-chosen sample of
the hand-built classics (`tests/classics_corpus/`) runs the same capture, replay, and invariant
checks as part of the test suite. That is the wedge:
TorchLens aims for captures that are **provably faithful, not just plausible**.
Plain forward hooks and static extraction utilities can be fast and useful, but
they can also silently miss dynamic paths, reused modules, functional ops,
fused attention, recurrent unrolls, or metadata needed to tell two sites apart.

The residual is tracked openly: the remaining tail is mostly genuinely huge
models, models with hard-to-build dependencies, or architectures that need more
bespoke inputs before an automated run is fair. They are counted as unverified,
not hidden as successes.


## Installation

Install Graphviz first (required for graph visualizations), then TorchLens:

```bash
sudo apt install graphviz   # Debian/Ubuntu; see graphviz.org for other platforms
pip install torchlens
```

Supported versions (each row below is executed by a CI leg on every release, not
asserted from metadata; the full platform statement is in
[docs/reference/limitations.md](docs/reference/limitations.md)):

| Component | Supported | How it is verified |
|---|---|---|
| Python | 3.10, 3.11, 3.12, 3.13 | one smoke-tier CI row per interpreter, plus clean-venv wheel installs at release |
| PyTorch | 2.1+ on Python 3.10/3.11; 2.2+ on 3.12; 2.6+ on 3.13 (the first torch with wheels for each interpreter) | floor rows pin the exact floor pair per interpreter; newest-admitted rows pin the current release; a daily canary runs latest |
| transformers (`torchlens[hf]`) | Policy: track the latest released transformers. 4.45 (the floor) and the current 4.x release install, import, pass the version-policy lockstep and every real-model R0 row that is not keyed to the 5.x band (the op-count goldens and four fixtures recorded on 5.x, including the T5 sdpa construction pin that only 5.18+ satisfies, are deselected on the 4.x legs until per-band expectations land); both the release-gating R0 deep sweep (`tests.yml`) and the nightly `hf-band-legs` candidate leg pin exactly the latest released 5.x (5.18.0), re-recorded on every release bump. The candidate leg additionally runs the offline, real-checkpoint R1 + 20-scenario RG workflow gallery against an exact passed-ID floor; its three remaining known limitations (typed, pinned refusals, not crashes) are enumerated in `tests/workflow_gallery/rg_enumerated_red_hf5.tsv` | floor / current / candidate constraint legs under `tests/support/proofnet/constraints/` (nightly `hf-band-legs`); release-gating pin in `.github/workflows/tests.yml` |
| Platforms | Linux/CPU is the tested platform; macOS and Windows run a nightly import + capture + save/load canary only | see the CI-attested platforms section of the limitations doc |

`pip install torchlens` installs the CPU-agnostic core (torch is resolved by pip for
your platform); optional integrations are extras, e.g. `pip install "torchlens[hf]"`.
Every declared extra is resolved on every supported interpreter by the nightly
metadata gate, and `torchlens[all]` means "everything that installs together" (the
excluded members are named in `pyproject.toml`).


## Quickstart

```python
import torch
import torchvision.models as models
import torchlens as tl

model = models.alexnet(weights=None)
x = torch.randn(1, 3, 224, 224)

log = tl.trace(model, x)
print(log.summary())
```

```
Model: AlexNet
+-----------------------------+---------------+--------+-------+
| Layer                       | Output Shape  | Params | Train |
+-----------------------------+---------------+--------+-------+
| input                       | [1,3,224,224] | 0      | -     |
| features (Sequential)       | [1,256,6,6]   | 2.5 M  | yes   |
| avgpool (AdaptiveAvgPool2d) | [1,256,6,6]   | 0      | -     |
| classifier (Sequential)     | [1,1000]      | 58.6 M | yes   |
| output                      | [1,1000]      | -      | -     |
+-----------------------------+---------------+--------+-------+
Params: 61,100,840 unique; trainable: 61,100,840
Ops: 22 total
Edges: 23 total
Forward FLOPs: 1.4 GFLOPs  MACs: 718.9 MFLOPs
```

Index any operation by name, module path, or ordinal:

```python
log['relu_1_2'].out.shape      # torch.Size([1, 64, 55, 55])
log['features.6'].out.shape    # same op via module path
log[7].func_name               # 'conv2d'
log['conv2d_3'].out.shape      # short name (ordinal suffix optional)
log[-1].layer_label            # 'output_1'
```

Visualize the graph as a PDF:

```python
log.draw()                        # unrolled by default
log.draw(vis_mode='rolled')       # rolled (compact for recurrent)
log.draw(vis_mode='unrolled')     # every pass as a distinct node
```

<img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/alexnet.png" width=30% height=30%>


## What You Can Do

### 1. Flexible feature extraction

Save everything, or select exactly what you need:

```python
# Save only relu activations
log = tl.trace(model, x, save=tl.func('relu'))

# Save all ops inside the 'classifier' submodule
log = tl.trace(model, x, save=tl.in_module('classifier'))

# Save conv2d ops that are immediately followed by a relu, keeping a 4-op lookback window
conv_before_relu = tl.func('conv2d') & tl.followed_by(tl.func('relu'))
log = tl.trace(model, x, save=conv_before_relu,
               lookback=4, lookback_payload_policy='detached_raw')

# Stop capture early (can be faster than a plain forward pass)
log = tl.trace(model, x, save=tl.in_module('features.6'), halt=tl.in_module('features.6'))

# Lightweight sparse recording for tight loops -- materialize structure later
recording = tl.record(model, x, save=tl.func('relu'))
trace = recording.to_trace()

# One-line activation pull
act = tl.pluck(model, x, 'relu_1_2')   # returns tensor directly

# Batch extraction across a dataset (any iterable of unbatched samples works)
dataset = [torch.randn(3, 224, 224) for _ in range(8)]
tl.extract_dataset(model, dataset, layers=['relu_1_2', 'conv2d_3_7'],
                   batch_size=32, output_dir='torchlens-activations/')
```

**Performance note:** With `halt=` and `tl.record`, capture can run *faster
than the raw forward pass* -- measured at 0.84x raw on ResNet-18 and 0.83x on
GPT-2 (HookedTransformer) at 25% depth. Full exhaustive capture runs at
roughly 14x the raw forward and amortizes on large models. See
[docs/performance.md](docs/performance.md) for the full benchmark table. To steer or patch a
model over many forwards or through `generate()`, skip capture entirely with
`tl.when(site, action).bind(model)`; see
[choosing a fast path](docs/guides/fast_paths.md).

Save and load traces portably:

```python
tl.save(log, 'my_trace')
loaded = tl.load('my_trace')
```

### 2. Forward AND backward pass

Capture per-op gradients with the same API:

```python
x = torch.randn(1, 3, 224, 224, requires_grad=True)
log = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
log.log_backward(log[log.output_layers[0]].out.sum())

grad = log['relu_1_2'].grad      # gradient tensor flowing through that op
print(grad.shape)                 # torch.Size([1, 64, 55, 55])
```

Narrow gradient saving to specific ops with the same selector predicates:

```python
log = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=tl.func('relu')))
log.log_backward(log[log.output_layers[0]].out.sum())
```

<img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/gradients.png" width=30% height=30%>

Backward capture is PyTorch-only. Non-torch backends expose derived leaf-level
gradients through a second AD pass. See [docs/backward.md](docs/backward.md).

### Influence geometry in both directions

**The first tool that gets ResNets right:** TorchLens computes receptive fields back toward an
input and projective fields forward toward an output over the captured DAG, then can cross-check
the geometric answer with an empirical gradient support mask. It includes ResNet skip connections
because it analyzes executed graph paths rather than multiplying a module list.

```python
target = log["features.6"]
rf = target.receptive_field
unit = rf.center_unit(batch_index=0)
box = rf.at((3, 3))
check = rf.check(unit)
# Projective geometry needs a windowed path; AlexNet's dense classifier head is not,
# so select a downstream conv endpoint explicitly with target=.
outgoing = target.projective_field.at((3, 3), target=log['features.8'])
```

<img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/receptive_projective_fields.svg" width="70%" alt="Receptive and projective field directions through a neural-network graph">

Every geometric claim is verification-backed, not asserted: `rf.check(unit)` cross-checks
one unit against an empirical gradient support mask, and
`tl.validate(model, x, scope="receptive_field")` runs the armed gradient tripwire over a
whole model with a PASS / FAIL / INDETERMINATE verdict. No other extraction tool ships a
receptive-field subsystem with this verification loop.

See the [receptive and projective fields guide](docs/receptive_projective_fields.md) for the
status contract, visual overlays, validation, and layer-to-layer queries.

### 3. Vast metadata per operation

Every operation records shape, dtype, device, timing, FLOPs, parameter info,
module containment, graph distances, conditional context, RNG state, and more.
The full print of any op includes all of this:

```python
print(log['conv2d_3_7'])
```

```
Layer conv2d_3_7, operation 7/22:
    Output tensor: shape=(1, 384, 13, 13), dtype=torch.float32, size=253.5 KB
        tensor([[-0.0198,  0.0946,  0.1109, ...
    Related Layers:
        - parent layers: maxpool2d_2_6
        - child layers: relu_3_8
    Params: Computed from params with shape (384, 192, 3, 3), (384,); 663936 params total (2.5 MB)
    Function: conv2d (grad_fn_handle: ConvolutionBackward0)
    Computed inside module: features.6:1
    Config: out_channels=384, in_channels=192, kernel_size=(3, 3), padding=(1, 1)
    Time elapsed: 1.4 ms
    Lookup keys: -17, 7, conv2d_3, conv2d_3:1, conv2d_3_7, conv2d_3_7:1, features.6, features.6:1
```

Every op also records the Python call stack that produced it, with file and
line number:

```python
loc = log['conv2d_3_7'].code_context[0]
print(loc.file, loc.line_number, loc.func_name)
```

Metadata is available as pandas DataFrames:

```python
df = log.to_pandas()            # one row per op
params_df = log.params.to_pandas()
modules_df = log.modules.to_pandas()
```

### 4. Automatic visualization

```python
log.draw()                           # default: unrolled with sibling ordering
log.draw(vis_mode='rolled')          # compact rolled layout
log.draw(vis_mode='unrolled')        # every pass as a distinct node
```

Control nesting depth to zoom in on submodules:

<img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/nested_modules_example.png" width=80% height=80%>

For recurrent models, the rolled view collapses repeated structure cleanly:

<img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/simple_recurrent.png" width=30% height=30%>

```python
class SimpleRecurrent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=5, out_features=5)
    def forward(self, x):
        for r in range(4):
            x = self.fc(x)
            x = x + 1
            x = x * 2
        return x

recurrent_model = SimpleRecurrent()
seq = torch.randn(6, 5)
recurrent_log = tl.trace(recurrent_model, seq)
print(recurrent_log['linear_1:2'].out)     # second pass of the linear layer
recurrent_log.draw(vis_mode='rolled')
```

And the HTML export is already **interactive** -- one self-contained file with
pan, zoom, and node-hover metadata, no server and no extra dependency:

```python
from pathlib import Path
viewer = tl.export.html(recurrent_log, Path('trace.html'))   # open in any browser
```

See [docs/reference/export.md](docs/reference/export.md) for the SVG, Netron,
Model Explorer, and profiler export family.

### 5. Interventions

Ablate, steer, scale, or replace activations during the forward pass:

```python
# Zero-ablate all relu activations inline during capture
ablated = tl.trace(model, x, save=tl.func('relu'),
                   intervene=tl.when(tl.func('relu'), tl.zero_ablate()))
print(ablated['relu_1_2'].out.abs().max())  # tensor(0.)

# Scale relus to 50%
scaled = tl.trace(model, x, save=tl.func('relu'),
                  intervene=tl.when(tl.func('relu'), tl.scale(0.5)))
```

Available helpers: `tl.zero_ablate`, `tl.mean_ablate`,
`torchlens.intervention.scramble_elements`, `tl.steer`, `tl.scale`, `tl.clamp`,
`tl.noise`, `tl.project_onto`, `tl.project_off`, `tl.swap_with`, `tl.splice_module`.

For post-hoc DAG replay and isolated experiments, capture with
`intervention_ready=True` and use `log.fork()` + `log.push()` /
`log.run(model, x)`. Live hooks during rerun require capture-time selectors
(e.g. `tl.func(...)`, `tl.module(...)`); finalized labels resolve via
`log.find_sites(...)`. See [docs/intervention_api.md](docs/intervention_api.md)
for the full reference.

Compare multiple runs side by side with `tl.bundle`:

```python
# Default save policy retains inputs, so the bundle can PROVE the two runs
# saw identical inputs before comparing (comparisons never assume it).
clean_log = tl.trace(model, x)
patched_log = tl.trace(model, x,
                       intervene=tl.when(tl.func('relu'), tl.zero_ablate()))
bundle = tl.bundle({'clean': clean_log, 'patched': patched_log}, baseline='clean')
bundle.compare_at('relu_1_2')   # one site; a multi-site selector must resolve uniquely
```

**Facets** provide named sub-views for attention heads, LSTM outputs, and
fused projections (for models with those structures):

The following is an API sketch; `vit_model` and `lstm` must be models whose module
structures provide the shown paths, and are not defined by this generic example.

```python
# API sketch; `vit_model` and `lstm` are application-supplied models with these paths.
# ViT / transformer model with attention blocks
log = tl.trace(vit_model, x)
q = log.modules['blocks.0.attn'].facets['q']    # query vectors for head 0
h_n = log.modules['lstm'].facets['h_n']         # LSTM final hidden state
```

See [docs/facets.md](docs/facets.md) for the full facets reference, including
activation patching helpers, SDPA reconstruction, and TransformerLens aliases.

See [docs/intervention_api.md](docs/intervention_api.md) for the full selector
and helper reference.

### 6. Works on anything, including dynamic and recurrent models

TorchLens uses eager-mode Python-level function wrapping rather than graph
tracing. This means it captures whatever actually runs, including:

- Dynamic control flow (if/else branching, loops, early exits)
- Recurrent architectures (RNNs, LSTMs, state-space models)
- Transformer variants including fused attention
- Graph neural networks
- Mixed architectures

This is the key differentiator from static-graph extractors like
`torchvision.feature_extraction`, which require static computational graphs
and cannot handle dynamic architectures.

**Distributed boundaries.** With `tl.distributed.arm()` enabled before rank-local
capture, explicit in-forward `torch.distributed` Python collectives become first-class
boundary nodes. Diagnose rank sets with `tl.merge_report(...)` and merge compatible
rank traces with `tl.merge_ranks(...)`. Sharded tensor topologies such as DTensor/FSDP/TP
and pipeline point-to-point graphs still refuse with typed findings; see the
[merged-trace contract](docs/reference/merged_trace_contract.md).

**Multi-backend.** The same `tl.trace` API works across frameworks via
`backend=`:

| Capability | PyTorch | JAX (preview) | tinygrad (preview) | MLX (preview) | Paddle (preview) | TensorFlow (preview) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Forward capture + graph/metadata | yes | yes | yes | yes | yes | yes |
| Module hierarchy | `torch_module` | Equinox/Flax NNX `pytree_module`; raw `function_root` | `object_module`; raw `function_root` | `object_module`; raw `function_root` | `object_module`; raw `function_root` | Keras/`tf.Module` `object_module`; raw `function_root` |
| Control-flow unroll | eager Python | `lax.scan`/`cond`/`while_loop` | lazy UOp graph | limited | dygraph/eager Python only | eager Python control flow |
| Static-label `save=` | yes | yes | yes | yes | yes | yes |
| Portable array `.tlspec` payloads | full | forward/derived arrays | forward/derived arrays | forward/derived arrays | forward/derived arrays | forward arrays |
| Gradients | full backward graph | leaf-level + zero-tap T1 intermediate derived | leaf-level + T1 intermediate derived | leaf-level + custom-VJP-tap T1 intermediate derived | leaf-level + T1 intermediate derived | leaf + exact T1 intermediate derived (eager entries, `tl.backends.tf.GradOptions`) |
| Recurrence grouping (multi-pass layers) | yes | yes | yes (eager) | yes (eager) | yes (eager) | yes (eager; static FuncGraph path stays ungrouped) |
| Validation oracle (live) | whole-forward replay | per-equation replay + perturbation | per-UOp replay + perturbation | per-op replay + perturbation | replay + perturbation + coverage guard | per-op replay (allowlisted) + self-consistency |
| Interventions | yes | -- | -- | yes (+`halt=`) | yes (+`halt=`, value-dependent predicates) | yes (eager entries, static-label, fail-closed) |
| Halt / fastlog / streaming | yes | -- | -- | halt only | halt only | -- |

Preview `save=` selectors filter what is exposed, not what is captured (no memory
reduction); preview `tl.validate(...)` returns a status whose `bool()` raises for
partial coverage; and every preview auto-routes genuine framework models on
`backend=None`. See `docs/backends.md` for the per-backend contract.

```python
# API sketch; supply compatible models and input in an application context.
log = tl.trace(torch_model, x)                      # PyTorch (default)
log = tl.trace(jax_fn,      inputs, backend='jax')  # JAX preview
log = tl.trace(tg_fn,       inputs, backend='tinygrad')
log = tl.trace(paddle_model, x,     backend='paddle')
log = tl.trace(tf_model,    x,      backend='tf')
```

PyTorch remains the full-feature backend. Preview backends are pinned and
documented in [`docs/`](docs/).

### 7. Profile it, and stream statistics over whole datasets

Every trace already carries per-op timings, FLOPs, and memory; `profile()`
presents them per op, module, or call, sorted and truncated how you like, and
`honesty()` names the clock and instrumentation state behind every number
instead of just saying "measured":

```python
prof_log = tl.trace(model, torch.randn(1, 3, 64, 64))
print(prof_log.profile(level='module', sort_by='flops', top_k=8))
print(prof_log.profile().honesty())

# Nsight-ready: wrap each captured op in an identity-carrying NVTX range
nvtx_log = tl.trace(model, torch.randn(1, 3, 64, 64),
                    capture=tl.options.CaptureOptions(emit_nvtx=True))
```

Dataset-level statistics stream in constant memory -- including *exact*
full-data linear CKA, not a minibatch approximation:

```python
stats = tl.aggregate(
    model,
    [torch.rand(1, 3, 64, 64) for _ in range(2)],   # any iterable of inputs
    metrics={'relu_1_2': tl.stats.Aggregator(tl.stats.Mean(), tl.stats.Norm()),
             'output': tl.stats.Quantile()},
)
print(stats['relu_1_2']['Mean'], sorted(stats['output']))
```

For training loops, build the sparse recorder once and log whichever steps
you like; discovery helpers list what there is to ask for:

```python
from torchlens.fastlog import Recorder
from torchlens.utils import list_modules, list_ops

with Recorder(model, save=tl.func('relu')) as rec:
    for step in range(2):                    # your optimizer loop
        rec.log(torch.rand(1, 3, 64, 64))    # sparse capture of this step

print(list_modules(model)[:3])               # every module address + class
print(list_ops(model, torch.randn(1, 3, 64, 64))[:3])   # op counts per forward
```

FLOP totals follow a declared convention (fma=2 by default; `flop_convention="fma1"`
recounts under fma=1 where a MAC split is derivable, and refuses typed where it is
not), and custom ops get first-class cost rules:

```python
# API sketch; register in your application before tracing (process-global registry).
from torchlens.capture.flops import register_op_rule
register_op_rule('my_custom_op', lambda output_shape, param_shapes, args, kwargs: 0)
```

See [docs/reference/stats.md](docs/reference/stats.md) for the full streaming-stats
surface, [docs/native-torch.md](docs/native-torch.md) for what native torch tooling is
authoritative for (and the exact recipes we point at), and
[docs/migration/from_hooks.md](docs/migration/from_hooks.md) for an honest two-way
comparison with forward hooks.


## Gallery

TorchLens visualizes any architecture -- no matter how exotic. Explore the
**[Model Menagerie](https://modelmenagerie.ai)**: a browsable atlas of **thousands of cataloged
neural-network architecture entries** -- from McCulloch & Pitts (1943) to today's frontier
models -- each with structured metadata and a TorchLens-rendered diagram. Thousands of them
currently carry the replay-and-invariant verification described above.

> **Early preview.** The gallery is live and growing; full-text search, a downloadable dataset, and
> richer per-model pages are on the way.

A sample across families is shown below.

**Classic CNN + Vision Transformer**

| GoogLeNet (inception + buffer edges) | Stable Diffusion (U-Net denoiser) | CLIP (vision + language towers) |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/googlenet.jpg" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/stable_diffusion.png" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/clip.jpg" height="200"> |

**State-Space + Recurrence**

| Mamba (selective SSM) | Recurrent Gemma (linear recurrence) | Whisper (audio encoder-decoder) |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/mamba.jpg" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/recurrent_gemma.jpg" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/whisper.jpg" height="200"> |

**Mixture-of-Experts + Generative**

| Mixtral (sparse MoE) | Hierarchical VAE | Perceiver |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/mixtral.jpg" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/hierarchical_vae.png" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/perceiver.jpg" height="200"> |

**Graph Networks + Exotic**

| DimeNet (molecular GNN) | CORnet-S (visual cortex, unrolled) | LLaMA (decoder-only LLM) |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/dimenet.png" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/cornet_s.png" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/llama.jpg" height="200"> |

**Reinforcement Learning + Quantum ML + Scale**

| Decision Transformer (offline RL) | Quantum ML circuit | 3,000-node graph (SFDP layout) |
|:---:|:---:|:---:|
| <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/decision_transformer.jpg" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/qml.png" height="200"> | <img src="https://raw.githubusercontent.com/johnmarktaylor91/torchlens/main/images/menagerie/large_graph_3k.jpg" height="200"> |


## Pin your architecture in CI

Use the provisional address-free structural hash to catch an unintended graph change without
pinning model weights or module names:

```python
# `model` and `x` are your model and a representative example input.
import torchlens as tl

pinned = tl.assert_unchanged(model, x, expected=None)  # prints and returns a hash
tl.assert_unchanged(model, x, pinned)  # raises if the architecture changes
```

See [`tl.hash`](docs/reference/hash.md) for trace-level hashing and its structural scope.

## Compatibility

Before filing a bug for a model-specific failure, run the runtime compatibility
report:

```python
compat = tl.compat.report(model, x)
print(compat.to_markdown())
```

`tl.compat.report` inspects the model wrapper, modules, parameter sharing,
input tensors, CUDA visibility, and common framework markers, then reports
each row as `pass`, `known_broken`, `scope`, or `not_tested`.

`torch.compile` coexists with capture. On torch >= 2.6, every capture holds the
public `torch.compiler.set_stance("force_eager")` scoped to the forward, so
compiled regions run their original eager Python: the interior is fully logged
with full verified semantics, zero graph breaks or new compiles happen during
capture, compiled caches stay intact, and wrapper install/uninstall costs at
most one bounded recompile on the next compiled call. On torch < 2.6 the
historical fallback holds: a Dynamo-traced region reached mid-capture is
bypassed with a one-per-forward warning and the returned trace honestly
contains only what ran outside it (`capture_verified=False`, reason
`"dynamo_region_not_logged"`). TorchLens remains **not** compatible with
TorchScript or `torch.export` -- those forwards do not run as ordinary Python,
so the wrappers cannot intercept ops. It also has specific behaviors around
FSDP, sparse tensors, meta tensors, quantization, and `torch.func.vmap`.

See [LIMITATIONS.md](docs/LIMITATIONS.md) for the full matrix: what fails, what
works, and the recommended workaround for each context.

TorchLens recovers most detached `from torch import ...` references with a disclosed rescue
re-run and a small mechanical belt. The historical broad `sys.modules` crawl and
`patch_policy=` rollout are deleted, along with the former no-op arguments. For the strongest
and simplest guarantee, call `torchlens.backends.torch.wrappers.wrap_torch()` before creating
detached references. The optional `escape_detector="shadow"` diagnoses raw callable escapes. See
[detached-reference handling](docs/migration/scoped_detached_patching.md) and the
[limitations catalog](docs/reference/limitations.md).

A traced model keeps TorchLens wrapper state attached afterwards, so pickling
the WHOLE module (`pickle.dumps(model)` or `torch.save(model)`) can fail with a
`PicklingError` after tracing. Call `tl.release_model(model)` to restore
whole-model serializability; `state_dict()` saves, traces, and saved
activations are unaffected either way.


## Tutorials and Docs

| Resource | Description |
|---|---|
| [torchlens_in_10_minutes.ipynb](notebooks/torchlens_in_10_minutes.ipynb) | Core workflow: trace, index, visualize |
| [facets_tutorial.ipynb](notebooks/facets_tutorial.ipynb) | Attention heads, LSTM facets, patching |
| [backward_tutorial.ipynb](notebooks/backward_tutorial.ipynb) | Gradient capture and backward visualization |
| [training_tutorial.ipynb](notebooks/training_tutorial.ipynb) | Training with captured activations |
| [huggingface_tutorial.ipynb](notebooks/huggingface_tutorial.ipynb) | HuggingFace transformer models |
| [fastlog_tutorial.ipynb](notebooks/fastlog_tutorial.ipynb) | High-throughput sparse recording |
| [docs/intervention_api.md](docs/intervention_api.md) | Full selector and helper reference |
| [docs/backward.md](docs/backward.md) | Backward capture details and limitations |
| [docs/facets.md](docs/facets.md) | Facets, patching, and SDPA reconstruction |
| [docs/guides/fast_paths.md](docs/guides/fast_paths.md) | Choosing a fast path: capture-free steering (`bind`), `tl.record` against `tl.trace`, cheaper traces, rerun limits |
| [docs/performance.md](docs/performance.md) | Speed knobs and benchmark numbers |
| [docs/reference/debug.md](docs/reference/debug.md) | Trace diagnostics: lineage, non-finites, costs, and gradients |
| [docs/reference/export.md](docs/reference/export.md) | Static, profiling, tabular, and tracker exports (incl. the interactive HTML viewer) |
| [docs/reference/stats.md](docs/reference/stats.md) | Streaming dataset statistics: `tl.stats` accumulators and `tl.aggregate` |
| [docs/native-torch.md](docs/native-torch.md) | What native torch tooling is authoritative for, with recipes |
| [docs/migration/from_hooks.md](docs/migration/from_hooks.md) | Honest two-way comparison with forward hooks |
| [docs/reference/hash.md](docs/reference/hash.md) | Provisional structural hashes and CI architecture pins |
| [docs/reference/attribution.md](docs/reference/attribution.md) | Native input and layer attribution methods |
| [docs/reference/collapse.md](docs/reference/collapse.md) | Smart-collapse visual reference and label contract |
| [docs/reference/glossary.md](docs/reference/glossary.md) | Public terminology and stable mechanism names |
| [docs/reference/limitations.md](docs/reference/limitations.md) | Edge scenarios, typed symptoms, and remedies |


## Security

Portable bundles contain a pickle file in `metadata.pkl`. It is read through a
restricted, default-deny unpickler, and foreign callables are never imported at
load time -- but executing what an artifact describes (running a runnable
artifact, or opting into `trust_custom_callables=`) is code execution, and
tracing a model runs its `forward()`. Only load bundles and trace models from
sources you trust. See [SECURITY.md](SECURITY.md) for the full trust model,
the supported-versions table, and dependency-advisory status (including the
transformers 4.x advisories and their remedy).


## Other Packages You Should Check Out

TorchLens focuses on activation extraction, graph visualization, and intervention
and intentionally omits model loading, stimulus management, and analysis pipelines.
These packages cover that ground well:

- [Cerbrec](https://cerbrec.com): interactive visualization and debugging for deep neural networks (uses TorchLens under the hood for PyTorch graph extraction)
- [ThingsVision](https://github.com/ViCCo-Group/thingsvision): model loading, stimulus management, and representational analysis for vision models
- [Net2Brain](https://github.com/cvai-roig-lab/Net2Brain): end-to-end pipeline for comparing DNN representations to neural data
- [surgeon-pytorch](https://github.com/archinetai/surgeon-pytorch): lightweight activation extraction with training-loss hooks
- [deepdive](https://github.com/ColinConwell/DeepDive): model loading and benchmarking across many model families
- [torchvision feature_extraction](https://pytorch.org/vision/stable/feature_extraction.html): fast activation extraction for models with static computational graphs
- [rsatoolbox](https://github.com/rsagroup/rsatoolbox): representational similarity analysis for DNN activations and brain data


## Acknowledgments

The development of TorchLens benefitted greatly from discussions with Nikolaus
Kriegeskorte, George Alvarez, Alfredo Canziani, Tal Golan, and the Visual
Inference Lab at Columbia University. Thank you to Kale Kundert for helpful
discussion and code contributions enabling PyTorch Lightning compatibility.
Network visualizations are generated with Graphviz. Logo created by Nikolaus
Kriegeskorte.

TorchLens borrows conventions and ideas, with credit, from many tools its
users already know. The full credit roster -- every project and paper whose
ideas shaped a shipped feature -- is in
[docs/acknowledgments.md](docs/acknowledgments.md).


## Citing TorchLens

To cite TorchLens, please cite
[this paper](https://www.nature.com/articles/s41598-023-40807-0):

Taylor, J., Kriegeskorte, N. Extracting and visualizing hidden activations and
computational graphs of PyTorch models with TorchLens. *Sci Rep* 13, 14375
(2023). https://doi.org/10.1038/s41598-023-40807-0

If you find TorchLens useful, a star on this repo is appreciated.


## Contact

TorchLens is in active development. Questions, bug reports, and suggestions are
welcome via [email](mailto:johnmarkedwardtaylor@gmail.com),
[Twitter](https://twitter.com/johnmark_taylor), the
[issues page](https://github.com/johnmarktaylor91/torchlens/issues), or the
[discussion board](https://github.com/johnmarktaylor91/torchlens/discussions).
