# Namespaces: one example per `tl.<module>`

## Capturing and options

### namespaces-options
Group capture settings in a `CaptureOptions` and pass it as `capture=`.

- covers: `tl.options`, `tl.options.CaptureOptions`
- tasks: capture, options
- fixture: toy

```python
options = tl.options.CaptureOptions(save_grads=True)
trace = tl.trace(model, x, capture=options)
print(type(trace.options).__name__)
print(options.save_grads)
# Output:
# TraceSettings
# True
```

### namespaces-recording
Dry-run a save predicate to see which ops it would keep before recording.

- covers: `tl.recording`, `tl.recording.dry_run`
- tasks: record, select-save
- fixture: toy

```python
plan = tl.recording.dry_run(model, x, save=tl.func("relu"))
print(type(plan).__name__, len(plan.contexts) > 0, plan.predicate_failures)
# Output:
# DryRun True ()
```

### namespaces-observers
Log a scalar from inside a forward; the trace keeps it.

- covers: `tl.observers`, `tl.observers.log_value`
- tasks: observe
- fixture: none

```python
class Net(torch.nn.Module):
    def forward(self, x):
        tl.observers.log_value("total", float(x.sum()))
        return x + 1
trace = tl.trace(Net(), torch.ones(2, 3))
print(trace.logged_values)
# Output:
# {'total': 6.0}
```

### namespaces-experimental
Stop a capture early, after a named module has run.

- covers: `tl.experimental`, `tl.experimental.stop_after`
- tasks: capture, stop
- fixture: cnn

```python
with tl.experimental.stop_after("1"):
    early = tl.trace(model, x)
print(early.op_labels)
print(tl.trace(model, x).op_labels)
# Output:
# ['input_1:1', 'conv2d_1_1:1', 'relu_1_2:1', 'output_1:1']
# ['input_1:1', 'conv2d_1_1:1', 'relu_1_2:1', 'conv2d_2_3:1', 'output_1:1']
```

### namespaces-distributed
Distributed capture is opt-in; by default it is not armed.

- covers: `tl.distributed`, `tl.distributed.is_armed`
- tasks: distributed
- fixture: none

```python
print(tl.distributed.is_armed())
# Output:
# False
```

### namespaces-backends
Look up the registered capture backends and the torch backend's spec.

- covers: `tl.backends`, `tl.backends.specs`, `tl.backends.spec`
- tasks: backends
- fixture: none

```python
names = [s.name for s in tl.backends.specs()]
print("torch" in names)
print(tl.backends.spec("torch").protocol().materialization)
# Output:
# True
# eager
```

## Interventions and comparisons

### namespaces-intervention
Check that a saved intervention recipe resolves to the same sites on a trace.

- covers: `tl.intervention`, `tl.intervention.check_spec_compat`, `tl.io.save_intervention`
- tasks: intervene, save
- fixture: toy

```python
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
tl.io.save_intervention(edited, "spec.tlspec", level="audit")
compat = tl.intervention.check_spec_compat(tl.load("spec.tlspec"), edited)
print(compat.has_identical_target_resolution)
# Output:
# True
```

### namespaces-bundle
A Bundle groups traces; `.at(selector)` gives one SuperOp across members.

- covers: `tl.bundle`, `tl.bundle.SuperOp`
- tasks: compare, bundle
- fixture: toy

```python
bundle = tl.Bundle({"a": tl.trace(model, x), "b": tl.trace(model, x * 2)})
site = bundle.at(tl.func("relu"))
print(isinstance(site, tl.bundle.SuperOp), sorted(site.members))
# Output:
# True ['a', 'b']
```

### namespaces-differential
Diff two captures of the same model op by op.

- covers: `tl.differential`, `tl.differential.diff`
- tasks: compare, diff
- fixture: toy

```python
import torchlens.differential
subject = tl.trace(model, x + 1.0)
reference = tl.trace(model, x)
report = tl.differential.diff(subject, reference)
print(len(report.rows) > 0, report.counts.get("added", 0))
# Output:
# True 0
```

### namespaces-experiment
Sweep one edit over candidate sites and keep each run in a bundle.

- covers: `tl.experiment`, `tl.experiment.sweep_sites`
- tasks: experiment, ablate
- fixture: toy

```python
import torchlens.experiment
base = tl.trace(model, x)
bundle = tl.experiment.sweep_sites(
    base, candidates={"relu": "relu_1_2"}, edit=tl.zero_ablate(),
    metric=lambda t: float(t[t.output_layers[0]].out.sum()), retain="all",
    model=model, inputs=x)
print(bundle.names)
# Output:
# ['baseline', 'relu']
```

### namespaces-fingerprint
A structural hash of a model: the same for any input of the same shape family.

- covers: `tl.fingerprint`, `tl.fingerprint.model`, `tl.fingerprint.trace`
- tasks: fingerprint, compare
- fixture: toy

```python
pin = tl.fingerprint.model(model, x)
print(len(pin))
print(pin == tl.fingerprint.model(model, torch.randn(3, 4)))
print(pin == tl.fingerprint.trace(tl.trace(model, x)))
# Output:
# 64
# True
# True
```

## Saving, loading and exporting

### namespaces-io
Inspect a saved `.tlspec` folder's manifest without loading it.

- covers: `tl.io`, `tl.io.inspect_tlspec`
- tasks: save, load
- fixture: toy

```python
tl.trace(model, x).save("run.tlspec")
manifest = tl.io.inspect_tlspec("run.tlspec")
print(manifest["kind"])
# Output:
# trace
```

### namespaces-export
List the export targets and write a trace to CSV.

- covers: `tl.export`, `tl.export.list_targets`, `tl.export.csv`
- tasks: export
- fixture: toy

```python
print("csv" in tl.export.list_targets())
out = tl.export.csv(tl.trace(model, x), "model.csv")
print(out.read_text(encoding="utf-8").startswith("# torchlens capture honesty:"))
# Output:
# True
# True
```

### namespaces-examples
Load a small bundled example trace by name.

- covers: `tl.examples`, `tl.examples.load`
- tasks: examples
- fixture: none

```python
trace = tl.examples.load("tiny_linear")
print(type(trace).__name__)
# Output:
# Trace
```

### namespaces-dataset-extraction
Extract one site over a small dataset to disk, then read it back.

- covers: `tl.dataset_extraction`, `tl.dataset_extraction.open_extraction`
- tasks: extract-dataset
- fixture: cnn

```python
stimuli = [torch.randn(3, 8, 8) for _ in range(4)]
tl.dataset_extraction.extract_dataset(
    model, stimuli, {"relu": "1"}, batch_size=2, path="ext", progress=False
)
reader = tl.dataset_extraction.open_extraction("ext")
print(reader.keys, reader.n_stimuli, tuple(reader.row(0)["relu"].shape))
# Output:
# ['relu'] 4 (4, 8, 8)
```

## Reports and debugging

### namespaces-report
Run the environment doctor: versions and checks.

- covers: `tl.report`, `tl.report.doctor`
- tasks: health, doctor
- fixture: toy

```python
report = tl.report.doctor(model, x)
print(type(report).__name__)
print(report.torch_version == torch.__version__, len(report.checks) > 0)
# Output:
# DoctorReport
# True True
```

### namespaces-debug
Find the first NaN in a model whose weights went bad.

- covers: `tl.debug`, `tl.debug.find_nan`
- tasks: debug, nan
- fixture: none

```python
bad = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU())
bad[0].weight.data.fill_(float("nan"))
report = tl.debug.find_nan(bad, torch.ones(1, 3))
print(report.found, report.kind)
# Output:
# True nan
```

### namespaces-validation
Validate a model's capture end to end against its real forward.

- covers: `tl.validation`, `tl.validation.validate_forward_pass`
- tasks: validate
- fixture: toy

```python
print(tl.validation.validate_forward_pass(model, x))
# Output:
# True
```

### namespaces-errors
Every TorchLens refusal is a `tl.errors.TorchLensError` you can catch.

- covers: `tl.errors`, `tl.errors.TorchLensError`
- tasks: errors
- fixture: toy

```python
trace = tl.trace(model, x)
try:
    trace.decode_output(style="no_such_style")
except tl.errors.TorchLensError as error:
    print(type(error).__name__)
# Output:
# DecodeStyleUnknownError
```

### namespaces-inventory
List the sites a model offers for extraction.

- covers: `tl.inventory`, `tl.inventory.list_sites`
- tasks: extract-layers, sites
- fixture: cnn

```python
import torchlens.inventory
print([row.selector for row in tl.inventory.list_sites(model)])
print(tl.inventory.list_sites(model, x).rung)
# Output:
# ['0', '1', '2']
# forward
```

### namespaces-types
The public type of each fact family hangs off `tl.types`.

- covers: `tl.types`, `tl.types.TraceCost`, `tl.types.TraceGraph`
- tasks: types
- fixture: toy

```python
trace = tl.trace(model, x)
print(isinstance(trace.cost, tl.types.TraceCost))
print(isinstance(trace.graph, tl.types.TraceGraph))
# Output:
# True
# True
```

## Analysis kits

### namespaces-stats
Stream a running mean over batches without keeping them.

- covers: `tl.stats`, `tl.stats.Mean`
- tasks: stats
- fixture: none

```python
running = tl.stats.Mean()
running.update(torch.tensor([1.0, 2.0, 3.0]))
running.update(torch.tensor([5.0]))
print(running.compute())
# Output:
# 2.75
```

### namespaces-repgeom
A representational dissimilarity matrix from stimulus-by-feature activations.

- covers: `tl.repgeom`, `tl.repgeom.rdm`
- tasks: rsa, geometry
- fixture: none

```python
acts = torch.randn(6, 4)
print(tl.repgeom.rdm(acts).shape)
print(tl.repgeom.rdm(acts, output="condensed").shape)
# Output:
# (6, 6)
# (15,)
```

### namespaces-features
Flatten one site's saved output into a stimulus-by-feature matrix.

- covers: `tl.features`, `tl.features.site_matrix`
- tasks: features
- fixture: cnn

```python
import torchlens.features
trace = tl.trace(model, torch.randn(3, 3, 8, 8), save="all")
matrix = tl.features.site_matrix(trace, "relu_1_2").matrix
print(tuple(matrix.shape))
# Output:
# (3, 256)
```

### namespaces-transforms
Built-in activation transforms: flatten and spatial pooling.

- covers: `tl.transforms`, `tl.transforms.flatten`, `tl.transforms.pool_spatial`, `tl.transforms.TransformContext`, `tl.transforms.NCHW`
- tasks: transforms
- fixture: cnn

```python
print(tuple(tl.transforms.flatten().apply(x).shape))
ctx = tl.transforms.TransformContext(roles=tl.transforms.NCHW)
print(tuple(tl.transforms.pool_spatial("max").apply(x, ctx).shape))
# Output:
# (1, 192)
# (1, 3)
```

### namespaces-attribution
Saliency: the input gradient of one output class.

- covers: `tl.attribution`, `tl.attribution.saliency`
- tasks: attribution
- fixture: toy

```python
result = tl.attribution.saliency(model, x, target=0)
print(tuple(result.values.shape))
print(bool((result.values >= 0).all()))
# Output:
# (3, 4)
# True
```

### namespaces-receptive-field
Verify a conv net's receptive fields with a gradient check.

- covers: `tl.receptive_field`, `tl.receptive_field.verify`
- tasks: receptive-field, validate
- fixture: none

```python
net = torch.nn.Conv2d(1, 1, 3, padding=1)
armed = tl.trace(net, torch.randn(1, 1, 6, 6, requires_grad=True),
    capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"))
report = tl.receptive_field.verify(armed, units="center")
print(report.verdict.value, report.passed)
# Output:
# pass True
```

### namespaces-neuro
Plan a dataset extraction: which sites fit and what they cost.

- covers: `tl.neuro`, `tl.neuro.plan_extraction`
- tasks: neuro, extract-dataset
- fixture: cnn

```python
probe = torch.randn(4, 3, 8, 8)
plan = tl.neuro.plan_extraction(model, probe, num_stimuli=8, batch_size=4)
print(len(plan.included_sites) > 0, plan.table().startswith("site"))
# Output:
# True True
```

### namespaces-preprocessing
Resolve a declared input normalization into a preprocessing record.

- covers: `tl.preprocessing`, `tl.preprocessing.resolve`
- tasks: preprocessing
- fixture: none

```python
import torchlens.preprocessing
resolution = tl.preprocessing.resolve({"mean": [0.5] * 3, "std": [0.5] * 3})
print(resolution.status, resolution.preprocessing.source)
print(resolution.declared.mean)
# Output:
# authoritative explicit_declaration
# (0.5, 0.5, 0.5)
```

### namespaces-postprocessing
List the output detectors that decode a model's raw output.

- covers: `tl.postprocessing`, `tl.postprocessing.list`
- tasks: decode, output
- fixture: none

```python
import torchlens.postprocessing
names = tl.postprocessing.list()
print(type(names).__name__, len(names) > 0)
# Output:
# list True
```

### namespaces-facets
Describe which semantic facets a module class offers.

- covers: `tl.facets`, `tl.facets.describe`
- tasks: facets, lm
- fixture: none

```python
info = tl.facets.describe("GPT2Attention")
print(sorted(info))
print("q" in info["facets"])
# Output:
# ['facets', 'functions']
# True
```

### namespaces-mechinterp
Logit-lens trajectory statistics over a batch: how each block's prediction sharpens toward the output.

- covers: `tl.mechinterp`, `tl.mechinterp.measure_trajectory`
- tasks: lm, mechinterp
- fixture: lm
- requires: transformers

```python
import torchlens.mechinterp
stats = tl.mechinterp.measure_trajectory(lm, ids, statistics=["entropy"])
print(type(stats).__name__)
print(stats.statistics, stats.num_examples)
# Output:
# TrajectoryStats
# ('entropy',) 1
```

### namespaces-mechinterp-ablation-controls
Check an ablation claim with controls: rerun it under the other baselines and read the verdict.

- covers: `tl.mechinterp.ablation_controls`, `tl.mechinterp.AblationControlsReport`
- tasks: ablate, controls, mechinterp
- fixture: toy

```python
import torchlens.mechinterp
def metric(trace):
    return trace.output_ops[-1].out.sum()
report = tl.mechinterp.ablation_controls(model, x, tl.module("fc1"), metric, axis=0)
print(isinstance(report, tl.mechinterp.AblationControlsReport), report.baseline)
print([(row.family, row.name) for row in report.rows])
print([family.family for family in report.families], isinstance(report.survives, bool))
# Output:
# True zero
# [('baseline', 'zero'), ('baseline', 'mean'), ('baseline', 'resample')]
# ['baseline'] True
```

### namespaces-transformer
Per-token next-token loss and entropy strips from one sequence's logits.

- covers: `tl.transformer`, `tl.transformer.token_metrics`
- tasks: lm, metrics
- fixture: lm
- requires: transformers

```python
import torchlens.transformer
logits = lm(ids).logits[0].detach()
tokens = tuple(str(t) for t in ids[0].tolist())
loss, entropy = tl.transformer.token_metrics(logits, ids[0], tokens)
print(len(loss.values), loss.values[0] is None)
print(loss.source_fingerprint == entropy.source_fingerprint)
# Output:
# 5 True
# True
```

### namespaces-transformer-generation-step
Picture one attention layer as it stood at one step of a recorded generation.

- covers: `tl.transformer.attention_view`
- tasks: attention, generation, lm
- fixture: none
- requires: transformers

```python
import torchlens.transformer
from transformers import GPT2Config, GPT2LMHeadModel
torch.manual_seed(0)
config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, attn_implementation="eager")
lm = GPT2LMHeadModel(config).eval()
greedy = {"max_new_tokens": 3, "min_new_tokens": 3, "do_sample": False, "use_cache": True}
ids = torch.tensor([[5, 17, 42, 8, 3]])
generation = tl.record(lm, ids, save=tl.func("softmax"), generate_kwargs=greedy)
view = tl.transformer.attention_view(generation, "transformer.h.0.attn", step=2)
print(len(generation.steps), tuple(view.pattern.shape))
print(view.episode.step, view.episode.role)
# Output:
# 3 (2, 7, 7)
# 2 decode
```

## Integrations

### namespaces-agent
The read-only tool list an agent can call.

- covers: `tl.agent`, `tl.agent.list_tools`, `tl.agent.guide`
- tasks: agent
- fixture: none

```python
tools = tl.agent.list_tools()
print("torchlens_doctor" in [tool["name"] for tool in tools])
print(tl.agent.guide().splitlines()[0])
# Output:
# True
# # TorchLens for AI agents
```

### namespaces-appliances
Register an appliance by name and find it in the registry.

- covers: `tl.appliances`, `tl.appliances.register`, `tl.appliances.names`
- tasks: appliances
- fixture: none

```python
echo = lambda trace: trace
tl.appliances.register("example.echo", echo, summary="the trace", replace=True)
print(tl.appliances.names()["example.echo"])
# Output:
# the trace
```

### namespaces-bridge
Join a torch profiler event list onto a trace's ops.

- covers: `tl.bridge`, `tl.bridge.profiler`
- tasks: profile, bridge
- fixture: toy

```python
trace = tl.trace(model, x)
events = [{"ph": "X", "name": "aten::unrelated", "ts": 0, "dur": 5}]
joined = tl.bridge.profiler.join(trace, {"traceEvents": events})
print(joined["schema"])
print(joined["unmatched_event_counts"])
# Output:
# torchlens.profiler_join.v2
# {'aten::unrelated': 1}
```

### namespaces-compat
Convert a `torch.fx` graph module into a TorchLens migration contract.

- covers: `tl.compat`, `tl.compat.from_fx`
- tasks: migrate, compat
- fixture: toy

```python
graph_module = torch.fx.symbolic_trace(model)
contract = tl.compat.from_fx(graph_module)
print(contract["schema"], len(contract["layers"]))
# Output:
# torchlens.fx_migration.v1 5
```

### namespaces-trackers
Collect scalars in memory with the tracker sink.

- covers: `tl.trackers`, `tl.trackers.MemorySink`, `tl.trackers.ScalarPoint`
- tasks: training, trackers
- fixture: none

```python
import torchlens.trackers
sink = tl.trackers.MemorySink()
sink.emit_scalar(tl.trackers.ScalarPoint("gradients/norm/0.weight", 3, 0.5))
print([(p.tag, p.step) for p in sink.scalars])
sink.close()
print(sink.closed)
# Output:
# [('gradients/norm/0.weight', 3)]
# True
```

### namespaces-viz
Render a 2-D tensor as a heatmap image.

- covers: `tl.viz`, `tl.viz.heatmap`
- tasks: draw, viz
- fixture: none

```python
image = tl.viz.heatmap(torch.arange(12.0).reshape(3, 4), width=60, height=60)
print(image.size, image.mode)
# Output:
# (60, 60) RGB
```
