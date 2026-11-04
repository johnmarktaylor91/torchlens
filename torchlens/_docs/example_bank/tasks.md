# Tasks: each task card's first call carried to a printed result

## Capturing and reading activations

### tasks-capture
Capture a forward pass and list its layers.

- covers: `tl.trace`, `Trace.layer_labels`
- tasks: capture
- fixture: toy

```python
trace = tl.trace(model, x)
print(trace.layer_labels)
print(tuple(trace["relu_1_2"].out.shape))
# Output:
# ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
# (3, 8)
```

### tasks-select-save
Save only the relu activations during capture.

- covers: `tl.trace`, `tl.func`, `Trace.saved_ops`
- tasks: select-save
- fixture: toy

```python
trace = tl.trace(model, x, save=tl.func("relu"))
print([op.label for op in trace.saved_ops])
# Output:
# ['relu_1_2:1']
```

### tasks-extract-layers
Get named modules' outputs as a dict of tensors.

- covers: `tl.extract`
- tasks: extract-layers
- fixture: cnn

```python
features = tl.extract(model, x, ["0", "2"])
print({name: tuple(t.shape) for name, t in features.items()})
# Output:
# {'0': (1, 4, 8, 8), '2': (1, 2, 8, 8)}
```

### tasks-extract-dataset
Extract module outputs over a whole stimulus set in batches, written to a folder.

- covers: `tl.extract_dataset`
- tasks: extract-dataset
- fixture: cnn

```python
import os

stimuli = torch.randn(6, 3, 8, 8)
shards = tl.extract_dataset(model, stimuli, ["0", "2"], batch_size=2, path="features", progress=False)
print(len(shards), os.path.isdir("features"))
# Output:
# 3 True
```

### tasks-to-disk
Stream the activations of one module to a `.tlspec` file while capturing.

- covers: `tl.to_disk`, `tl.in_module`
- tasks: to-disk
- fixture: cnn

```python
import os

trace = tl.trace(model, x, save=tl.in_module("0"), storage=tl.to_disk("run.tlspec"))
print(os.path.exists("run.tlspec"))
print([op.label for op in trace.saved_ops])
# Output:
# True
# ['conv2d_1_1:1']
```

### tasks-save-load
Save a trace to a path and load it back from that path.

- covers: `tl.load`, `Trace.save`
- tasks: save-load
- fixture: toy

```python
tl.trace(model, x).save("run.tlspec")
trace = tl.load("run.tlspec")
print(trace.layer_labels)
# Output:
# ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
```

### tasks-record-many
Record a few sites cheaply over a forward pass.

- covers: `tl.record`
- tasks: record-many
- fixture: toy

```python
recording = tl.record(model, x, save=tl.func("relu"))
print(recording.status, len(recording.records))
print(tuple(recording.records[0].ram_payload.shape))
# Output:
# complete 1
# (3, 8)
```

### tasks-record-generation
Record one module's output across a language model's generation steps.

- covers: `tl.record`
- tasks: record-generation
- fixture: lm
- requires: transformers

```python
recording = tl.record(lm, ids, save=tl.module("transformer.h.0.mlp"),
                      generate_kwargs={"max_new_tokens": 3, "do_sample": False})
print(type(recording).__name__, len(recording.steps), recording.prompt_length)
print(tuple(recording.sequences.shape), recording.recording.status)
# Output:
# GenerationRecording 3 5
# (1, 8) complete
```

### tasks-gradients
Capture with gradients saved, run backward, and read a layer's gradient.

- covers: `Trace.backward`, `Layer.grad`
- tasks: gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
grad = trace.layers["relu_1_2"].grad
print(tuple(grad.shape), grad.dtype)
# Output:
# (3, 8) torch.float32
```

## Interventions

### tasks-steer-generate
Bind a steering edit to a language model once and generate through it.

- covers: `tl.when`, `tl.steer`
- tasks: steer-generate
- fixture: lm
- requires: transformers

```python
direction = torch.randn(32)
steer = tl.steer(direction, strength=4.0, dim=-1)
bound = tl.when(tl.module("transformer.h.0.mlp"), steer).bind(lm)
steered = bound.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
print(tuple(steered.shape))
# Output:
# (1, 8)
```

### tasks-steer-once
Steer one module's output during a single capture.

- covers: `tl.when`, `tl.steer`, `tl.module`
- tasks: steer-once
- fixture: cnn

```python
direction = torch.ones(4)
steered = tl.trace(model, x, intervene=tl.when(tl.module("0"), tl.steer(direction, strength=4.0, dim=1)))
clean = tl.trace(model, x)
print(torch.allclose(steered.modules["0"].out, clean.modules["0"].out + 4.0))
print(torch.equal(steered.modules["2"].out, clean.modules["2"].out))
# Output:
# True
# False
```

### tasks-ablate
Zero-ablate a module on a fork of a captured trace and see the output move.

- covers: `Trace.fork`, `Trace.do`, `tl.zero_ablate`, `Trace.modules`
- tasks: ablate
- fixture: cnn

```python
clean = tl.trace(model, x)
ablated = tl.trace(model, x).fork().do(tl.module("1"), tl.zero_ablate(), model=model, inputs=x)
print(bool((ablated.modules["1"].out == 0).all()))
print(torch.equal(ablated.modules["2"].out, clean.modules["2"].out))
# Output:
# True
# False
```

### tasks-patch
Patch a module's output from a clean run into a corrupted run.

- covers: `tl.patch_from`
- tasks: patch
- fixture: cnn

```python
x_clean, x_corrupt = x, torch.randn(1, 3, 8, 8)
clean = tl.trace(model, x_clean)
patched = tl.trace(model, x_corrupt).fork().do(tl.module("1"), tl.patch_from(clean), model=model, inputs=x_corrupt)
print(torch.equal(patched.modules["1"].out, clean.modules["1"].out))
print(torch.allclose(patched.modules["2"].out, clean.modules["2"].out))
# Output:
# True
# True
```

### tasks-sweep
Run one capture per scale factor at a module, with an unedited baseline.

- covers: `tl.sweep`, `tl.scale`
- tasks: sweep
- fixture: cnn

```python
runs = tl.sweep(model, x, tl.module("1"), [tl.scale(s) for s in (0.0, 0.5, 2.0)], include_baseline=True)
print(runs.names)
print([bool((t.modules["1"].out == 0).all()) for t in runs])
# Output:
# ['baseline', 'sweep_0', 'sweep_1', 'sweep_2']
# [False, True, False, False]
```

### tasks-compare-runs
Put a clean and an ablated run in one bundle and rank the sites that changed.

- covers: `tl.Bundle`, `Bundle.rank_changes`
- tasks: compare-runs
- fixture: cnn

```python
ablated = tl.trace(model, x).fork().do(tl.module("1"), tl.zero_ablate(), model=model, inputs=x)
bundle = tl.Bundle({"clean": tl.trace(model, x), "ablated": ablated}, baseline="clean")
print(bundle.names)
print([label for label, _ in bundle.rank_changes(k=2, metric="relative_l2")])
# Output:
# ['clean', 'ablated']
# ['conv2d_2_3', 'output_1']
```

## Pictures, summaries and checks

### tasks-draw
Draw the model's graph to a file.

- covers: `tl.draw`
- tasks: draw
- fixture: cnn

```python
import os

drawing = tl.draw(model, x, path="model.svg")
print(os.path.exists("model.svg"))
# Output:
# TorchLens graph written to model.svg
# True
```

### tasks-summary
Print a text summary of the model's forward pass.

- covers: `tl.summary`
- tasks: summary
- fixture: cnn

```python
text = tl.summary(model, x)
print(text.splitlines()[0])
print("conv2d" in text.lower())
# Output:
# Sequential | input (1, 3, 8, 8) float32 | eval mode, no_grad, state restored
# True
```

### tasks-validate
Check that TorchLens captures this model correctly.

- covers: `tl.validate`
- tasks: validate
- fixture: toy

```python
print(tl.validate(model, x))
# Output:
# True
```

### tasks-export-html
Write a capture as a self-contained HTML page.

- covers: `tl.export.html`
- tasks: export-html
- fixture: toy

```python
out = tl.export.html(tl.trace(model, x), "trace.html")
print(out.name, out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>"))
# Output:
# trace.html True
```

### tasks-fingerprint
Pin a model's computation, then assert later that it has not changed.

- covers: `tl.fingerprint.assert_unchanged`
- tasks: fingerprint
- fixture: toy

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pin = tl.fingerprint.assert_unchanged(model, x, expected=None)
print(type(pin).__name__)
print(tl.fingerprint.assert_unchanged(model, x, expected=pin) == pin)
# Output:
# str
# True
```

### tasks-doctor
Get the environment and model report to read before debugging.

- covers: `tl.report.doctor`
- tasks: doctor
- fixture: toy

```python
report = tl.report.doctor(model=model, inputs=x)
text = report.to_markdown()
print(type(report).__name__, isinstance(text, str), len(report.checks) > 0)
# Output:
# DoctorReport True True
```

### tasks-agent-inspect
Read an artifact's manifest with the read-only agent tool, without unpickling it.

- covers: `tl.agent.call_tool`
- tasks: agent-inspect
- fixture: toy

```python
tl.trace(model, x).save("run.tlspec")
overview = tl.agent.call_tool("torchlens_overview", {"path": "run.tlspec", "mode": "manifest"})
print(overview["status"], sorted(overview["data"])[:5])
# Output:
# ok ['declared_payload_bytes', 'declared_payload_count', 'gradient_blob_count', 'has_backward_pass', 'kind']
```

## Analysis on top of captures

### tasks-attribution
Grad-CAM heat map for one class of a small classifier.

- covers: `tl.attribution.grad_cam`
- tasks: attribution
- fixture: none

```python
torch.manual_seed(0)
clf = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
                          torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 3)).eval()
cam = tl.attribution.grad_cam(clf, torch.randn(1, 3, 8, 8), target=0, layer="0", overlay=False)
print(tuple(cam.values.shape), cam.extra["layer"])
# Output:
# (1, 1, 8, 8) 0
```

### tasks-probe
Extract probe features for a stimulus set to a folder, then load them for a probe.

- covers: `tl.extract_dataset`
- tasks: probe
- fixture: cnn

```python
stimuli = torch.randn(6, 3, 8, 8)
tl.extract_dataset(model, stimuli, ["2"], batch_size=3, path="probe_features", progress=False)
features = tl.extract_dataset(model, stimuli, ["2"], batch_size=3, progress=False)["2"]
print(tuple(features.flatten(1).shape))
# Output:
# (6, 128)
```

### tasks-train-sae
Extract a layer's activations in large batches as a sparse-autoencoder training set.

- covers: `tl.extract_dataset`
- tasks: train-sae
- fixture: cnn

```python
import os

stimuli = torch.randn(8, 3, 8, 8)
shards = tl.extract_dataset(model, stimuli, ["1"], batch_size=4, path="sae_acts", progress=False)
print(len(shards), os.path.isdir("sae_acts"))
# Output:
# 2 True
```

### tasks-rdm
A representational dissimilarity matrix from features.

- covers: `tl.repgeom.rdm`
- tasks: rdm
- fixture: cnn

```python
stimuli = torch.randn(5, 3, 8, 8)
features = tl.extract(model, stimuli, ["2"])["2"].flatten(1)
rdm = tl.repgeom.rdm(features)
print(tuple(rdm.shape))
# Output:
# (5, 5)
```

### tasks-neuro-compare
Score a model's layers against recorded responses with RSA, ridge and CKA.

- covers: `tl.neuro.compare`
- tasks: neuro-compare
- fixture: cnn

```python
stimuli = torch.randn(12, 3, 8, 8)
features = tl.extract(model, stimuli, ["0", "2"])
source = {name: t.flatten(1) for name, t in features.items()}
brain = torch.randn(12, 3)
result = tl.neuro.compare(source, brain, analyses=("rsa", "ridge", "cka"))
print(result.sites, result.analyses)
# Output:
# ('0', '2') ('rsa', 'ridge', 'cka')
```

### tasks-stats-dataset
Average one op's activations over batches without keeping them.

- covers: `tl.stats.aggregate`, `tl.stats.Aggregator`, `tl.stats.Mean`
- tasks: stats-dataset
- fixture: toy

```python
batches = [torch.randn(3, 4) for _ in range(3)]
stats = tl.stats.aggregate(model, batches, metrics={"relu_1_2": tl.stats.Aggregator(tl.stats.Mean())})
print(sorted(stats["relu_1_2"]))
# Output:
# ['Mean']
```

### tasks-serving-scale
TorchLens is not a serving-time monitor: score a sample of logged requests offline instead.

- covers: `tl.extract`
- tasks: serving-scale
- fixture: toy

```python
sampled_requests = torch.randn(16, 4)
features = tl.extract(model, sampled_requests, ["fc1"])
probe = torch.ones(8) / 8
scores = features["fc1"] @ probe
print(tuple(scores.shape))
# Output:
# (16,)
```
