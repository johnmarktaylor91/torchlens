# TorchLens agent index

A package for extracting activations from PyTorch models and saving portable bundles. Import as `import torchlens as tl`.

Page paths below are relative to https://github.com/johnmarktaylor91/torchlens/blob/main/

## Read first

- Steering or patching many forwards, or through `generate()`: build the edit once and bind it, `tl.when(site, edit).bind(model)`. Never run `tl.trace` per step: a capture per step costs far more than a hook.
- Address sites with selectors: `tl.module("layer3")` is a module's output, `tl.func("relu")` every relu op, `tl.in_module("layer3")` the ops inside a module. Op labels such as `"relu_1_2"` exist only after a capture.
- Edit a captured trace with `trace.fork().do(site, edit, model=model, inputs=x)`; edit during capture with `tl.trace(model, x, intervene=tl.when(site, edit))`.
- Hand results on as a saved path (`trace.save("run.tlspec")`, reopened with `tl.load`), not printed tensors. Open your own or a trusted file with `tl.load` directly; inspect an untrusted one with the read-only tools, manifest first.
- Errors are typed: branch on `exc.fields['code']` and follow `exc.fields['remedy']`, never the message text.
- Trace the eager module, never a `torch.compile`, TorchScript or exported artifact, and never run captures from several threads at once.

## Core calls

```python
import torchlens as tl
trace = tl.trace(model, x)
trace = tl.trace(model, x, save=tl.func("relu"))
features = tl.extract(model, x, ["layer3", "avgpool"])
tl.extract_dataset(model, stimuli, ["layer3", "avgpool"], path="features")
tl.when(tl.module("model.layers.6"), tl.steer(direction, strength=4.0)).bind(lm)
ablated = tl.trace(model, x).fork().do(tl.module("layer3"), tl.zero_ablate(), model=model, inputs=x)
tl.draw(model, x)
trace = tl.load("run.tlspec")
```

## Tasks

| Task | Call | Avoid | Page |
|---|---|---|---|
| Capture a forward pass and read an activation | `trace = tl.trace(model, x)` | Hand-written forward hooks, which miss functional calls such as torch.relu. | docs/field-guide/see_and_inspect.md |
| Keep only the activations you need | `trace = tl.trace(model, x, save=tl.func("relu"))` | Saving every activation of a large model and filtering afterwards. | docs/field-guide/see_and_inspect.md |
| Get named layers' outputs as tensors | `features = tl.extract(model, x, ["layer3", "avgpool"])` | Hand-written hooks and a dict. | docs/field-guide/extract_features.md |
| Extract features over a whole dataset | `tl.extract_dataset(model, stimuli, ["layer3", "avgpool"], path="features")` | tl.trace once per batch in a loop. | docs/field-guide/extract_features.md |
| Steer a language model during generation | `tl.when(tl.module("model.layers.6"), tl.steer(direction, strength=4.0)).bind(lm)` | tl.trace at every generation step. | docs/guides/fast_paths.md |
| Ablate a component and measure the effect | `ablated = tl.trace(model, x).fork().do(tl.module("layer3"), tl.zero_ablate(), model=model, inputs=x)` | Editing the model's weights or forward code. | docs/field-guide/intervene.md |
| Draw the computational graph | `tl.draw(model, x)` | Exporting to ONNX or torch.fx to look at it. | docs/quickstart.md |
| Reopen a saved trace | `trace = tl.load("run.tlspec")` | Opening the file with pickle or torch.load. | docs/agent-reference/package/key-concepts.md |
| Add a direction to one site and capture the result | `steered = tl.trace(model, x, intervene=tl.when(tl.module("layer3"), tl.steer(direction, strength=4.0, dim=1)))` | Editing the model's forward code or weights. | docs/field-guide/intervene.md |
| Patch activations from one input into another | `patched = tl.trace(model, x_corrupt).fork().do(tl.module("layer3"), tl.patch_from(tl.trace(model, x_clean)), model=model, inputs=x_corrupt)` | Copying tensors between two hook runs by hand. | docs/field-guide/intervene.md |
| Sweep edits over one site | `runs = tl.sweep(model, x, tl.module("layer3"), [tl.scale(s) for s in (0.0, 0.5, 2.0)], include_baseline=True)` | A hand-written loop of separate captures with no shared baseline. | docs/intervention_api.md |
| Compare two runs site by site | `bundle = tl.Bundle({"clean": tl.trace(model, x), "ablated": ablated}, baseline="clean")` | Diffing two printed summaries by eye. | docs/intervention_api.md |
| Print a summary of the model | `print(tl.summary(model, x))` | Counting parameters by hand from model.parameters(). | docs/reference/summary.md |
| Stream a large capture to disk | `trace = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))` | Capturing everything in memory, then saving. | docs/intervention_api.md |
| Get gradients of intermediate activations | `trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))` | Reading .grad on intermediate tensors, which autograd does not keep. | docs/backward.md |
| Check that a capture is faithful | `tl.validate(model, x)` | Trusting a capture of custom kernels unchecked. | README.md |
| Record a few sites over many forwards cheaply | `recording = tl.record(model, x, save=tl.func("relu"))` | A full tl.trace per forward when only a few sites are needed. | docs/performance.md |
| Compute an input attribution map | `cam = tl.attribution.grad_cam(model, x, target=3)` | Hand-rolled hooks and gradients. | docs/reference/attribution.md |
| Collect features to train a linear probe | `tl.extract_dataset(model, stimuli, ["layer4"], path="probe_features")` | Calling tl.trace per batch to gather probe features. | docs/field-guide/extract_features.md |
| Open an interactive viewer of a trace | `tl.export.html(tl.trace(model, x), "trace.html")` | Screenshots of a static graph. | docs/reference/export.md |
| Diagnose a capture problem | `print(tl.report.doctor(model=model, inputs=x).to_markdown())` | Guessing before reading the report. | docs/agent-reference/common-patterns.md |
| Inspect an untrusted artifact | `overview = tl.agent.call_tool("torchlens_overview", {"path": "run.tlspec", "mode": "manifest"})` | Calling tl.load on an untrusted artifact before reading its manifest. | docs/for-ai-agents.md |
| Record activations across a whole generation | `recording = tl.record(lm, ids, save=tl.module("model.layers.6"), generate_kwargs={"max_new_tokens": 16, "do_sample": False})` | A separate tl.trace per generated token. | docs/agent-reference/common-patterns.md |
| Gather activations to train a sparse autoencoder | `tl.extract_dataset(model, stimuli, ["layer4"], batch_size=64, path="sae_acts")` | Calling tl.trace per batch to build an SAE training set. | docs/recipes/sae_trainers.md |
| Build a representational dissimilarity matrix | `rdm = tl.repgeom.rdm(features)` | A hand-written pairwise distance loop. | docs/semantic_io.md |
| Compare model features with brain data | `result = tl.neuro.compare(trace, brain, analyses=("rsa", "ridge", "cka"))` | Re-implementing RSA and ridge scoring by hand. | docs/reference/neuro.md |
| Stream summary statistics over a dataset | `stats = tl.stats.aggregate(model, batches, metrics={"relu_1_2": tl.stats.Aggregator(tl.stats.Mean())})` | Saving all activations of every batch to average later. | docs/reference/stats.md |
| Fail a test when the architecture changes | `tl.fingerprint.assert_unchanged(model, x, expected=None)` | Comparing str(model), which misses changes inside forward. | docs/reference/hash.md |
| Monitor activations on a production inference server | none: not a TorchLens task | Wrapping a server's forward in a capture. | docs/non_goals.md |

Capabilities by name: docs/reference/agent_capabilities.md. Read-only tools over saved artifacts: `torchlens guide`.
