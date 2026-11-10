# TorchLens agent index

Inspect, visualize and intervene on any operation in any neural network. Import as `import torchlens as tl`.

Page paths below are relative to https://github.com/johnmarktaylor91/torchlens/blob/main/

## Read first

- Many forwards or `generate()`: bind once, `bound = tl.when(site, edit).bind(model)`, call `bound(x)` or `bound.generate(...)`; plain `model(x)` stays unedited. Never `tl.trace` per step. With the KV cache, later steps see only the new token: index `[:, -1]`.
- Sites: `tl.module("layer3")` is a module's output, `tl.func("relu")` every relu op, `tl.in_module("layer3")` the ops inside it. Labels like `"relu_1_2"` exist only after a capture.
- Tuple outputs: one site per tensor; `trace["attn"]` is ambiguous, `trace.modules["attn"].outs` lists them, an edit runs on each (check `out.shape`).
- The last hidden state is the final norm's output (`model.norm`, GPT-2 `transformer.ln_f`), not the last block's.
- Saved values are copies, but an edit's `out` is live: clone what you keep or in-place ops overwrite it.
- Edit a trace with `trace.fork().do(site, edit, model=model, inputs=x)`; during capture, `tl.trace(model, x, intervene=tl.when(site, edit))`.
- Batched VJPs: capture with `capture=tl.options.CaptureOptions(save_mode="view")` (a copy has no graph), then one `torch.autograd.grad(..., is_grads_batched=True)`.
- Batches: pass `input_kwargs={"attention_mask": mask}` and read each row's last real token, not `[:, -1]`; match a reference's batch size (`tl.extract_dataset` uses 32): kernels drift about 1e-5.
- Hand results on as a saved path (`trace.save("run.tlspec")`, `tl.load`), not printed tensors; untrusted files: read-only tools, manifest first.
- Errors are typed: branch on `exc.fields['code']`, follow `exc.fields['remedy']`, not the message.
- Trace the eager module, never a `torch.compile`d, scripted or exported one, or capture from several threads at once.

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
| Capture a forward pass and read an activation | `trace = tl.trace(model, x)` | Forward hooks miss functional calls like torch.relu. | docs/field-guide/see_and_inspect.md |
| Keep only the activations you need | `trace = tl.trace(model, x, save=tl.func("relu"))` | Saving every activation, then filtering. | docs/field-guide/see_and_inspect.md |
| Get named layers' outputs as tensors | `features = tl.extract(model, x, ["layer3", "avgpool"])` | Hand-written hooks and a dict. | docs/field-guide/extract_features.md |
| Extract features over a whole dataset | `tl.extract_dataset(model, stimuli, ["layer3", "avgpool"], path="features")` | tl.trace per batch in a loop. | docs/field-guide/extract_features.md |
| Steer a language model during generation | `tl.when(tl.module("model.layers.6"), tl.steer(direction, strength=4.0)).bind(lm)` | tl.trace per step. | docs/guides/fast_paths.md |
| Ablate a component and measure the effect | `ablated = tl.trace(model, x).fork().do(tl.module("layer3"), tl.zero_ablate(), model=model, inputs=x)` | Editing the model's code or weights. | docs/field-guide/intervene.md |
| Draw the computational graph | `tl.draw(model, x)` | Exporting to ONNX or torch.fx. | docs/quickstart.md |
| Reopen a saved trace | `trace = tl.load("run.tlspec")` | pickle or torch.load. | docs/internal/agent-reference/package/key-concepts.md |
| Add a direction to one site and capture the result | `steered = tl.trace(model, x, intervene=tl.when(tl.module("layer3"), tl.steer(direction, strength=4.0, dim=1)))` | Editing the model's code or weights. | docs/field-guide/intervene.md |
| Patch activations from one input into another | `patched = tl.trace(model, x_corrupt).fork().do(tl.module("layer3"), tl.patch_from(tl.trace(model, x_clean)), model=model, inputs=x_corrupt)` | Copying tensors between hook runs. | docs/field-guide/intervene.md |
| Sweep edits over one site | `runs = tl.sweep(model, x, tl.module("layer3"), [tl.scale(s) for s in (0.0, 0.5, 2.0)], include_baseline=True)` | Separate captures with no shared baseline. | docs/intervention_api.md |
| Compare two runs site by site | `bundle = tl.Bundle({"clean": tl.trace(model, x), "ablated": ablated}, baseline="clean")` | Diffing printed summaries by eye. | docs/intervention_api.md |
| Print a summary of the model | `print(tl.summary(model, x))` | Counting model.parameters() by hand. | docs/reference/summary.md |
| Stream a large capture to disk | `trace = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))` | Capturing all in memory, then saving. | docs/intervention_api.md |
| Get gradients of intermediate activations | `trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))` | Reading .grad on intermediates (autograd drops it). | docs/backward.md |
| Check that a capture is faithful | `tl.validate(model, x)` | Trusting custom kernels unchecked. | README.md |
| Record a few sites over many forwards cheaply | `recording = tl.record(model, x, save=tl.func("relu"))` | A full tl.trace per forward for a few sites. | docs/performance.md |
| Compute an input attribution map | `cam = tl.attribution.grad_cam(model, x, target=3)` | Hand-rolled hooks and gradients. | docs/reference/attribution.md |
| Collect features to train a linear probe | `tl.extract_dataset(model, stimuli, ["layer4"], path="probe_features")` | tl.trace per batch. | docs/field-guide/extract_features.md |
| Open an interactive viewer of a trace | `tl.export.html(tl.trace(model, x), "trace.html")` | Screenshots of a static graph. | docs/reference/export.md |
| Diagnose a capture problem | `print(tl.report.doctor(model=model, inputs=x).to_markdown())` | Guessing before reading the report. | docs/internal/agent-reference/common-patterns.md |
| Inspect an untrusted artifact | `overview = tl.agent.call_tool("torchlens_overview", {"path": "run.tlspec", "mode": "manifest"})` | tl.load on an untrusted artifact before its manifest. | docs/for-ai-agents.md |
| Draw a reproducible subsample before extracting features | `tl.extract_dataset(model, stimuli[chosen], sites, path="subsample", stimulus_ids=ids)` | sample_size= or seed=: neither exists; the manifest records ids. | docs/field-guide/seeded_subsample.md |
| Record activations across a whole generation | `recording = tl.record(lm, ids, save=tl.module("model.layers.6"), generate_kwargs={"max_new_tokens": 16, "do_sample": False})` | tl.trace per token. | docs/internal/agent-reference/common-patterns.md |
| Gather activations to train a sparse autoencoder | `tl.extract_dataset(model, stimuli, ["layer4"], batch_size=64, path="sae_acts")` | tl.trace per batch. | docs/field-guide/sae_trainers.md |
| Build a representational dissimilarity matrix | `rdm = tl.repgeom.rdm(features)` | A pairwise distance loop. | docs/semantic_io.md |
| Compare model features with brain data | `result = tl.neuro.compare(trace, brain, analyses=("rsa", "ridge", "cka"))` | Hand-written RSA and ridge scoring. | docs/reference/neuro.md |
| Stream summary statistics over a dataset | `stats = tl.stats.aggregate(model, batches, metrics={"relu_1_2": tl.stats.Aggregator(tl.stats.Mean())})` | Saving every batch to average later. | docs/reference/stats.md |
| Fail a test when the architecture changes | `tl.fingerprint.assert_unchanged(model, x, expected=None)` | Comparing str(model): it misses forward changes. | docs/reference/hash.md |
| Monitor activations on a production inference server | none: not a TorchLens task | Wrapping a server's forward in a capture. | docs/migration/cross_engine.md |

Capabilities: docs/reference/agent_capabilities.md. Saved traces: `torchlens guide`. Examples: torchlens/_docs/example_bank/INDEX.md.
