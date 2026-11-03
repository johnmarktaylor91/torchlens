# Connectors to other tools

TorchLens hands what it records to other packages in the form each one
expects. This page lists every connector that ships, one entry per external
tool: what the connector does, the exact call, and the pip extra it needs.

Every connector is import-inert. `import torchlens` installs none of them, and
a connector whose dependency is missing raises an `ImportError` naming the
extra to install at the first call, never at import. Install an extra with
`pip install "torchlens[<extra>]"`. `torchlens[all]` bundles the SHAP,
Grad-CAM, inseq, steering-vectors, repeng, dialz, depyf, compat-shims and
viz extras; the rest are installed one at a time. Connectors under `tl.bridge` resolve lazily
(`tl.bridge.captum` imports TorchLens code only); the viewers and trackers
live under `tl.export`.

Two kinds of tool need no connector at all. Plain tensors and tables (every
saved activation is a `torch.Tensor`; `tl.export.csv` and `tl.export.parquet`
write tables) go straight into scikit-learn, pandas or numpy. Tools that are
alternative ways into a network rather than consumers of its record, such as
TransformerLens, nnsight (as a live tracer), pyvene and torchview, have no
connector; TorchLens's `torchlens.mechinterp` kit is modelled on
TransformerLens's and is documented in [the mech-interp reference](reference/mechinterp.md).

The examples below each run on a small model in a fresh interpreter. The
`model`, `x` and `trace` names are shared across them:

```python
import torch
import torchvision
import torchlens as tl

model = torchvision.models.resnet18(weights=None).eval()
x = torch.randn(2, 3, 64, 64)
trace = tl.trace(model, x)
```

## Analysis tools TorchLens feeds

### Captum

[pytorch/captum](https://github.com/pytorch/captum). Runs any Captum
attribution object against the inputs a trace retained, and turns any site you
pick out in TorchLens into a layer Captum's layer methods can hook.

- `tl.bridge.captum.attribute(log, method, target, *, inputs=None, **kwargs)`
  calls `method.attribute(...)` on the trace's first saved input tensor (or
  `inputs=`). When `method.layer` is a site probe (below), the call runs the
  model through the capture-free bound-intervention door with the probe
  spliced at that op, so Captum's forward hook on the probe reads and replaces
  exactly that tensor; the model itself keeps no hook afterwards.
- `tl.bridge.captum.layer(log, site, *, pass_index=None)` returns the live
  `nn.Module` for a module address, a pass-qualified address, an op label, a
  selector or a layer object when a single-call module returns that tensor
  (an op label resolves to the outermost such module: `relu_17_66` in
  resnet18 is `layer4`, not `layer4.1.relu`). Every other site, an op no
  module returns (the residual `add` of a basic block) or one call of a
  module that runs several times, returns a site probe: an identity module
  spliced into the forward at the op's structural `site_key` (its
  `repr` names the label and key), usable only through
  `tl.bridge.captum.attribute`. `pass_index=` (1-based) picks the call of a
  module that runs more than once: `layer(log, "layer4.1.relu", pass_index=2)`
  is `layer(log, "layer4.1.relu:2")`, and it needs a plain module address
  (`bridge_pass_index_needs_module_address`; a non-positive value refuses
  `bridge_pass_index_invalid`). One probe per method object
  (`bridge_captum_multi_probe_unsupported`).
- Extra: `captum`.

```python
from captum.attr import LayerIntegratedGradients

lig = LayerIntegratedGradients(model, tl.bridge.captum.layer(trace, "layer4"))
attribution = tl.bridge.captum.attribute(trace, lig, target=0, n_steps=4)
print(tuple(attribution.shape))

# A functional op: the residual add inside the last basic block.
add_label = trace["layer4.1.relu:2"].parents[0]
lig_add = LayerIntegratedGradients(model, tl.bridge.captum.layer(trace, add_label))
print(tuple(tl.bridge.captum.attribute(trace, lig_add, target=0, n_steps=4).shape))
```

### SAELens

[jbloomAus/SAELens](https://github.com/jbloomAus/SAELens). Encodes or decodes
one saved activation with a sparse autoencoder (a pretrained `SAE` from
SAELens, or anything with `encode`/`decode`).

- `tl.bridge.sae_lens.encode(log, site, sae)` and
  `tl.bridge.sae_lens.decode(log, site, sae)` call the SAE on the saved
  output at `site`. Shapes are the SAE's business: pick a site whose output
  matches the SAE's input width, and trace with `config.use_cache = False` on
  Hugging Face decoders so a block address names one tensor.
- `tl.bridge.sae.splice(model_or_log, inputs, *, site, sae, latents_edit=None)`
  needs no extra: it replaces the site activation with the SAE's
  reconstruction, replays the rest of the network and reports reconstruction
  fidelity and output effect. `latents_edit` edits the latents first (ablate
  or steer a feature).
- Extra: `sae` (SAELens itself; Python 3.10 or newer). `splice` works with any
  encode/decode pair.

### SHAP

[shap/shap](https://github.com/shap/shap). Runs a SHAP explainer (default
`shap.DeepExplainer`) against the live model a trace still references.

- `tl.bridge.shap.explain(log, *, background, inputs=None, explainer_class=None, **kwargs)`
  returns the SHAP values and the explainer. `background` is required: SHAP
  values are contributions relative to a background's expected output, and
  explaining an input against itself gives all zeros, so there is no default.
- Extra: `shap`.

### pytorch-grad-cam

[jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
Runs a CAM method on the module behind a TorchLens site, using the trace's
saved input.

- `tl.bridge.gradcam.cam(log, site, *, inputs=None, targets=None, cam_class=None, **kwargs)`
  returns the CAM array and the resolved target module (`result["cam"]`,
  `result["target_layers"]`); `cam_class` defaults
  to `pytorch_grad_cam.GradCAM`. Keywords the CAM constructor declares
  (`reshape_transform`) go to the constructor, the rest (`aug_smooth`,
  `eigen_smooth`) to the call.
- `tl.bridge.gradcam.layer(log, site)` is the same module resolver Captum
  uses, with the same refusals.
- Extra: `gradcam`.

```python
result = tl.bridge.gradcam.cam(trace, "layer4")
print(result["cam"].shape, type(result["target_layers"][0]).__name__)
```

### inseq

[inseq-team/inseq](https://github.com/inseq-team/inseq). Runs an inseq
feature-attribution model over text and normalises the result.

- `tl.bridge.inseq.attribute(model_or_id, inputs, *, method="integrated_gradients", generated_texts=None, attribution_model=None, **kwargs)`
  loads the model through `inseq.load_model` (or takes a prebuilt
  `attribution_model`) and attributes `inputs`. inseq ignores unknown
  keywords with only a warning, so check spellings against inseq's own
  signatures.
- Extra: `inseq`.

### rsatoolbox

[rsagroup/rsatoolbox](https://github.com/rsagroup/rsatoolbox). Turns saved
activations into rsatoolbox `Dataset` objects, with the stimulus axis as
observations and the flattened units as channels, so representational
similarity analysis runs on rsatoolbox's side.

- `tl.bridge.rsatoolbox.dataset(source, site=None)` builds one `Dataset` from
  a `Trace`, a loaded extraction or an extraction directory. The site must be
  stimulus-indexed (its leading axis is the batch of stimuli); a buffer or a
  per-step scalar is refused rather than mislabelled.
- `tl.neuro.datasets(source, sites=None, *, pool=None, obs=None, stimulus_ids=None)`
  does the same for many sites at once, and `tl.neuro.rdms(...)` computes
  RDMs with provenance. Both appear only when rsatoolbox is installed. The
  full treaty between the two packages is in [the neuro reference](reference/neuro.md).
- Extra: `neuro`.

```python
dataset = tl.bridge.rsatoolbox.dataset(trace, "avgpool")
print(dataset.n_obs, dataset.n_channel)
```

### Brain-Score

[brain-score/vision](https://github.com/brain-score/vision). Supplies the
activation extractor a Brain-Score model wrapper builds internally, with
TorchLens capture in place of forward hooks, so benchmark layer lists can
name functional ops as well as modules.

- `tl.bridge.brain_score.get_activations_fn(model, *, layer_map=None, device=None, logits_lookup="output_1")`
  returns the `get_activations(images, layer_names)` callable
  `ActivationsExtractorHelper` expects. It needs no Brain-Score install.
- `tl.bridge.brain_score.activations_extractor(model, preprocessing, *, identifier=None, layer_map=None, device=None, **kwargs)`
  wires that callable into a real `ActivationsExtractorHelper`.
- `tl.bridge.brain_score.per_layer(log, benchmark, *, sites=None, **kwargs)`
  scores saved layers independently with any callable benchmark; offline.
- `tl.bridge.xarray.neuroid_assembly(source, site)` builds a `NeuroidAssembly`
  over one site for bring-your-own-alignment work.
- Extra: `brainscore` (Python 3.11 or newer) for `activations_extractor` and
  `neuroid_assembly`; the other two calls run without it.

```python
get_activations = tl.bridge.brain_score.get_activations_fn(model)
acts = get_activations(x, ["layer1", "logits"])
print({name: arr.shape for name, arr in acts.items()})
```

### xarray

[pydata/xarray](https://github.com/pydata/xarray). Builds a labelled
`presentation x neuroid` array from one site or from a whole trace, with
recorded stimulus ids as coordinates and provenance in `attrs`.

- `tl.bridge.xarray.data_array(source, site)` for one site, from a `Trace`, a
  loaded extraction or an extraction directory.
- `tl.export.xarray(log)` for every saved output that shares a presentation
  count.
- Extra: none; `xarray` must be installed.

## Other interpretability and steering tools

### steering-vectors

[steering-vectors/steering-vectors](https://github.com/steering-vectors/steering-vectors).
Trains the package's own steering vector from saved activations of positive
and negative prompts, without re-running the model.

- `tl.bridge.steering_vectors.vector(log, positive_site, negative_site=None, *, negative_log=None, read_token_index=-1, attention_mask=None, negative_attention_mask=None, trainer=None, layer=None, layer_type="decoder_block", **kwargs)`.
  The negative prompts usually live in a second trace (`negative_log=`); one
  token per prompt is read before training (`-1` is the last real token,
  padding-aware when a mask is known); `layer=` also returns a real
  `SteeringVector`. Bit-identical to `train_steering_vector` with
  `read_token_index=-1, batch_size=1`.
- Extra: `steering`.

### repeng

[vgel/repeng](https://github.com/vgel/repeng). Builds a `repeng.ControlVector`
from contrastive saved activations, replicating `read_representations`.

- `tl.bridge.repeng.control_vector(log, positive_site, negative_site=None, *, layer, negative_log=None, read_token_index=-1, attention_mask=None, negative_attention_mask=None, method="pca_diff", model_type=None)`.
  repeng's layer `i` reads the output of `model.layers.<i>`, except the last
  layer, where Hugging Face returns the final-norm output (`model.norm`).
- Extra: `repeng`.

### dialz

[Dialz](https://github.com/cardiffnlp/dialz). Builds a `dialz.SteeringVector`
the same way, against `dialz.vector.read_representations`.

- `tl.bridge.dialz.vector(log, positive_site, negative_site=None, *, layer, negative_log=None, read_token_index=-1, attention_mask=None, negative_attention_mask=None, method=None, model_type=None)`;
  `method=None` follows the installed release's default.
- Extra: `dialz` (Python below 3.13).

### LIT (Learning Interpretability Tool)

[PAIR-code/lit](https://github.com/PAIR-code/lit). Wraps a live torch model
as a real LIT `Model` so Google's served dashboard (counterfactual editing,
embedding projection, salience, side-by-side comparison) can drive it, with
TorchLens-captured activations exposed as embedding fields.

- `tl.bridge.lit.model(net, tokenizer, *, task="classification", sites=..., pooling="mean_masked", labels=None, salience=False, ...)`
  takes the live model, never a finished trace (LIT calls `predict` on new
  inputs). `sites` is required: the `"blocks"` preset or explicit module
  addresses. `tl.bridge.lit.dataset(examples, *, labels=None)` builds a small
  LIT dataset and `tl.bridge.lit.layout()` a layout.
- Extra: `lit`. Full guide: [LIT bridge](reference/lit_bridge.md).

### nnsight

[ndif-team/nnsight](https://github.com/ndif-team/nnsight). Normalises a
cached nnsight-style trace (a mapping, an object with `to_dict()`, or an
object with `nodes`) into a stable TorchLens payload for side-by-side
inspection. A live nnsight tracer is refused with a `TypeError` naming the
accepted shapes.

- `tl.bridge.nnsight.from_trace(trace)`.
- Extra: none (offline). The `nnsight` extra only pins the version the
  real-package test runs against.

## Hugging Face, training loops and compiled code

### Hugging Face transformers and timm

[huggingface/transformers](https://github.com/huggingface/transformers),
[huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models).
Traces a Hugging Face model from text, an image or a processor dict,
resolving the tokenizer or processor for you.

- `tl.bridge.hf.trace_text(model, text, ...)`, `tl.bridge.hf.trace_image(model, image, **kwargs)`,
  `tl.bridge.hf.trace_multimodal(model, input_dict, **kwargs)`.
- `tl.compat.from_huggingface(model_object_id, ...)` and
  `tl.compat.from_timm(model_class_name, *, pretrained=False, **kwargs)` load
  a model through the package's own loader, ready to trace;
  `tl.compat.from_sentence_transformers(model, *, prompt=None)` describes a
  Sentence Transformers model for migration.
- Extra: `hf` (transformers and timm); `compat-shims` for sentence-transformers.

### Hugging Face Trainer and PyTorch Lightning

[huggingface/transformers](https://github.com/huggingface/transformers),
[Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning).
Watches a training run from inside the framework's own callback slot, with
the framework's global step as the step source.

- `torchlens.trackers.HFTrainerWatchCallback(to=...)` and
  `torchlens.trackers.LightningWatchCallback(to=...)` stream per-step
  statistics to a sink (`TensorBoardSink`, `WandbSink`, `JSONLSink`).
- `tl.callbacks.lightning.LayerProfilerCallback` is a different tool: a
  per-epoch structure profiler that re-traces under `eval()`.
- Extra: `lightning` for the Lightning callbacks; the Trainer callback needs
  `transformers` (`hf`). Details: [trackers reference](reference/trackers.md).

### torchextractor

[antoinebrl/torchextractor](https://github.com/antoinebrl/torchextractor).
Drop-in `Extractor` spelling for code written against torchextractor, backed
by TorchLens capture.

- `tl.compat.torchextractor.Extractor(...)` and
  `tl.compat.from_torchextractor(model, layers=None)`.
- Extra: `compat-shims`.

### depyf

[thuml/depyf](https://github.com/thuml/depyf). Compiles a model under depyf
once and returns the files it dumped (decompiled Dynamo bytecode, FX graphs,
generated kernels). A companion for `torch.compile` debugging; TorchLens
itself traces the eager module.

- `tl.bridge.depyf.dump(model, x, path, **prepare_debug_kwargs)` returns the
  written paths; a cached compile that dumps nothing raises and names
  `torch.compiler.reset()`.
- Extra: `depyf`.

### PyTorch profiler (Kineto)

[pytorch/kineto](https://github.com/pytorch/kineto). Joins a Kineto or Chrome
trace produced by `torch.profiler` with the captured layers, so each
profiler event lands on the TorchLens op it timed.

- `tl.bridge.profiler.join(log, kineto_trace)` assigns each complete event to
  at most one layer: a `record_function` range named exactly like a layer
  label, else the k-th outermost `aten::<op>` event to the k-th layer of that
  op. Unmatched event types are reported, never guessed.
- `tl.bridge.profiler.execution_trace(log, path)` writes TorchLens's own
  per-layer execution-trace JSON (not the Chakra schema).
- Extra: none.

## Viewers, trackers and sharing

### Netron

[lutzroeder/netron](https://github.com/lutzroeder/netron). Writes a lossy
ONNX `ModelProto` JSON file that Netron opens with typed shapes on every
edge and the module hierarchy as drill-down boxes. It is a graph for
inspection, not a runnable ONNX model.

- `tl.export.netron(log, path=None, *, granularity="module", depth=1, show_buffers="meaningful", attachment=False, open=False, baseline=None)`
  returns the written path; `open=True` launches the Netron viewer on it.
- Extra: none to write the file; `netron` to open it from Python. Full
  guide: [Netron export](reference/netron_export.md).

```python
from pathlib import Path

path = tl.export.netron(trace, Path(globals().get("DOCS_TMPDIR", "/tmp")) / "resnet18.netron.json")
print(path.name, path.exists())
```

### Model Explorer

[google-ai-edge/model-explorer](https://github.com/google-ai-edge/model-explorer).
Writes a JSON graph collection that Google Model Explorer opens through
File > Open, with TorchLens's measured shapes and memory on the nodes.

- `tl.export.model_explorer(log, path, **kwargs)` writes the file;
  `tl.export.model_explorer_diff(members, out_dir)` writes a side-by-side
  collection for a bundle; `tl.export.model_explorer_serve(source, **kwargs)`
  starts the viewer on an export.
- Extra: none to write; the `ai-edge-model-explorer` package to serve. Full
  guide: [Model Explorer export](reference/model_explorer.md).

```python
from pathlib import Path

path = tl.export.model_explorer(trace, Path(globals().get("DOCS_TMPDIR", "/tmp")) / "resnet18.me.json")
print(path.name, path.exists())
```

### TensorBoard, Weights & Biases, MLflow and Aim

[tensorflow/tensorboard](https://github.com/tensorflow/tensorboard),
[wandb/wandb](https://github.com/wandb/wandb),
[mlflow/mlflow](https://github.com/mlflow/mlflow),
[aimhubio/aim](https://github.com/aimhubio/aim). Logs a finished trace's
per-layer facts to an experiment tracker you already opened; TorchLens never
creates runs or imports a tracker SDK itself.

- `tl.export.tensorboard(log, writer, *, step, prefix="torchlens")`,
  `tl.export.wandb(log, run=None, name="torchlens_trace")`,
  `tl.export.mlflow(log, client=None, prefix="torchlens", *, run_id=None)` and
  `tl.export.aim(log, run=None, prefix="torchlens")`. The tracker object is
  passed in and duck-typed on the method it needs; an `MlflowClient` needs
  `run_id=`, the fluent `mlflow` module does not.
- For live training, the sinks in `torchlens.trackers` stream per-step
  statistics to TensorBoard or W&B (see the Trainer and Lightning entry).
- Extras: `tensorboard`, `wandb`; MLflow and Aim have none (install the
  package). Reference: [`tl.export`](reference/export.md).

### Chrome tracing, Perfetto, speedscope and flame graphs

Per-layer timing in the formats the common profiler viewers open.

- `tl.export.chrome_trace(log, path)` (chrome://tracing and Perfetto),
  `tl.export.chrome_trace_diff(bundle, path)`, `tl.export.speedscope(log, path)`,
  `tl.export.flamegraph(log, path)`, `tl.export.memory_timeline(log, path)`.
- Extra: none.

### treescope

[google-deepmind/treescope](https://github.com/google-deepmind/treescope).
Registers bounded renderers so a `Trace`, `Layer` or `Op` shows as a
collapsible card inside DeepMind's treescope notebook renderer instead of a
megabyte dump.

- `tl.bridge.treescope.register()`, `unregister()`, the `registered()` context
  manager and `status()`. Never registers over `torch.Tensor`.
- Extra: `treescope`. Guide: [treescope cards](guides/treescope_cards.md).

### lovely-tensors and torchshow

[xl0/lovely-tensors](https://github.com/xl0/lovely-tensors),
[xwying/torchshow](https://github.com/xwying/torchshow). Familiar one-line
tensor summaries and image plots over TorchLens objects.

- `tl.compat.lovely(obj)` and `tl.compat.torchshow.show(obj)`.
- Extra: `viz`.

### Hugging Face Hub

[huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub).
Publishes a saved trace, bundle or intervention spec to a Hub repository, and
loads a published trace back in one call.

- `tl.bridge.huggingface.push_to_hub(artifact, repo_id, *, token=None, private=None, path_in_repo="torchlens_artifact.pkl", commit_message=..., create_repo=True, dry_run=False, api=None, save_level="portable")`.
  A `Trace` or `Bundle` is always serialised through the privacy-scrubbed
  portable `.tlspec` bundle path and uploaded as a `.tar.gz` (the default
  name switches to `torchlens_artifact.tar.gz`); raw pickling is reserved for
  plain objects with no bundle path, because a raw pickle would embed home
  paths and the user name. `dry_run=True` serialises locally and reports the
  planned upload without contacting the Hub.
  The save step warns when the bundle embeds custom module attributes or
  pre-forward buffer values; review them before publishing.
- `tl.bridge.huggingface.load_from_hub(repo_id, *, path_in_repo="torchlens_artifact.tar.gz", revision=None, token=None, cache_dir=None, extract_dir=None, api=None, **load_kwargs)`
  downloads that tarball (`hf_hub_download`), extracts it into `extract_dir`
  (default: a fresh temporary directory) and opens the bundle through the
  governed `tl.load` door, with `load_kwargs` passed on, so a published trace
  comes back under the same safety gates as a local `.tlspec`. Nothing is
  unpickled outside them: a download that is not a tarball holding one
  bundle (a raw `.pkl`, two bundles, no `manifest.json`) refuses
  `hub_artifact_not_a_bundle` before any member is read, and a tarball with
  a member that would land outside the extraction directory, or a link or
  device member, refuses `hub_artifact_tar_unsafe` before anything is
  written.
- Extra: `hf` (brings `huggingface_hub`).

```python
plan = tl.bridge.huggingface.push_to_hub(trace, "someone/resnet18-trace", dry_run=True)
print(plan["format"], plan["path_in_repo"], plan["size_bytes"] > 0)
```

```python
loaded = tl.bridge.huggingface.load_from_hub("someone/resnet18-trace")  # needs network
print(len(loaded.layer_list))
```

### Model Context Protocol (MCP)

[modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk).
Serves TorchLens's agent tools (doctor, overview, dump, explain, site
queries, payload statistics, compare, schema, ledger) over stdio so an MCP
host can inspect saved `.tlspec` and `.tlledger` artifacts. Read-only: the
server never runs model code or writes files.

- `python -m torchlens.bridge.mcp`; the pure tool layer
  `tl.bridge.mcp.call_tool(name, arguments)` needs no MCP install.
- Extra: `mcp`. Guide: [for AI agents](for-ai-agents.md).

## The count

Connectors on this page, by external tool: Captum, SAELens, SHAP,
pytorch-grad-cam, inseq, rsatoolbox, Brain-Score, xarray, steering-vectors,
repeng, dialz, LIT, nnsight, Hugging Face transformers and timm, Hugging Face
Trainer, PyTorch Lightning, torchextractor, depyf, the PyTorch profiler,
Netron, Model Explorer, TensorBoard, Weights & Biases, MLflow, Aim,
speedscope, treescope, lovely-tensors, torchshow, the Hugging Face Hub and
MCP. Which projects shaped TorchLens's own features, as opposed to receiving
its output, is on the [acknowledgments page](acknowledgments.md).
