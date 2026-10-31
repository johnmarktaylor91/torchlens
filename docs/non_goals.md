# What TorchLens does not do

TorchLens records what an eager PyTorch forward actually runs, lets you edit it, and checks the
record. Some neighbouring jobs are deliberately left to other tools. This page names them, so a
missing feature is a stated boundary rather than a silent gap. Edge cases inside the supported
scope (what can go wrong and what you see) are in the
[limitations catalog](reference/limitations.md).

## Out of scope: use these tools instead

| Job | Why it is out of scope | Use instead |
| --- | --- | --- |
| Training-data attribution: influence functions, TracIn and related estimators | It needs per-example training gradients or Hessian approximations over a training set, which is a training-time product, not a record of one forward. | Captum's `captum.influence` module, or dedicated data-attribution libraries such as TRAK, Kronfluence and dattri. |
| Adversarial-attack engineering: attack suites, C&W, adversarial patches | Building attacks is an optimization product of its own. | Captum's `captum.robust` module, or a dedicated attack library. TorchLens can still show what an adversarial input changed: trace the clean and the perturbed input and compare them with `tl.changed`. |
| Remove-and-retrain metrics (ROAR) and label-randomization retraining | They require retraining the model many times. | Your own training pipeline. TorchLens ships retraining-free attribution checks: `tl.attribution.infidelity` and `tl.attribution.sensitivity`. |
| Training sparse autoencoders or transcoders | TorchLens extracts the activations and hands them off; it ships no trainer. | SAELens or your own loop, fed as shown on the [SAE training hand-off](sae_training_handoff.md) page. |
| Driving training or evaluation loops | TorchLens observes loops; it never owns or drives them. | Your loop, with TorchLens called inside it ([monitor training](monitor_training.md)); trainable interventions train under your optimizer ([example](migration/from_pyvene.md#trainable-interventions)). |
| CCA, SVCCA and PWCCA | A different statistic from the CKA kernels TorchLens ships; `torchlens.neuro` refuses the names and says so. | netrep, or the documented recipes. |

## Boundaries of where TorchLens runs

| Boundary | What it means | What to do |
| --- | --- | --- |
| Remote or hosted execution | TorchLens runs in your Python process, where your model runs. It has no remote-execution service, so it cannot capture a model that runs on another machine. | Run TorchLens on the machine that holds the model. To move an experiment, save its recipe with `trace.save_intervention(...)`; the `.tlspec/` directory is the portable unit. |
| Serving engines (vLLM, SGLang) | Serving engines run fused kernels, paged KV caches and CUDA graphs outside eager PyTorch, so there is nothing for TorchLens to record. `tl.compat.report` flags them with the `serving_engine` row. | Load the same checkpoint as an eager Hugging Face model and trace that twin ([deployment envelope](reference/deployment_envelope.md#serving-engines-capture-the-hugging-face-twin)). |
| Compiled and exported graphs (`torch.compile`, TorchScript, `torch.export`, ONNX) | TorchLens records eager execution; a compiled artifact no longer runs the Python it came from. | Trace the eager source module. |
| Several captures at once | Capture uses process-global state, so it is single-threaded by design. | Capture from the main process, one capture at a time. |

## Neighbouring tools and domains

Each line says where TorchLens stops and the other tool starts. The rival-side descriptions are
short summaries; check each tool's own documentation before relying on them.

- PyTorch's own `TorchFunctionMode` and `TorchDispatchMode`: if you need only the stream of ops
  one call runs, a mode of your own is the zero-dependency choice. TorchLens builds on the same
  mechanisms and adds labels, module structure, saved activations, a validated record and editing.
- `torch.ao` quantization observers: they summarize activation ranges to calibrate quantization.
  TorchLens is not a quantization toolkit; it can record the activations an observer summarizes,
  and `tl.compat.report` flags packed-int GPTQ and AWQ weights.
- Torch-Pruning's dependency graph: it plans structural pruning. TorchLens prunes nothing; its
  graph is the record of one run, not a pruning plan.
- EasyEdit and other knowledge-editing toolkits (ROME, MEMIT): TorchLens implements no
  model-editing method. It can apply a weight change "as if" on a fork
  (`fork.do(tl.param(name), edit)`) and show what the change moved.
- DAAM and diffusers hooks: TorchLens ships no diffusion-specific views, such as per-token
  cross-attention maps, and does not drive a denoising loop.
- `torch_geometric.explain`: graph-neural-network explainers belong to PyTorch Geometric.
  TorchLens has no graph-learning explainers.
- TensorFlow and Keras explanation libraries (for example iNNvestigate), Flax NNX's `nnx.tabulate`,
  and Julia's LRP packages: TorchLens is PyTorch-first. Its TensorFlow and JAX backends are
  previews ([backends](backends.md)) and carry no attribution methods.
- Training dynamics: TorchLens records per-site distributions across training
  ([monitor training](monitor_training.md)); it is not an experiment tracker and leaves logging to
  TensorBoard or Weights & Biases.
- Security detection (backdoors, trojans, data poisoning): no detector ships. TorchLens supplies
  the activations and edits such a detector is built from.
- Representational similarity: in scope. `torchlens.repgeom` and `torchlens.neuro` ship RDMs,
  RSA and CKA ([neuro reference](reference/neuro.md)); the CCA family stays with netrep.
- Non-language modalities (vision, audio, multimodal): in scope. Capture, selectors and edits do
  not depend on the modality; some methods are defined only for certain layers (guided
  backpropagation needs exact ReLUs, for example).
