# Where to go instead

Jobs TorchLens does not do, the tool that does each one, and how a result comes back to
TorchLens when you want to check it on an eager model. Tool status is as of 2026-11-03; see
[the landscape page](landscape.md) for versions and sources.

<!-- JMT review: plain factual prose, written for review. -->

## Capture at serving scale, on remote GPUs, or across devices

TorchLens records a local eager forward. Serving engines run fused kernels, paged KV caches and
CUDA graphs outside eager PyTorch, so there is nothing for it to record there, and
`tl.report.doctor` says so in its `serving_engine` row.

- Capture or steering inside vLLM: nnsight 0.8's vLLM back end, Neuronpedia's interp-engine,
  vllm-lens (UK AI Security Institute), EasySteer for steering.
- Remote GPUs: nnsight with NDIF.
- Tensor-parallel and multi-GPU models: nnsight's tensor-parallel back end. For data-parallel
  feature extraction with TorchLens, see [multi-GPU extraction](multi_gpu_extraction.md).

Coming back: load the same checkpoint as a Hugging Face model, trace it, and check the site or
steering vector on the eager twin
([deployment envelope](../reference/deployment_envelope.md#serving-engines-capture-the-hugging-face-twin)).

## Train a sparse autoencoder or a probe

- Sparse autoencoders: SAELens and the trainers on [the SAE trainers recipe](../recipes/sae_trainers.md).
  TorchLens writes the activations they train on (`tl.extract_dataset`) and splices a trained
  SAE back in (`torchlens.bridge.sae.splice`, [pretrained SAEs](../recipes/pretrained_saes.md)).
- Probes: the training loop is yours (scikit-learn or plain PyTorch); `tl.extract_dataset`
  writes the features with their provenance ([non-goals](../non_goals.md)).

## Train an intervention (DAS, LoReFT)

pyvene and pyreft train interventions. TorchLens ships no trainer; it reads a saved pyreft
LoReFT artifact with `torchlens.bridge.pyreft.read` and applies it as a subspace edit
([Coming from pyvene](from_pyvene.md)).

## Attribution beyond the TorchLens set

`tl.attribution` covers gradients, integrated gradients, occlusion, Grad-CAM, SHAP-style and
perturbation curves.

- Influence functions and adversarial attacks: use Captum.
- TCAV: Captum, or [the TCAV recipe](../recipes/tcav.md), which runs it on TorchLens sites with a
  causal check.
- DeepLift and LIME for text: Captum or inseq. TorchLens provides no DeepLift or LIME for text.

Coming back: `torchlens.bridge.captum` and `torchlens.bridge.inseq` point those tools at a
TorchLens site ([Coming from Captum](from_captum.md), [Coming from inseq](inseq.md)).

## Language-model conveniences across model families

TransformerLens 4 gives canonical hook names across model families, its pretrained loader and
LayerNorm folding. TorchLens resolves TransformerLens hook names on a stock Hugging Face model
and refuses to trace a `TransformerBridge`
([Coming from TransformerLens](from_transformerlens.md)).

## Read one module's output

A forward hook is enough: five lines, no dependency. See
[when five lines of hooks are enough](comparison.md#when-you-need-torchlens-and-when-five-lines-of-hooks-are-enough).

## Compiled, scripted or exported models

TorchLens does not trace `torch.compile`, TorchScript or `torch.export` artifacts. Trace the
eager module they came from.

## Representational similarity and brain scores

rsatoolbox and Brain-Score are where those analyses run. TorchLens hands them features with
provenance ([extract features](../field-guide/extract_features.md),
[the brain-benchmarking seam](../reference/brain_benchmarking_seam.md)).
