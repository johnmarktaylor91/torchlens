# Comparable tools, checked November 2026

Where TorchLens sits next to the interpretability and model-inspection tools people ask
about, with each tool's status as of the check date. Every row carries the date it was
checked and the source consulted; the wording rules that bind comparative sentences in this
repository live in `docs/migration/_claims.json`, and the per-tool migration tables are
linked from the Tool column.

## Status at a glance

| Tool | What it is | Status (checked 2026-11-03) | Next to TorchLens | Source |
| --- | --- | --- | --- | --- |
| [TransformerLens](from_transformerlens.md) 4.x | Transformer-specific hook points and the ActivationCache analysis kit; `TransformerBridge` wraps stock Hugging Face models as the main model class from 4.0 | Active: v4.0.0 released 2026-09-21, v4.3.0 released 2026-10-23 | TorchLens traces the stock Hugging Face model and resolves TransformerLens names on it; it refuses to trace a `TransformerBridge` (see [from_transformerlens.md](from_transformerlens.md)) | github.com/TransformerLensOrg/TransformerLens releases |
| [nnsight](from_nnsight.md) 0.8 | Source-local tracing with remote execution (NDIF), vLLM and tensor-parallel back ends, quantized models, diffusers pipelines, `.source` for module-less operations | Active: 0.8 announced 2026-09-09, v0.8.2 released 2026-11-02 | For remote, serving-scale or multi-GPU capture, nnsight is the tool; TorchLens records every executed operation of a local eager forward and verifies the recording by replay | nnsight.net/blog/2026/09/09/nnsight-08 |
| interp-engine (Neuronpedia) | A vLLM-based capture and steering engine with standard capture points, running Neuronpedia's own service | Active: announced 2026-08-31 | Serving-scale capture at a fixed set of points. An independent review (2026-09-02) found the earlier server feeding an SAE the raw MLP output instead of the normed one with nothing crashing; that bug class is what TorchLens's per-op replay check exists to catch on the eager twin | neuronpedia.org/blog/interp-engine; gkoreli.com/oss-radar-06-interp-engine |
| vllm-lens (UK AI Security Institute) | Residual-stream activation extraction and steering vectors inside vLLM | Active: repository pushed 2026-10-13 | Serving-scale extraction and steering; TorchLens is the eager-model side where every site is visible and an effect can be verified before it is served | github.com/UKGovernmentBEIS/vllm-lens |
| EasySteer | Steering inside vLLM, including router logits and head outputs at serving speed | Active: v0.29.0 released 2026-09-10 | Build and check a steering vector in TorchLens, serve it in EasySteer (see [repe.md](repe.md)) | github.com/ZJU-REAL/EasySteer releases |
| EasyEdit2 | The steering framework of the EasyEdit project: it generates steering vectors for a language model and applies them at inference, without changing weights (EMNLP 2025 system demonstration) | Active: no tagged releases; repository pushed 2026-10-04 | Build and check a steering vector in TorchLens (see [entropy steering](../field-guide/entropy_steering.md)), apply it in EasyEdit2; EasyEdit's knowledge-editing methods stay outside TorchLens ([non-goals](../non_goals.md)) | arXiv:2504.15133; github.com/zjunlp/EasyEdit (README_2.md) |
| VLM-Lens | Extraction of internal representations from vision-language models (EMNLP 2025 system demonstration) | Beta; repository last pushed 2026-04-25 | A vision-language model is an ordinary PyTorch model to TorchLens: the same capture, selectors and replay check apply, with no model-family list | aclanthology.org/2025.emnlp-demos.68; github.com/compling-wat/vlm-lens |
| [pyvene](from_pyvene.md) / pyreft | Interventions as durable, serializable objects; trained interventions (DAS, LoReFT) | Idle: last pushes 2026-03-06 (pyvene) and 2026-03-05 (pyreft); last pyvene release v0.1.8, 2025-05-26 | TorchLens persists intervention recipes as `.tlspec/` directories and reads pyreft artifacts through `tl.bridge.pyreft`; it ships no trainer | github.com/stanfordnlp/pyvene; github.com/stanfordnlp/pyreft |
| [penzai](penzai.md) | Named-axis JAX model trees with functional surgery | Archived: read-only since its last push on 2025-06-22 | Existing penzai code keeps working; new JAX work can use the TorchLens JAX backend or another home | github.com/google-deepmind/penzai |
| treescope | The pretty-printer that grew out of penzai | Idle: last release v0.1.10 on 2025-08-08, last push 2026-06-18 | The opt-in `torchlens.bridge.treescope` upgrade still works; TorchLens's own notebook cards need no treescope (see [treescope cards](../guides/treescope_cards.md)) | github.com/google-deepmind/treescope |
| [SAELens](saelens.md) | The registry and loader for published sparse autoencoders | Active: v6.54.14 released 2026-11-02 | SAELens loads the SAE, TorchLens runs the model and splices it in (see [pretrained SAEs](../field-guide/pretrained_saes.md)) | github.com/jbloomAus/SAELens releases |

"Idle" means no push to the public repository in the eight months before the check date, not
that the package is broken or unmaintained; "archived" is the repository's own read-only flag.

## Serving-scale capture

Serving engines (vLLM, SGLang) run fused kernels, paged KV caches and CUDA graphs outside eager
PyTorch, so TorchLens has nothing to record there and `tl.report.doctor` says so with its
`serving_engine` row. For capture or steering at serving speed, the tools above are the
destinations: nnsight 0.8's vLLM back end, Neuronpedia's interp-engine, vllm-lens for
extraction and steering, EasySteer for steering. TorchLens's part is the eager twin: load the
same checkpoint as a Hugging Face model, capture it with every operation visible, verify the
recording by replay, and carry the checked site or vector to the serving tool
([deployment envelope](../reference/deployment_envelope.md#serving-engines-capture-the-hugging-face-twin)).

## Discovery and monitoring

Sparse autoencoders are a discovery instrument: they propose features and let you inspect a
model's activations in a basis you did not choose. Monitoring a deployed model for a known
behaviour is a different job, and the field's production monitors use trained probes on the
same activations rather than SAE latents (Goodfire, "Probe monitors 101", 2026-09-09;
arXiv:2610.12445, 2026-10-08). TorchLens serves both: `torchlens.bridge.sae.splice` runs a
published SAE against a captured site and reports what the splice did, and `tl.extract_dataset`
writes the activations a probe is trained on, with provenance. TorchLens ships no probe trainer;
the training loop is yours (see [non-goals](../non_goals.md)).

## What this page does not claim

Reaching module-less operations, diffusers pipelines, quantized models and vision-language
models is shared ground: nnsight 0.8 reaches all of them, with `.source`, `DiffusionModel` and
dtype quantization, and TorchLens claims nothing exclusive there. TorchLens's distinct claim is
narrower and stays the same across every row: the full executed graph of a local eager
forward, recorded with metadata, replayed and checked against the real forward, saved and
diffable, with typed refusals where capture cannot be trusted.
