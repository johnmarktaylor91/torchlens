# Using TorchLens with other tools

The bridges that ship in `torchlens.bridge`, sorted by the direction the data moves. A bridge is
imported on its own (`from torchlens.bridge import captum`) and needs the other tool installed;
nothing here is imported by `import torchlens`. Checked 2026-11-03.

<!-- JMT review: plain factual prose, written for review. -->

## Into TorchLens: another tool's artifact runs inside a trace

| From | Bridge | What comes in | Page |
| --- | --- | --- | --- |
| pyreft | `torchlens.bridge.pyreft.read` | A saved LoReFT intervention, applied as a TorchLens subspace edit | [Coming from pyvene](from_pyvene.md) |
| SAELens | `torchlens.bridge.sae_lens.encode`, `reconstruct` | A published sparse autoencoder, run on a captured site | [pretrained SAEs](../recipes/pretrained_saes.md) |
| any SAE library | `torchlens.bridge.sae.splice` | An SAE's reconstruction spliced back into the forward, with what the splice did | [pretrained SAEs](../recipes/pretrained_saes.md) |
| Hugging Face Hub | `torchlens.bridge.huggingface_hub.from_hub` | A `.tlspec` bundle someone published | [glossary](../reference/glossary.md) |
| Hugging Face transformers | `torchlens.bridge.transformers.trace_text`, `trace_image`, `trace_multimodal` | A model and its processor, traced on raw text or images | [quickstart](../quickstart.md) |

## Out of TorchLens: a trace or its features feed another tool

| To | Bridge | What goes out | Page |
| --- | --- | --- | --- |
| rsatoolbox | `torchlens.bridge.rsatoolbox.dataset` | Per-site features as an rsatoolbox `Dataset`, rows tied to stimulus ids | [extract features](../field-guide/extract_features.md) |
| xarray and Brain-Score assemblies | `torchlens.bridge.xarray.data_array`, `neuroid_assembly` | Per-site presentation-by-neuroid arrays | [NeuroAI journey](../neuroai/journey.md) |
| Brain-Score | `torchlens.bridge.brain_score.activations_extractor`, `per_layer` | An activations extractor over TorchLens sites | [brain-benchmarking seam](../reference/brain_benchmarking_seam.md) |
| Captum | `torchlens.bridge.captum.attribute`, `layer` | A TorchLens site as a Captum layer target | [Coming from Captum](from_captum.md) |
| inseq | `torchlens.bridge.inseq.attribute` | Text attribution on a TorchLens-addressed site | [Coming from inseq](inseq.md) |
| SHAP | `torchlens.bridge.shap.explain` | A site as the explained output | [Coming from Captum](from_captum.md) |
| pytorch-grad-cam | `torchlens.bridge.pytorch_grad_cam.cam`, `layer` | A TorchLens site as the CAM target layer | [Grad-CAM with torch-cam](../recipes/torchcam.md) |
| steering-vectors, repeng, dialz | `torchlens.bridge.steering_vectors.from_direction`, `torchlens.bridge.repeng.from_direction`, `torchlens.bridge.dialz.from_direction` | A TorchLens `Direction` as that library's steering object | [Coming from RepE](repe.md) |
| Hugging Face Hub | `torchlens.bridge.huggingface_hub.push_to_hub` | A `.tlspec` bundle, published | [glossary](../reference/glossary.md) |
| treescope | `torchlens.bridge.treescope.register` | TorchLens objects rendered in treescope | [treescope cards](../guides/treescope_cards.md) |
| torch.profiler | `torchlens.bridge.profiler.join` | Profiler events joined to TorchLens operations | [profilers alongside](profilers_alongside.md) |
| depyf | `torchlens.bridge.depyf.dump` | Decompiled source next to the trace | [glossary](../reference/glossary.md) |
| agents over MCP | `torchlens.bridge.mcp.main` | The agent registry served over stdio | [TorchLens for AI coding agents](../for-ai-agents.md) |

## Both directions

| With | How |
| --- | --- |
| Hugging Face attention | `torchlens.bridge.hf_attention.register_attention` registers TorchLens as an attention implementation, so attention internals are recorded ops in a model that otherwise calls a fused kernel |
| Plain tensors and files | Every saved activation is a `torch.Tensor`; `trace.save(path)` and `tl.load(path)` move a whole recording, and any tool that reads tensors reads the rest |
