# Also in TorchLens

The README shows the core of TorchLens. This page lists the rest, grouped by task, one line per
feature, each linked to the page that documents it. Features whose names may still change are
described in words only; the linked page gives the current spelling.

## Capture

- Records the dict, tuple, list and dataclass structure of a model's inputs and outputs (`capture_container_structure=True`). [Containers](containers.md)
- Lets you register your own container types so TorchLens can walk and rebuild them (`tl.types.register_container`). [Containers](containers.md)
- Saves each op's random state when `save_rng=True` is set, so replays redraw the same dropout masks. [Glossary](reference/glossary.md)
- Records each version of a registered buffer (for example BatchNorm running stats) as a graph node (`Op.is_buffer`). [Buffers](buffers.md)
- Records `torch.func` transforms such as `vmap` and `grad` as single boundary ops (`Op.is_transform`). [Glossary](reference/glossary.md)
- Traces a model built on the meta device, with no weights loaded, and gets its full graph, module tree and every op's shape and dtype. [Weights-free capture](reference/weightsfree_capture.md)
- Captures a multi-step loop that calls one model many times as one episode. [Episode capture](reference/episode_capture.md)
- Keeps each generated token's new KV-cache slices for the decoder layers you choose. [KV cache](guides/kv_cache.md)
- Streams selected activations to a bundle on disk during capture instead of holding them in memory. [Performance](performance.md)
- Merges the captures from each rank of a distributed run into one view, joined at their collective ops (`tl.merge_ranks`). [Glossary](reference/glossary.md)
- Extracts chosen layers over a whole dataset in batches to disk, and picks up where it left off after an interruption. [Extract features](field-guide/extract_features.md)
- Records an event stream instead of a graph, and builds a full trace from it on request (`tl.record`). [Glossary](reference/glossary.md)
- Recovers a recording that a crash or interrupted run left half-written (`tl.recording.recover`). [Glossary](reference/glossary.md)

## Reading

- Runs a dataloader through a model and accumulates running means, covariances, PCA or CKA of any site's activations or gradients (`tl.stats.aggregate`). [Streaming statistics](reference/stats.md)
- Prints a live stats line for a site each time it runs. [Value displays](reference/lovely_reprs.md)
- Selects units that are dead, always saturated, or barely varying across many inputs. [Intervention API](intervention_api.md)
- Selects units by how much they vary across a recurrent layer's repeated passes. [Intervention API](intervention_api.md)
- Selects every op on the paths between two regions, or within a given number of hops of one. [Intervention API](intervention_api.md)
- Draws a seeded random selection of the same size as any selection, as a control. [Intervention API](intervention_api.md)
- Selects the part of an activation along a direction or subspace, such as a probe vector. [Intervention API](intervention_api.md)
- Checks computed receptive fields against gradients from the real model (`tl.receptive_field.verify`). [Receptive and projective fields](receptive_projective_fields.md)
- Builds representational dissimilarity matrices (RDMs) and compares them. [Neuro reference](reference/neuro.md)
- Computes linear CKA with a bootstrap confidence interval over stimuli. [Glossary](reference/glossary.md)
- Shrinks activations with a seeded sparse random projection as they are saved. [Activation transforms](reference/transforms.md)
- Writes a plain-language report of what a trace contains. [Report surfaces](reference/report_family.md)
- Tables time and memory per op, module or call (`trace.profile()`). [Report surfaces](reference/report_family.md)

## Editing

- Replaces a site's output with the output of a module you supply (`tl.splice_module`). [Glossary](reference/glossary.md)
- Removes, or keeps only, an activation's component along a direction (`tl.project_out`, `tl.project_onto`). [Glossary](reference/glossary.md)
- Limits an edit to chosen token positions, including each row's first or last real token under padding. [Glossary](reference/glossary.md)
- Packages edits as one reusable object and saves it to a file. [Intervention API](intervention_api.md)
- Runs one edited trace per value and returns them together as a bundle (`tl.sweep`). [Glossary](reference/glossary.md)
- Sweeps every combination of sites and edits into a labelled grid of results. [Experiments](reference/experiments.md)
- Patches one attention head, or its attention pattern, from a clean run into a corrupted run, exactly, even when the model uses fused attention (SDPA). [Facets](facets.md)

## Gradients and training

- Gives each backward pass its own record, and selects ops by backward pass number (`tl.in_backward_pass`). [Glossary](reference/glossary.md)
- Selects the gradients flowing into or out of each autograd step (`tl.grad_input`, `tl.grad_output`). [Glossary](reference/glossary.md)
- Clips the gradient at any op to a maximum norm during the backward pass (`tl.clip_norm`). [Glossary](reference/glossary.md)
- Finds where NaN or Inf gradients first appeared in a backward pass. [Debug reference](reference/debug.md)
- Finds the first op whose output drifts from a float64 rerun of the same model by more than its dtype explains. [Debug reference](reference/debug.md)
- Runs training-loop checks on a model and optimizer without any capture (`tl.debug.ChecksSession`). [Training checks](reference/checks_kit.md)
- Warns when mixed-precision training never accepts an optimizer step (`tl.debug.Watchdog`). [Training checks](reference/checks_kit.md)
- Streams gradient and activation statistics from a training run to TensorBoard, W&B, MLflow or Comet, with callbacks for Lightning and the Hugging Face Trainer. [Trackers](reference/trackers.md)

## Drawing

- Draws every layer and head's attention pattern on one overview page, in the style of BertViz's model view, with a detail page per layer. [Transformer pictures](reference/tviz.md)
- Animates the repeated passes at one site of a recurrent model (`trace.animate`). [Glossary](reference/glossary.md)
- Shows activation heatmaps or channel grids inside the graph's nodes (`tl.viz.feature_maps`). [Glossary](reference/glossary.md)
- Draws how a site's value distribution changes over training steps. [Training ribbon](field-guide/training_ribbon.md)

## Saving and sharing

- Exports per-op timing as Chrome tracing, speedscope or flamegraph files (`tl.export.chrome_trace`). [Export reference](reference/export.md)
- Exports a timeline of tensor memory over the forward pass (`tl.export.memory_timeline`). [Export reference](reference/export.md)
- Exports saved outputs as an xarray Dataset in Brain-Score's layout (`tl.export.xarray`). [Export reference](reference/export.md)
- Writes a single-file HTML report of a trace that opens offline (`tl.export.html`). [Offline report](guides/offline_report.md)
- Exports a trace's tensors to one safetensors file and reads them back. [Export reference](reference/export.md)
- Fails if a model's structure differs from a pinned hash. [Fingerprint reference](reference/hash.md)

## Tooling

- Checks your environment and model for known trouble spots, such as offloaded or quantized layers (`tl.report.doctor`). [Report surfaces](reference/report_family.md)
- Gives code written against torchextractor a drop-in `Extractor` class (`tl.compat.Extractor`). [Coming from torchextractor](migration/from_torchextractor.md)
- Registers your own output decoder, such as class names for a custom classifier (`tl.postprocessing.register`). [Glossary](reference/glossary.md)
- Accepts TransformerLens names like `hook_pattern` for facets, if you turn it on (`tl.facets.enable_transformer_lens_aliases`). [Facets](facets.md)
- Renders traces as Treescope cards in notebooks. [Treescope cards](guides/treescope_cards.md)
- Raises named error types with specific codes and remedies (`tl.errors`), so code can catch one failure and tell it from the others. [Capture outcomes](reference/capture_outcomes.md)
- Ships a Claude Code plugin with a TorchLens skill that tells agents which call to use. [For AI agents](for-ai-agents.md)
