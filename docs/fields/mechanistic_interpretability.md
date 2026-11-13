# TorchLens for mechanistic interpretability

Mechanistic interpretability needs three things from a model's activations: an exact address for every site (a head's query, a block's residual stream, one MLP neuron), the ability to edit that site and measure the effect on the output, and enough speed to sweep the edit over many sites and prompts. TorchLens records every operation of a forward pass with a stable label, lets you patch, ablate or steer any of them with the same call whether you run once or through generation, and hands the result to the SAE, probing and steering libraries you already use.

The lists below are in the order most people want them: see one site, edit it, then scale up. Every link is to a file in this repository.

## Start here

- [Tutorial notebooks](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/): twelve short notebooks on laptop-sized models. Notebook 05 is the first intervention, 07 compares runs in a bundle, 08 is the semantic shortcuts ("head 3's query") and 10 has a mechanistic-interpretability branch.
- [The decoder running example](../field-guide/running_examples.md#the-decoder-on-the-ioi-prompt): the small decoder on the IOI prompt that every Field guide page starts from.
- [See the graph and inspect any operation](../field-guide/see_and_inspect.md): a picture of every operation the forward ran, and the output, module and arguments of any one of them.
- [Intervene on a site and measure the effect](../field-guide/intervene.md): ablation, the hook check, and the same edit bound live for sweeps and `generate()`.
- [Hugging Face tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/huggingface_tutorial.ipynb): capture a transformer from `transformers`, browse its structure, read activations from any point, draw the graph.

## Patching, ablation and steering

- [Intervention explainers](../intervention_explainers.md): the three ways TorchLens applies one edit (replay, live hook, rerun) and which question each answers.
- [Intervention API reference](../intervention_api.md): selectors, edit helpers, hooks, replay and bundles.
- [Stochastic and population edits](../stochastic_edits.md): mean ablation, resampling ablation and causal scrubbing from one population noun and one edit.
- [Editing generation and the KV cache](../guides/kv_cache.md): what "intervene during generation" can mean and how each is spelled.
- [Performance: choosing a fast path](../guides/fast_paths.md): full capture, replay and capture-free live hooks differ in cost by two orders of magnitude; pick before a sweep.
- [Paired-prompt activation patching notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/workflows/paired_prompt_patching.ipynb): patches each of GPT-2 small's 144 attention heads from a France prompt into an Italy prompt and maps which heads carry " Paris".
- [Causal trace notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/workflows/causal_trace_recipe.ipynb): a full clean-versus-corrupt causal trace with a per-site heatmap.
- [Ablation grid notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/workflows/ablations_grid.ipynb): an ablation sweep over discovered sites.
- [Steering notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/workflows/steering.ipynb): build a steering vector and apply it at replay time.
- [Intervention worked examples](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/intervention/): eighteen short scripts, from the first five minutes to per-position steering, bundles of three or more prompts, SAE attachment and publishing a portable `.tlspec`.

## Circuits, attribution and lenses

- [The mech-interp kit](../reference/mechinterp.md): residual decomposition, direct logit contributions, per-head results, the component edge graph and patching grids.
- [ACDC circuit discovery as a driver loop](../field-guide/acdc_ioi.md): edge pruning on IOI written as a loop over TorchLens calls.
- [A tuned lens in TorchLens's logit lens](../field-guide/tuned_lens.md) and the [train a tuned lens notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/train_tuned_lens_recipe.ipynb).
- [A Jacobian lens in a few lines](../field-guide/jacobian_lens.md) and the [Jacobian lens notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/jacobian_lens_recipe.ipynb): what a block's residual vector is disposed to make the model say, fitted on GPT-2 small and compared with the logit lens.
- [Per-step attribution for generated text](../field-guide/generation_step_attribution.md): attribute each generated token to the tokens before it.
- [Attribution graphs with transcoders notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/attribution_graph_transcoders_recipe.ipynb): GPT-2 small with public transcoders spliced in.
- [Function vectors notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/function_vectors_recipe.ipynb): score every head's causal indirect effect and sum the top heads into a task vector.
- [DAS causal basis notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/das_causal_basis_recipe.ipynb): distributed alignment search through a bound hook, tested against a random basis.
- [Measure faithfulness notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/measure_faithfulness_recipe.ipynb): ranked removal, output drop and locality checks on an attribution.
- [EAP-IG circuit notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/eap_ig_circuit_recipe.ipynb): edge attribution patching with integrated gradients on GPT-2 small's IOI task, with the circuit's faithfulness replayed by `patch_edges`.
- [Logit lens notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/logit_lens_recipe.ipynb): every GPT-2 small block's next-token prediction, with TorchLens and with plain PyTorch hooks side by side.
- [Vision logit lens notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/vision_logit_lens_recipe.ipynb): per-block class votes of a ViT classifier and CLIP's open-vocabulary readout of every vision block.
- [LEACE concept erasure notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/leace_concept_erasure_recipe.ipynb): scrub grammatical number from every GPT-2 small block with bound `tl.when` rules and watch subject-verb agreement collapse.
- [Layer-wise relevance propagation for transformers](../field-guide/lrp_transformers_lxt.md) and the [epsilon-LRP litmus test](../field-guide/lrp_epsilon_litmus.md).
- [Conditional steering with a per-token audit](../field-guide/conditional_steering.md): steer only when a condition direction crosses a threshold.
- [Transformer pictures](../reference/tviz.md): token-labelled attention heatmaps and grids, the attention atlas, coloured tokens.

## Sparse autoencoders

- [SAE splicing tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/sae_splice_tutorial.ipynb): test whether an SAE's features are causally used, not only whether they reconstruct.
- [Pretrained SAEs: SAELens loads, TorchLens runs](../field-guide/pretrained_saes.md): Gemma Scope and every other published SAE in the SAELens registry.
- [Train an SAE on TorchLens activations](../sae_training_handoff.md) and [SAE trainers: dictionary_learning and sparsify](../field-guide/sae_trainers.md): extract, train elsewhere, splice back and measure.
- [SAE latents and top examples for SAEDashboard and delphi](../field-guide/sae_latent_export.md).
- [Model diffing with a crosscoder](../field-guide/sae_crosscoder_diff.md): one dictionary over two models' activations.
- [CLIP SAEs with Prisma](../field-guide/prisma_clip_sae.md).

## Probes and representations

- [Linear probes](../field-guide/linear_probes.md): leakage-safe splits, an obfuscation check and a causal check.
- [Probe accuracy by layer notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/probe_accuracy_by_layer_recipe.ipynb): accuracy against depth with an untrained-network control.
- [Contrastive direction notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/contrastive_direction_recipe.ipynb): a positive-minus-negative direction with replay-time steering.
- [Streaming CKA notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/streaming_cka.ipynb): compare representations on streamed data in constant memory.
- [Streaming statistics](../reference/stats.md): dataset-level statistics of activations or gradients, one batch at a time.
- [Training external machinery against TorchLens surfaces](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/external_training_tutorial.ipynb): TorchLens ships no probe fitters or SAE loops; this is how your own trainer plugs in.

## Replications on GPT-2

- [Paper replications](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/replications/): scripts that state one published claim, measure it and print a verdict against a rule fixed before the first run: induction heads (Olsson et al. 2022), the IOI name movers (Wang et al. 2022) and causal tracing (Meng et al. 2022).
- [The same three lines on any architecture](../gallery/any_architecture.md): the trace, draw and edit calls on six architecture families.

## Experiments at scale

- [Experiments: lineage, vary, site sweeps and grids](../reference/experiments.md): the bundle as the record of an experiment.
- [Episode capture](../reference/episode_capture.md) for multi-step generation loops.
- [Facets](../facets.md) and the [facets tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/facets_tutorial.ipynb): one access surface for heads, gates and other structural views of an activation, including TransformerLens aliases.
- [Fast activation recording](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/fastlog_tutorial.ipynb): the sparse recording path for when you know what to keep before the forward runs.

## Connectors and migration

- [Connectors to other tools](../bridges.md): SAELens, steering-vectors, repeng, dialz, nnsight, inseq, LIT, Captum, SHAP, treescope and the rest, one entry each.
- Coming from [TransformerLens](../migration/from_transformerlens.md) (and the [migration notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/workflows/transformer_migration.ipynb)), [nnsight](../migration/from_nnsight.md), [pyvene](../migration/from_pyvene.md), [SAELens](../migration/saelens.md), [baukit](../migration/baukit.md), [AutoCircuit](../migration/autocircuit.md), [RepE](../migration/repe.md) or plain [forward hooks](../migration/from_hooks.md).
- [Which tool for this job?](../migration/comparison.md) and [where to go instead](../migration/where_to_go_instead.md): the jobs another tool does better.
- [Glossary](../reference/glossary.md): the definition of every public name used above.
