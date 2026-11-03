# TorchLens Recipes

These notebooks are self-contained intervention recipes. They use small local
models only, so they can run in CI or with `papermill` without network
downloads.

| Recipe | Difficulty | Description |
| --- | --- | --- |
| [`causal_trace_recipe.ipynb`](causal_trace_recipe.ipynb) | Intermediate | Clean-versus-corrupt activation patching, per-site sweeps with `trace.find_sites`, output-divergence scoring, and `tl.viz.causal_trace_heatmap`. |
| [`contrastive_direction_recipe.ipynb`](contrastive_direction_recipe.ipynb) | Beginner | RepEng-style positive-minus-negative activation direction with replay-time steering. |
| [`measure_faithfulness_recipe.ipynb`](measure_faithfulness_recipe.ipynb) | Intermediate | ROAD-style ranked removal, output-drop, locality checks, and `tl.validate(scope="intervention")`. |
| [`trainable_intervention_recipe.ipynb`](trainable_intervention_recipe.ipynb) | Advanced | Gradient-based learning of a trainable activation intervention parameter. |
| [`generation_step_iterator_recipe.ipynb`](generation_step_iterator_recipe.ipynb) | Intermediate | Per-step hook planning for generation loops that mirror `model.generate()`. |
| [`train_tuned_lens_recipe.ipynb`](train_tuned_lens_recipe.ipynb) | Intermediate | Train a tuned lens from a disk-mode `tl.extract_dataset` artifact and apply it through `logit_lens(lens=...)`, which checks the recorded model identity. |
| [`attribution_graph_transcoders_recipe.ipynb`](attribution_graph_transcoders_recipe.ipynb) | Advanced | Attribution graph on GPT-2 small with the public Dunefsky et al. transcoders (Ameisen et al. 2025; Dunefsky et al. 2024): splice the transcoders in as a replacement model, build the graph with `tl.attribution.build_graph`, read what writes the answer logit, check the top edge by ablation and export for Neuronpedia's viewer. Downloads GPT-2 and the transcoders (about 2.2 GB) on first run. |
