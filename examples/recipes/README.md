# TorchLens Recipes

These notebooks are self-contained intervention recipes. They use small local
models only, so they can run in CI or with `papermill` without network
downloads.

| Recipe | Difficulty | Description |
| --- | --- | --- |
| `causal_trace_recipe.ipynb` | Intermediate | Clean-versus-corrupt activation patching, per-site sweeps with `tl.sites`, output-divergence scoring, and `tl.viz.causal_trace_heatmap`. |
| `contrastive_direction_recipe.ipynb` | Beginner | RepEng-style positive-minus-negative activation direction with replay-time steering. |
| `measure_faithfulness_recipe.ipynb` | Intermediate | ROAD-style ranked removal, output-drop, locality checks, and `tl.validate(scope="intervention")`. |
| `trainable_intervention_recipe.ipynb` | Advanced | Gradient-based learning of a trainable activation intervention parameter. |
| `generation_step_iterator_recipe.ipynb` | Intermediate | Per-step hook planning for generation loops that mirror `model.generate()`. |
