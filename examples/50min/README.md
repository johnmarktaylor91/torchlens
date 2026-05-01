# TorchLens 50-Minute Notebooks

These notebooks are longer workflow recipes for users who want the full shape of an analysis rather than a minimal snippet. They are deterministic, CPU-friendly, checked in with outputs, and intended to run end-to-end with `papermill` without network access.

| Notebook | Workflow |
| --- | --- |
| [`causal_trace_recipe.ipynb`](causal_trace_recipe.ipynb) | Full clean-vs-corrupt causal trace on a tiny transformer-style model. |
| [`iou_heatmap.ipynb`](iou_heatmap.ipynb) | Attention-mask IOU heatmap across heads. |
| [`transformer_migration.ipynb`](transformer_migration.ipynb) | TransformerLens cache and hook migration patterns using TorchLens selectors. |
| [`intervention_fork_replay.ipynb`](intervention_fork_replay.ipynb) | Fork, replay, and bundle comparison workflow. |
| [`cog_neuro_extraction.ipynb`](cog_neuro_extraction.ipynb) | Per-layer cognitive neuroscience extraction and offline Brain-Score-style evaluation. |
| [`paired_prompt_patching.ipynb`](paired_prompt_patching.ipynb) | Single-site clean-vs-corrupt prompt patching session. |
| [`steering.ipynb`](steering.ipynb) | Steering vector construction and replay-time application. |
| [`ablations_grid.ipynb`](ablations_grid.ipynb) | Ablation sweep over resolved `tl.sites`-style targets. |
| [`custom_hooks.ipynb`](custom_hooks.ipynb) | Custom hook composition with replay. |
