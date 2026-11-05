# TorchLens workflows

Longer recipes that show the full shape of an analysis; the short single-task notebooks are in [`examples/quick/`](../quick/). These are deterministic, run on a CPU without network access, and are checked in without outputs (run them to see the results). The two exceptions are `paired_prompt_patching.ipynb`, which downloads GPT-2 small, and `multi_call_bundle.ipynb`, which downloads two small pretrained models; both are checked in with their outputs.

| Notebook | Workflow |
| --- | --- |
| [`causal_trace_recipe.ipynb`](causal_trace_recipe.ipynb) | Full clean-vs-corrupt causal trace on a tiny transformer-style model. |
| [`iou_heatmap.ipynb`](iou_heatmap.ipynb) | Attention-mask IOU heatmap across heads. |
| [`transformer_migration.ipynb`](transformer_migration.ipynb) | TransformerLens cache and hook migration patterns using TorchLens selectors. |
| [`intervention_fork_replay.ipynb`](intervention_fork_replay.ipynb) | Fork, replay, and bundle comparison workflow. |
| [`cog_neuro_extraction.ipynb`](cog_neuro_extraction.ipynb) | Per-layer cognitive neuroscience extraction and offline Brain-Score-style evaluation. |
| [`paired_prompt_patching.ipynb`](paired_prompt_patching.ipynb) | Activation patching in pretrained GPT-2 small: patch each of the 144 attention heads from a France prompt into an Italy prompt and map which heads carry " Paris" (the README's experiment, step by step). |
| [`steering.ipynb`](steering.ipynb) | Steering vector construction and replay-time application. |
| [`ablations_grid.ipynb`](ablations_grid.ipynb) | Ablation sweep over `trace.find_sites` targets. |
| [`custom_hooks.ipynb`](custom_hooks.ipynb) | Custom hook composition with replay. |
| [`multi_call_bundle.ipynb`](multi_call_bundle.ipynb) | A draft-and-critique pipeline recorded as one Bundle: the token links from the draft into the critique's prompt, an edit to the draft followed into the critique, and the pipeline drawn as a graph of calls. The first run downloads GPT-2 small and DistilGPT2 (about 900 MB). |
