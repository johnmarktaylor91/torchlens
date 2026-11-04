# TorchLens quick recipes

Read these in this order if you are new to TorchLens:

1. [`pluck.ipynb`](pluck.ipynb) - Pluck your first activation in three lines.
2. [`find_first_nan.ipynb`](find_first_nan.ipynb) - Find the first NaN or Inf-producing operation in two lines.
3. [`visualize.ipynb`](visualize.ipynb) - Render a model graph and learn the layer labels.
4. [`extract_activations.ipynb`](extract_activations.ipynb) - Extract several layers into a dict of tensors.
5. [`extract_dataset.ipynb`](extract_dataset.ipynb) - Extract named-layer activations over a whole dataset in batches, in memory or on disk.
6. [`intervention.ipynb`](intervention.ipynb) - Zero-ablate one layer and replay downstream computation.
7. [`save_load.ipynb`](save_load.ipynb) - Round-trip a torch materialized-payload `.tlspec` bundle.
8. [`cog_neuro_rdm.ipynb`](cog_neuro_rdm.ipynb) - Build a compact model RDM for cognitive neuroscience workflows.

One script sits beside the notebooks: [`paris_vs_rome.py`](paris_vs_rome.py) is the
README's combined example made runnable. It patches each GPT-2 small attention head from a
France prompt into an Italy prompt and finds the two heads that make the model answer " Paris".
It downloads GPT-2 small (about 550 MB) on first use and takes about three minutes on a CPU.

Each notebook is deterministic, CPU-friendly, and designed to run end-to-end with papermill in under 30 seconds.
