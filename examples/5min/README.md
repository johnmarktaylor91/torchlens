# TorchLens 5-Minute Notebooks

Read these in this order if you are new to TorchLens:

1. [`peek.ipynb`](peek.ipynb) - Peek your first activation in three lines.
2. [`find_first_nan.ipynb`](find_first_nan.ipynb) - Find the first NaN or Inf-producing operation in two lines.
3. [`visualize.ipynb`](visualize.ipynb) - Render a model graph and learn the layer labels.
4. [`extract_activations.ipynb`](extract_activations.ipynb) - Extract several layers into a dict of tensors.
5. [`intervention.ipynb`](intervention.ipynb) - Zero-ablate one layer and replay downstream computation.
6. [`save_load.ipynb`](save_load.ipynb) - Round-trip a saved TorchLens `.tlspec` bundle.
7. [`cog_neuro_rdm.ipynb`](cog_neuro_rdm.ipynb) - Build a compact model RDM for cognitive neuroscience workflows.

Each notebook is deterministic, CPU-friendly, and designed to run end-to-end with papermill in under 30 seconds.
