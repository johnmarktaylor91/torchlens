# TorchLens quick recipes

Read these in this order if you are new to TorchLens:

1. [`pluck.ipynb`](pluck.ipynb) - Get one layer's output as a tensor.
2. [`find_first_nan.ipynb`](find_first_nan.ipynb) - Find the operation that first produces a NaN or Inf.
3. [`visualize.ipynb`](visualize.ipynb) - Draw the model graph and read the layer labels.
4. [`extract_activations.ipynb`](extract_activations.ipynb) - Extract several layers into a dict of tensors.
5. [`extract_dataset.ipynb`](extract_dataset.ipynb) - Extract named layers over a whole dataset in batches, in memory or on disk.
6. [`intervention.ipynb`](intervention.ipynb) - Zero one layer in a copy of a trace and see the output change.
7. [`save_load.ipynb`](save_load.ipynb) - Save a trace to a `.tlspec` folder and load it back.
8. [`cog_neuro_rdm.ipynb`](cog_neuro_rdm.ipynb) - Build an RDM from one layer's activations.

The notebooks use small CPU models and download nothing. Install TorchLens as in the
[main README](../../README.md#install); `visualize.ipynb` also needs the Graphviz application.

One script sits beside the notebooks: [`paris_vs_rome.py`](paris_vs_rome.py) is the
README's combined example made runnable. It patches each GPT-2 small attention head from a
France prompt into an Italy prompt and finds the two heads that make the model prefer " Paris"
over " Rome". It needs `pip install torchlens[hf]`, downloads GPT-2 small (about 550 MB) on
first use and takes a few minutes on a CPU.
