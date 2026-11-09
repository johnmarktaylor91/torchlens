# TorchLens for debugging and training networks

When a network misbehaves, the question is usually "which operation first went wrong, and what fed it": the first NaN, the layer whose activations drifted, the branch that ran when it should not have, the tensor that is eating memory. TorchLens records every operation of a forward pass with its inputs, outputs, shapes, bytes, source line and module, so these questions become lookups on a trace instead of a hook-and-print session, and the same records can be sampled during training to watch distributions drift over time.

The lists below start with the one-off questions and move to monitoring a whole training run. Every link is to a file in this repository.

## Start here

- [Tutorial notebooks](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/): twelve short notebooks on laptop-sized models. Notebook 01 captures a forward, 02 reads a number out, 03 draws the graph, 04 keeps memory in check and 06 reads gradients.
- [Find the first NaN notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/find_first_nan.ipynb): the first NaN- or Inf-producing operation, in two lines.
- [See the graph and inspect any operation](../field-guide/see_and_inspect.md): a picture of every operation the forward ran, and the output, module and arguments of any one of them, including operations no module returns.
- [Model summaries](../reference/summary.md): a one-call, metadata-only pass that lists every layer with shapes, parameter counts and cost.
- [Debug utilities example script](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/debug_utilities.py): a tiny model with one unstable operation and the calls that find it.

## Locate a fault

- [tl.debug reference](../reference/debug.md): locate non-finite saved tensors, walk dependencies, compare runs, rank cost, when the graph itself is the diagnostic.
- [Extracting source code with TorchLens](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/source_code_extraction.ipynb): the call stack and the actual source line of every operation.
- [Conditional branch attribution](../conditional_attribution.md): which Python `if` branch each operation ran under, so you can see the control flow the forward actually took.
- [Containers](../containers.md): the input and output container structure of each operation, for faults inside tuples, dicts and lists.
- [Buffers](../buffers.md): BatchNorm running statistics and other registered buffers, recorded alongside activations.
- [Compare two runs in a bundle](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/07_compare_runs.ipynb): the same model before and after a change, diffed operation by operation.

## Gradients and the backward pass

- [Gradients tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/06_gradients.ipynb): what drives one unit, read from the trace.
- [Backward capture](../backward.md) and the [backward tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/backward_tutorial.ipynb): per-layer output gradients, the autograd graph connected back to forward layers, and validation of the backward record.
- [The gradient contract](../gradient_contract.md): what TorchLens promises about saved gradients.

## Memory and speed

- [Memory debugging: which question, which tool](../memory_debugging.md): which tool answers which memory question, and where TorchLens's per-operation byte accounting fits beside the PyTorch allocator tools.
- [Keep what you need](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/04_keep_what_you_need.ipynb): save only the activations you want when memory or time runs out.
- [Performance guide](../performance.md) and [choosing a fast path](../guides/fast_paths.md): the capture modes and what each costs.
- [Profilers to use alongside TorchLens](../migration/profilers_alongside.md): the PyTorch profiler, Perfetto and friends measure what TorchLens does not.
- [Native PyTorch tools: what each is authoritative for](../native-torch.md): when the built-in tool is the right answer.

## Monitor training

- [Monitor training](../monitor_training.md): watch how the distribution at every site drifts over training, find where it broke, and keep the history as a portable, mergeable artifact.
- [A percentile-band training ribbon](../field-guide/training_ribbon.md): the ribbon plot of one site's distribution over steps, from watch histograms.
- [Training sanity checks (the checks kit)](../reference/checks_kit.md): range probes and sessions that flag the usual training faults.
- [Trackers: TensorBoard and wandb](../reference/trackers.md): feed TorchLens statistics to the tracker you already log to.
- [Training on saved activations](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/training_tutorial.ipynb): when a saved activation is an input to a loss that will feed `backward()`.
- [Train a truncated backbone through trace.run](../field-guide/trainable_backbone.md): cut a pretrained network at a stage and fine-tune the prefix, from one trace.
- [Weight spectra with WeightWatcher](../field-guide/weightwatcher.md): per-layer power-law metrics of the weight matrices.
- [Streaming statistics](../reference/stats.md): dataset-level statistics of activations or gradients in constant memory.

## Save, share and export

- [Save and load a trace notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/save_load.ipynb): round-trip a trace as a `.tlspec` bundle.
- [tl.export reference](../reference/export.md): static graph files, performance-viewer formats, tabular files and tracker objects.
- [The Netron export](../reference/netron_export.md) and [Google Model Explorer export](../reference/model_explorer.md): open the captured graph in a viewer.
- [The single-file offline HTML report](../guides/offline_report.md): one file to send to a colleague.
- [Change a model's skeleton, keep your analyses aligned](../skeleton_change_recipe.md): what to do with saved analyses after you swap or wrap a block.

## Connectors and migration

- [Connectors to other tools](../bridges.md): the PyTorch profiler, Chrome tracing and Perfetto, TensorBoard, Weights and Biases, MLflow and Aim, Netron, treescope, lovely-tensors and the rest, one entry each.
- Coming from [torchinfo](../migration/from_torchinfo.md), [torchsnooper](../migration/coming_from_torchsnooper.md), [torchview](../migration/from_torchview.md) or plain [forward hooks](../migration/from_hooks.md).
- [Glossary](../reference/glossary.md): the definition of every public name used above.
