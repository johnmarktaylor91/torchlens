# TorchLens for teaching how networks work

Teaching needs a picture of the whole computation, not a module list: every operation the forward pass actually ran, in order, with the shapes flowing between them, so a student can follow a residual connection, an attention head or a recurrent loop from input to output. TorchLens draws that picture from one call on any PyTorch model, lets a student click into any single operation to see its inputs, outputs and source line, and does the same for a 1943 threshold-neuron network and a modern transformer alike.

The lists below start with drawing and inspecting, then move to the tutorial notebooks and the Model Menagerie. Every link is to a file in this repository or to the Menagerie site.

## Start here

- [See the graph and inspect any operation](../field-guide/see_and_inspect.md): one call draws every operation the forward ran, including ones no module returns; one lookup gives any operation's output, module and arguments.
- [Visualize notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/visualize.ipynb): render a model graph and learn to read the layer labels.
- [Tutorial notebooks](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/): twelve short notebooks on laptop-sized models, each one rung up from the last. Notebook 01 is "capture and look", 02 reads one activation like a dict, 03 draws the graph when the summary is too long to read. They run on a laptop CPU in about a minute each, or in the browser on Colab, so they work as lab exercises.
- [TorchLens in 10 minutes](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/torchlens_in_10_minutes.ipynb): trace, intervene, run a backward pass, see control flow drawn as IF, THEN and ELSE, and compare two traces.

## Drawing the graph

- [How TorchLens groups operations into layers](../guides/grouping.md): why `linear_1` with passes `linear_1:1` and `linear_1:2` is the same layer run twice, the thing students most often need explained about recurrent and weight-tied models.
- [Smart-collapse visual reference](../reference/collapse.md): how a large graph folds into readable blocks, with rendered examples at each level.
- [Premade views](../reference/lenses.md): `trace.draw(view=...)` views that each answer one question about the model.
- [Transformer pictures](../reference/tviz.md): token-labelled attention heatmaps and grids and the attention atlas, for the attention lecture.
- [Notebook cards and the treescope bridge](../guides/treescope_cards.md): interactive cards for a notebook walkthrough.
- [The Netron export](../reference/netron_export.md) and [Google Model Explorer export](../reference/model_explorer.md): open the captured graph in a viewer students may already know.

## Inspecting every operation

- [Extracting source code with TorchLens](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/source_code_extraction.ipynb): the call stack and the actual source line of every operation, so a drawn node points back at the line of PyTorch that produced it.
- [Conditional branch attribution](../conditional_attribution.md): which Python `if` branch each operation ran under.
- [Containers](../containers.md): how tuples, dicts and lists flow between operations.
- [Facets](../facets.md) and the [facets tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/facets_tutorial.ipynb): one activation split into its parts, such as an LSTM's output and hidden states or an attention layer's individual heads.
- [Backward capture](../backward.md) and the [backward tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/backward_tutorial.ipynb): draw the autograd graph and connect each backward node to its forward layer, for the backpropagation lecture.
- [Gradients tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/06_gradients.ipynb): what drives one unit.
- [Receptive and projective fields](../receptive_projective_fields.md): which input positions can affect a given position, and which positions it can affect, computed on the graph that actually ran.

## Hands-on exercises

- [Change and rerun](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/05_change_and_rerun.ipynb): does this layer matter? Zero it and watch the output change.
- [Intervention notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/intervention.ipynb): zero-ablate one layer and replay the downstream computation, in a few lines.
- [Pluck your first activation](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/pluck.ipynb): one activation out of a model in three lines.
- [Compare two runs](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/07_compare_runs.ipynb): the same model on two inputs, diffed operation by operation.
- [Hugging Face tutorial notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/huggingface_tutorial.ipynb): trace a real pretrained transformer and draw it.

## The Model Menagerie

- [The Model Menagerie](https://modelmenagerie.ai): a browsable gallery of over 6,600 neural network architectures, from McCulloch and Pitts's 1943 network of threshold neurons to current models, each with its full computational graph drawn by TorchLens. For a course it is a ready-made source of worked drawings: show the same operation vocabulary across eight decades of designs.
- [The same three lines on any architecture](../gallery/any_architecture.md): the trace, draw and edit calls on six architecture families from the Menagerie, the shortest proof that the tool works on whatever a student brings.
- [Model summaries](../reference/summary.md): a one-call table of every layer with shapes and parameter counts, for the first lecture before the graph.

## Further reading

- [Glossary](../reference/glossary.md): the definition of every public name used above.
- Coming from [torchview](../migration/from_torchview.md), [torchviz and hiddenlayer](../migration/from_torchviz_hiddenlayer.md) or [torchinfo](../migration/from_torchinfo.md): the drawing and summary tools many courses use today.
