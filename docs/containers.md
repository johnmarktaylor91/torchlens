# Containers

TorchLens can record the structure of supported input and output containers (tuples, lists, dicts, namedtuples, dataclasses and Hugging Face `ModelOutput` objects) and expose it through `op.container`. The public names in this guide are stable: `torchlens.types.Container`, `op.container`, the `input_at` and `output_at` selectors, `torchlens.types.register_container`, `Container.reconstruct`, `Trace.reconstruct_output`, `show_containers` (including `"nodes"`) and the `capture_container_structure` option.

The snippets below assume `import torchlens as tl` and a `model` whose forward returns a dict with a `logits` tensor and a legacy `past_key_values` tuple of `(key, value)` pairs per layer.

## Runtime View

Each output leaf `Op` with container metadata exposes `op.container`, a `Container` view built from the recorded structure.

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(capture_container_structure=True))
container = trace.ops[trace.output_layers[0]].container

container.kind
container.root_kind
container.root_id
container.reconstructable
```

Index the view with the same shape as the original output:

```python
logits_op = container["logits"]
key_op = container["past_key_values"][0][0]   # layer 0's key in a legacy (key, value) tuple cache
```

Indexing returns a leaf `Op` or another nested `Container`.

## Reconstruction

`Container.reconstruct()` rebuilds the original Python object from saved leaf values:

```python
output = container.reconstruct()
```

For model returns, use the trace convenience:

```python
output = trace.reconstruct_output()
```

The model output's structure is recorded on every capture, so `trace.reconstruct_output()` and a live `trace.run()` rebuild the tuple, dict, namedtuple, dataclass or `ModelOutput` the model returned, even on a default capture. Per-op container records need the opt-in below (without it the output op records carry `container_spec=None`). A container holding an unsupported tensor-bearing object (for example a Hugging Face `DynamicCache`) is recorded as `opaque`: `reconstruct_output()` raises a typed error instead of returning a partial object, and saving a runnable artifact rejects that output during preflight.

Input-container reconstruction and module-boundary container records need `capture_container_structure=True`, which records nested inputs and per-op container specs (available after capture and after a `.tlspec` load) without enabling intervention replay metadata. `replayable=True` also captures that structure as part of the runnable apparatus.

## Visualization

Container drawing is off by default. `show_containers` takes one of four settings; each one adds to the one before it.

| Setting | What is drawn |
| --- | --- |
| `False` (default) | The plain graph. Container paths already sit on the node labels (`@output.hidden.1`, `@input.inputs.mask`). |
| `"labels"` | The key or index of each container leaf as a small label on the edge into that leaf (`logits`, `0`, `1`). |
| `"collapsed"` | `"labels"`, plus: in the unrolled view, a container with more than `container_max_inline` leaves (default 12) of identical shape folds into one dashed summary box (`tuple x16`). |
| `"nodes"` | `"collapsed"`, plus one dashed record node per container, grey source and sink edges carrying the keys, dashed member-of ties for mid-graph containers, and a dotted box around an op-returned tuple whose leaves share one module. |

Any string setting forces the Graphviz `dot` engine. `container_max_inline` matters only under `"collapsed"` and `"nodes"`.

```python
trace.draw(show_containers="nodes")
```

`show_containers="nodes"` renders one collapsed container node labeled with the container type and role, such as `dict[3] (model input)` or `CausalLMOutputWithPast (model output)`. Model-input containers fan out from that node to their input leaves; model-output leaves fan into a sink node. Field, key, or index labels appear on the relevant edges.

Mid-graph containers keep the key label on the real producer-to-consumer edge and get a light dashed member-of tie to the container node; tensor dataflow is never routed through the container node.

A container that is the model's input or output is drawn as that boundary container even when a submodule built it: its node carries the model-input or model-output role and its leaves connect only through the keyed source or sink edges. When the homogeneous-container collapse already summarizes every leaf of a container, the summary node is the container node (for example `tuple x16 (model output)`) and no second node is drawn.

Container membership is per pass. A rolled node that aggregates several passes of a recurrent layer omits container decoration (typically only the final pass feeds the output container); unrolled nodes keep their per-pass labels.

## Custom Containers

Register your own containers with `tl.types.register_container(type, flatten, unflatten)`:

```python
class PairBox:
    def __init__(self, left, right):
        self.left, self.right = left, right


tl.types.register_container(
    PairBox,
    lambda value: ([value.left, value.right], None),
    lambda aux, children: PairBox(children[0], children[1]),
)
```

`flatten` returns `(children, aux_data)`. `children` must be a list or tuple of values TorchLens can traverse. `unflatten` receives the saved `aux_data` and reconstructed child values.

Built-in handling already covers tuples, lists, dicts, namedtuples, dataclasses, and HuggingFace-style `ModelOutput` objects.

## Backend Capability

Backends declare `input_container_structure` and `output_container_structure` capabilities:

- `torch`: `full_spec` for input and output
- `jax`: `paths_only` for input and output builtin pytrees; custom pytrees are not reconstructable
- `tinygrad`: `paths_only` for input and output
- `mlx`: `none` for input and output
- `paddle`: `paths_only` for input and output
- `tf`: `paths_only` for input and output

If an op has only path metadata and no full `container_spec`, `op.container` may return a path-only view with `reconstructable=False`. Backends with the role-appropriate capability set to `none` return no container view. TorchLens never promotes path-only metadata into a false reconstructable view.

Unknown output objects do not crash capture. TorchLens degrades to the existing fallback traversal and marks the structure as non-reconstructable.
