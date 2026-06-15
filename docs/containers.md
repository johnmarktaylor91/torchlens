# Containers

TorchLens records input and output container structure for supported Python containers and exposes it as a lightweight runtime view. The public names in this guide are provisional for review: `Container`, `input_at`, `output_at`, `register_container`, and `capture_container_structure`.

## Runtime View

Each output leaf `Op` with container metadata exposes `op.container`. The value is a computed `Container` view backed by the captured `ContainerSpec` plus sibling leaf ops; it is not stored as another trace record.

```python
trace = tl.trace(model, x, capture_container_structure=True)
container = trace.ops[trace.output_layers[0]].container

container.kind
container.root_kind
container.root_id
container.reconstructable
```

Index the view with the same shape as the original output:

```python
logits_op = container["logits"]
key_op = container["past_key_values"][0][1]
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

Final model-output and input-container reconstruction require container structure to be captured. Use `capture_container_structure=True` when you need nested inputs or the top-level return object reconstructed after capture or after `.tlspec` load without enabling intervention replay metadata. `intervention_ready=True` also captures final-output structure as part of its broader replay-template metadata. `capture_output_structure=True` remains as a deprecated alias for existing callers.

## Nested Output Selection

`tl.output_at(path)` selects an output leaf by nested path:

```python
site = trace.resolve_sites(tl.output_at(("past_key_values", 0, 1))).first()
```

The selector matches TorchLens typed paths for dict keys, HuggingFace `ModelOutput` keys, namedtuple/dataclass fields, and tuple/list indices.

## Visualization

Container visualization is opt-in so default graph output remains unchanged.

```python
trace.draw(show_containers="nodes")
```

`show_containers="nodes"` renders one collapsed container node labeled with the container type and role, such as `dict[3] (model input)` or `CausalLMOutputWithPast (model output)`. Model-input containers fan out from that node to their input leaves; model-output leaves fan into a sink node. Field, key, or index labels appear on the relevant edges.

For mid-graph containers, TorchLens does not route tensor dataflow through the container node. The key label stays on the real producer-to-consumer edge, and a light dashed arrowless member-of tie associates producer ops with the collapsed container node. Single-owner module clusters and large homogeneous-container collapse still apply as visual enhancers.

Existing modes are unchanged: `False`, `"labels"`, `"cluster"`, `"collapsed"`, and `"auto"` keep their prior rendering behavior.

## Custom Containers

Register proprietary containers with `tl.register_container(type, flatten, unflatten)`.

```python
tl.register_container(
    PairBox,
    lambda value: ([value.left, value.right], None),
    lambda aux, children: PairBox(children[0], children[1]),
)
```

`flatten` returns `(children, aux_data)`. `children` must be a list or tuple of values TorchLens can traverse. `unflatten` receives the saved `aux_data` and reconstructed child values.

Built-in handling already covers tuples, lists, dicts, namedtuples, dataclasses, and HuggingFace-style `ModelOutput` objects.

## Capability Degradation

## Backend Capability

Backends declare honest `input_container_structure` and `output_container_structure` capabilities:

- `torch`: `full_spec` for input and output
- `jax`: `paths_only` for input and output builtin pytrees; custom pytrees are not reconstructable
- `tinygrad`: `paths_only` for input and output
- `mlx`: `none` for input and output

If an op has only path metadata and no full `container_spec`, `op.container` may return a path-only view with `reconstructable=False`. Backends with the role-appropriate capability set to `none` return no container view. TorchLens never promotes path-only metadata into a false reconstructable view.

Unknown output objects do not crash capture. TorchLens degrades to the existing fallback traversal and marks the structure as non-reconstructable.
