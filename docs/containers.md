# Containers

TorchLens records output-container structure for supported Python containers and exposes it as a lightweight runtime view. The public names in this guide are provisional for review: `Container`, `output_at`, and `register_container`.

## Runtime View

Each output leaf `Op` with container metadata exposes `op.container`. The value is a computed `Container` view backed by the captured `ContainerSpec` plus sibling leaf ops; it is not stored as another trace record.

```python
trace = tl.trace(model, x, intervention_ready=True)
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

Final model-output reconstruction requires output structure to be captured. Use `intervention_ready=True` for traces where you need the top-level return object reconstructed after capture or after `.tlspec` load.

## Nested Output Selection

`tl.output_at(path)` selects an output leaf by nested path:

```python
site = trace.resolve_sites(tl.output_at(("past_key_values", 0, 1))).first()
```

The selector matches TorchLens typed paths for dict keys, HuggingFace `ModelOutput` keys, namedtuple/dataclass fields, and tuple/list indices.

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

If an op has only path metadata and no full `container_spec`, `op.container` may return a path-only view with `reconstructable=False`. Plain non-container ops return `None`.

Unknown output objects do not crash capture. TorchLens degrades to the existing fallback traversal and marks the structure as non-reconstructable.
