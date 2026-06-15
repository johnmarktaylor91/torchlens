# Semantic I/O Legibility

TorchLens can make model boundaries readable without changing the captured
graph. The current review-day surface covers decoded classifier outputs,
output provenance, original input display, and preprocessing provenance.

These names are **provisional until review-day signoff**. Do not add them to
the vault glossary yet.

## Output Decode

Classifiers with verified label metadata, such as a model exposing
`model.config.id2label` and `model.config.num_labels`, are decoded at capture
time:

```python
trace = tl.trace(model, x)
trace.output_table(top_n=5)
trace.summary(level="output")
trace.decode_output(top_n=3)
```

Use `output_style=` and `output_head=` when auto-detection should be
overridden or when a multi-output model needs an explicit logits head:

```python
trace = tl.trace(
    model,
    x,
    output_style="classification",
    output_head="logits",
)
```

`trace.output_postprocessor` records the resolver source, selected style,
selected head, label source, confidence, ambiguity flag, and captured top-N
bound. `trace.output_table(top_n, batch_items)` returns a dataframe with
`batch_item`, `rank`, `label`, and `prob`. `batch_items=None` returns all
captured items, an integer returns the first `N` batch items, and a sequence
selects explicit batch indices.

`trace.decode_output(top_n)` returns the captured JSON-primitive decoded
representation. It can recompute from retained logits when available; otherwise
requests above the capture-time bound raise a clear "logits not retained"
error.

For tabular graph export, decoded text is opt-in:

```python
df = trace.to_pandas(include_decoded_output_summary=True)
```

The extra `decoded_output_summary` column is populated only on output-node rows.
The default `trace.to_pandas()` schema is unchanged.

## Custom Output Detectors

Register custom output detectors through `tl.autoroute.output.register(...)`.
The detector receives the live model output object plus TorchLens metadata and
returns a `ResolvedPostprocessing` record or `None`.

```python
from typing import Any

from torchlens.data_classes.trace import ResolvedPostprocessing


@tl.autoroute.output.register(name="my_labels", priority=5)
def my_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Resolve a local label bank for output_style='my_labels'."""

    if meta.get("output_style") != "my_labels":
        return None
    return ResolvedPostprocessing(
        source="project",
        identifier="my_labels",
        verified=True,
        config={"id2label": {0: "cat", 1: "dog"}},
        description="project label bank",
        style="classification",
        selected_output_head=meta.get("output_head"),
        label_source="project-local",
        label_source_version="v1",
        confidence=1.0,
    )


trace = tl.trace(model, x, output_style="my_labels")
```

The registration API is intentionally fail-closed: return `None` for non-matches,
and unregister a temporary detector with
`tl.autoroute.output.unregister("my_labels")`.

## Original Input Display

Pass raw user input through `transform=` when the model expects tensors but the
human-readable input is text, images, or another object. TorchLens stores the
original value in `trace.raw_input` and can show it on the input node. The
portable save policy is controlled by `save_raw_input`.

```python
trace = tl.trace(
    model,
    "0.1,0.2,2.0,1.0",
    transform=text_to_tensor,
    save_raw_input="small",
)
trace.draw(show_input_transform_summary=True)
```

Auto-route bridges populate `trace.input_preprocessor` when they resolve a
known preprocessing pipeline. For local transforms, attach the same provenance
record shape when documenting the transform in a demo or integration:

```python
from torchlens.data_classes.trace import ResolvedPreprocessing

trace.input_preprocessor = ResolvedPreprocessing(
    source="project.transform",
    identifier="comma-vector-v1",
    verified=True,
    config={"format": "comma-separated floats"},
    description="comma-separated text -> rank-2 float tensor",
)
```

`draw(show_input_transform_summary=True)` adds the preprocessing summary next
to the raw-input node. The default render remains unchanged.

## Node Annotations

Use `trace.annotate(...)` to attach small JSON data, tensor blobs, or an image
path to selector-matched nodes. Annotation data is user-owned, survives
`rerun()`/fork/save/load where the payload policy supports it, and does not
change the default graph when no annotation-aware render hook is supplied.

```python
trace.annotate(tl.func("relu"), data={"note": "late block"}, max_fanout=4)
trace.annotate(tl.label("conv2d_1_1"), data=torch.zeros(8, 2))
annotated = trace.with_annotations(tl.func("linear"), image="thumbnail.png")
```

JSON-serializable `data=` is stored under the user annotation namespace.
Torch tensor payloads are stored in `trace._annotation_blobs` and are currently
torch-backend only. NumPy arrays should be converted explicitly to tensors or
JSON lists before annotation so they do not lose numeric structure.

## Model Profile

`trace.model_profile` is a computed, non-persisted descriptor for semantic I/O
readiness. It summarizes input modality, preprocessing source, output label
source/count, raw-stimulus count, whether raw images are available, and whether
the trace has enough semantic metadata for the image-classifier keystone flow.

```python
trace = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    output_style="classification",
    save_raw_input=True,
)
trace.model_profile
```

For local transforms, set `trace.input_preprocessor` to a
`ResolvedPreprocessing` record before presenting the profile. This keeps demos
and integrations honest about where preprocessing came from without adding a
new persisted field.

## Representation Geometry Thumbnails

`tl.repgeom` is the provisional representation-geometry surface. It has no
optional plotting dependency. `tl.repgeom.mds_evolution(...)` computes
classical two-dimensional MDS over the saved batch activation for selected
single-pass layers, Procrustes-aligns each layer to the previous selected
layer by default, and annotates the trace with coordinate tensors keyed by
`layer:<layer_label>`.

```python
mds_layers = tl.in_module("block1") | tl.in_module("block2")
trace = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)
coords_by_layer = tl.repgeom.mds_evolution(trace, save=mds_layers, min_n=8)
```

The selected activations must be saved by the original capture; this is why
the demo uses a curated `save=` subset instead of `save="all"`. `min_n`
defaults to 8 because the scatter is a visual summary over a stimulus batch,
not a statistical inference routine. Recurrent aggregate layers are rejected;
select a pass-qualified op when a reused module should be visualized.

Render MDS thumbnails with the draw-time node hook:

```python
trace.draw(
    node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8),
)
```

The hook reads coordinate tensors from `_annotation_blobs` and raw PIL stimuli
from `trace.raw_input` at draw time, then embeds a fresh PNG in the rendered
node. If raw images are missing or the batch size no longer matches the
coordinates, it renders an explicit point-cloud fallback.

## Keystone Image-Classifier Flow

The copy-paste flow for an image classifier is:

```python
mds_layers = tl.in_module("block1") | tl.in_module("block2")
trace = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)

trace.model_profile
trace.output_table(top_n=5)
trace.summary(level="output")
tl.repgeom.mds_evolution(trace, save=mds_layers, min_n=8)
trace.draw(node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8))
trace.draw(show_input_transform_summary=True)
```

Use `save_raw_input=True` for full-resolution original input display. The
default `save_raw_input="small"` keeps portable bundles bounded by downsampling
images before save, which is useful for storage but not ideal for this visual
demo.

## Runnable Demo

See `examples/semantic_io_legibility_demo.py` for a deterministic copy-paste
template. It demonstrates:

- auto-detected label tables from `config.id2label`;
- `summary(level="output")`, `decode_output()`, and gated pandas summaries;
- `output_style=` override and `tl.autoroute.output.register(...)`;
- original text input display with `show_input_transform_summary=True`;
- `trace.raw_input` and `trace.input_preprocessor` provenance.

See `notebooks/semantic_io_keystone_demo.ipynb` for the Sprint B keystone demo.
It uses a tiny deterministic image classifier and synthetic PIL stimuli so the
core flow runs without downloading external model weights.

## Provisional Public Names

- `tl.trace(..., output_style=..., output_head=...)`
- `Trace.output_table(top_n=5, batch_items=None)`
- `Trace.decode_output(top_n=None)`
- `Trace.output_postprocessor`
- `Trace.summary(level="output")`
- `tl.autoroute.output.register(...)`
- `tl.autoroute.output.unregister(...)`
- `tl.autoroute.output.list()`
- `tl.autoroute.output.info(...)`
- `tl.trace(..., transform=..., save_raw_input=..., batch_render=...)`
- `Trace.raw_input`
- `Trace.input_preprocessor`
- `Trace.draw(show_input_transform_summary=True)`
- `Trace.to_pandas(include_decoded_output_summary=True)`
- `Trace.annotate(selector, data=..., image=..., max_fanout=..., copy=...)`
- `Trace.with_annotations(selector, data=..., image=..., max_fanout=...)`
- `Trace.model_profile`
- `tl.repgeom.classical_mds(...)`
- `tl.repgeom.procrustes_align(...)`
- `tl.repgeom.activation_distance_matrix(...)`
- `tl.repgeom.mds_evolution(...)`
- `tl.repgeom.mds_scatter_node_spec(...)`
- `Trace.draw(node_spec_fn=...)`
- `Trace._annotation_blobs`
