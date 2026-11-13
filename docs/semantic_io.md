# Semantic I/O Legibility

TorchLens can make model boundaries readable without changing the captured
graph. The surface covers decoded classifier outputs,
output provenance, original input display, and preprocessing provenance.

These names are provisional and may change in any release.

## Output Decode

Classifiers with verified label metadata, such as a model exposing
`model.config.id2label` and `model.config.num_labels`, are decoded at capture
time:

```python
trace = tl.trace(model, x)
trace.output_table(k=5)
trace.decode_output(k=3)
```

Decoding is not a capture option. When auto-detection should be overridden,
or a multi-output model needs an explicit logits head, decode again at read
time from the retained output tensors:

```python
trace = tl.trace(model, x)
trace.decode_output(style="imagenet")  # force a style
trace.decode_output(head="logits")  # pick the head of a multi-output model
trace.output_table(k=5, style="classification", head="0")
trace.decode_output(style="hf_text", tokenizer=tok)
```

The read-time decode returns rows in the same shape as `trace.decode_output()`
and never rewrites the captured record. It needs the output activations
retained (the default save policy); a structure-only or output-excluding
capture refuses with `decoded_output_unavailable`. A `style` that no
detector resolves refuses typed (`decode_style_unknown`) with a did-you-mean
over the built-in styles (`classification`, `imagenet`, `imagenet1k`,
`hf_text`); a `head` that is not an output head refuses `decode_head_unknown`
with the available heads, and several qualifying heads with none named
refuse `decode_head_ambiguous`. A trace captured without a tokenizer (any
capture outside the bridge) still decodes with `style="hf_text"`: the rows carry
each next token's `token_id` and `text` is `None` until a `tokenizer=` is passed.
Passing the removed `output_style=` or `output_head=` to `CaptureOptions` or
`tl.trace` refuses naming `trace.decode_output(...)`.

Text models traced from a raw string (`tl.trace(model, "text")` routes a
Hugging Face language model through `tl.bridge.transformers.trace_text`)
decode as text automatically: the bridge attaches the tokenizer for the span
of the capture and the `hf_text` detector fires on it, decoding only a head
named `logits` (or one bare output tensor). A text classifier with
`config.id2label` gets its labels instead; the placeholder `id2label` every
Hugging Face config carries (`LABEL_0`, `LABEL_1`) never claims
vocabulary-width or per-token logits, so language models fall through to the
text decode.

`trace.provenance.output_postprocessor` records the resolver source, selected style,
selected head, label source, confidence, ambiguity flag, and captured top-N
bound. `trace.output_table(k, batch_items)` returns a dataframe with
`batch_item`, `rank`, `label`, and `prob` for a classification decode: the top
`k` classes per batch item. For a text decode it returns the top `k` next
tokens at the last position per batch item, with `batch_item`, `rank`,
`token_id`, `text`, and `prob`, where `prob` is the token's probability under
the softmax over the vocabulary at that position:

```python
trace = tl.trace(gpt2, "The capital of France is")
trace.output_table(k=3)  # three rows: rank 1-3, token_id, text, prob
```
`batch_items=None` returns all captured items, an integer returns the first
`N` batch items, and a sequence selects explicit batch indices.

`trace.decode_output(k)` returns the captured JSON-primitive decoded
representation. It can recompute from retained logits when available; otherwise
requests above the capture-time bound raise a clear "logits not retained"
error. When nothing was decoded at capture, it runs the output detectors over
the retained output tensors on demand and refuses typed when none is retained
(`decoded_output_unavailable`) or no detector resolves a style
(`decode_style_unknown`).

For tabular graph export, decoded text is opt-in:

```python
df = trace.to_pandas(include_decoded_output_summary=True)
```

The extra `decoded_output_summary` column is populated only on output-node rows.
The default `trace.to_pandas()` schema is unchanged.

## Custom Output Detectors

Register custom output detectors through `tl.postprocessing.register(...)`
(the `torchlens.postprocessing` namespace also exports `unregister`, `list`,
`info`, `snapshot` and the `ResolvedPostprocessing` record).
A detector runs twice: automatically at the end of every capture, with the
live model output object and `meta["style"] is None`, and on demand from
`trace.decode_output(style="my_labels")`, with the retained output tensors
keyed by head (`{"logits": tensor}`, or the one bare tensor) and
`meta["style"]` set to the requested style. It returns a
`ResolvedPostprocessing` record or `None`.

```python
from typing import Any

from torchlens.postprocessing import ResolvedPostprocessing


@tl.postprocessing.register(name="my_labels", priority=5)
def my_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
    """Resolve a local label bank for decode_output(style='my_labels')."""

    if meta.get("style") != "my_labels":
        return None
    return ResolvedPostprocessing(
        source="project",
        identifier="my_labels",
        verified=True,
        config={"id2label": {0: "cat", 1: "dog"}},
        description="project label bank",
        style="classification",
        selected_output_head=meta.get("head"),
        label_source="project-local",
        label_source_version="v1",
        confidence=1.0,
    )


trace = tl.trace(model, x)
rows = trace.decode_output(style="my_labels")
```

The registration API is intentionally fail-closed: return `None` for non-matches,
and unregister a temporary detector with
`tl.postprocessing.unregister("my_labels")`. Raw input has the matching door,
`tl.preprocessing.register(...)` and `tl.preprocessing.unregister(...)`: an input
detector receives `(model, payload, **kwargs)` before routing and claims the
call by re-entering `tl.trace` with what `forward` takes. The guide
[Adding conveniences for your model](guides/model_conveniences.md) walks
through an input detector, an output decoder and a custom
[facet](facets.md) on one model.

## Image Inputs

A PIL image, or a path to an image file, is a complete `tl.trace` input for a
vision model:

```python
trace = tl.trace(resnet, "dog.jpg")
trace = tl.trace(resnet, Image.open("dog.jpg"))
```

A `str` or `os.PathLike` naming an existing file with an image suffix (`.jpg`,
`.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tif`, `.tiff`), or a list of them,
is opened as an RGB PIL image before routing; a string with whitespace is text.
A missing image path refuses `image_path_missing` (unless the model has a text
tokenizer, in which case the string is text) and an undecodable file refuses
`image_path_unreadable`. The image is then preprocessed by the authority
`torchlens.preprocessing.resolve(model=)` finds: a torchvision model is
recognised from its loaded state when exactly one preset checkpoint already in
the torch hub cache equals it (`resolution_method="state_match"`), a timm
model from its data config, a Hugging Face model from its image processor.
When nothing resolves the capture refuses `preprocessing_unresolved`; record
the authority with `torchlens.preprocessing.attach_authority(model, weights)`
or pass your own transform through `CaptureOptions(input_transform=)`.

## Original Input Display

Pass raw user input through `CaptureOptions(input_transform=)` when the model expects tensors but the
human-readable input is text, images, or another object. TorchLens stores the
original value in `trace.input` and can show it on the input node. The
portable save policy is controlled by `save_raw_input`.

```python
trace = tl.trace(
    model,
    "0.1,0.2,2.0,1.0",
    capture=tl.options.CaptureOptions(
        input_transform=text_to_tensor,
        save_raw_input="small",
    ),
)
trace.draw(show_input_transform_summary=True)
```

Auto-route bridges populate `trace.provenance.input_preprocessor` when they resolve a
known preprocessing pipeline. For local transforms, attach the same provenance
record shape when documenting the transform in a demo or integration:

```python
from torchlens._data_classes.trace import ResolvedPreprocessing

trace._provenance_input_preprocessor = ResolvedPreprocessing(
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
`run()`/fork/save/load where the payload policy supports it, and does not
change the default graph when no annotation-aware render hook is supplied.

```python
trace.annotate(tl.func("relu"), data={"note": "late block"}, max_fanout=4)
trace.annotate(tl.label("conv2d_1_1"), data=torch.zeros(8, 2))
annotated = trace.annotate(tl.func("linear"), image="thumbnail.png", copy=True)
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
    capture=tl.options.CaptureOptions(
        input_transform=image_batch_to_tensor,
        save_raw_input=True,
    ),
)
trace.model_profile
```

For local transforms, set `trace.provenance.input_preprocessor` to a
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
    save=mds_layers,
    capture=tl.options.CaptureOptions(
        input_transform=image_batch_to_tensor,
        save_raw_input=True,
    ),
)
coords_by_layer = tl.repgeom.mds_evolution(trace, save=mds_layers, min_stimuli=8)
```

The selected activations must be saved by the original capture; this is why
the demo uses a curated `save=` subset instead of exhaustive saving. `min_stimuli`
defaults to 8 because the scatter is a visual summary over a stimulus batch,
not a statistical inference routine. Recurrent aggregate layers are rejected;
select a pass-qualified op when a reused module should be visualized.

Render MDS thumbnails with the draw-time node hook:

```python
trace.draw(
    node_spec_fn=tl.viz.nodes.mds_scatter(max_thumbnails=8),
)
```

The hook reads coordinate tensors from `_annotation_blobs` and raw PIL stimuli
from `trace.input` at draw time, then embeds a fresh PNG in the rendered
node. If raw images are missing or the batch size no longer matches the
coordinates, it renders an explicit point-cloud fallback.

Module outputs work the same way. Selecting `tl.module("layer1")` saves the
op that ends the stage, which for a reused module (ResNet's shared ReLU) is a
later pass; in an unrolled drawing that pass's node carries the scatter. When
the stages are drawn as collapsed boxes, pass the same hook as
`collapsed_node_spec_fn`, and each box shows the scatter of its call's output:

```python
stages = tl.module("layer1") | tl.module("layer2")
trace = tl.trace(model, image_list, save=stages)
tl.repgeom.mds_evolution(trace, save=stages)
trace.draw(depth=1, collapsed_node_spec_fn=tl.viz.nodes.mds_scatter())
```

A rolled box that stands for several calls of one module has no single output
and is left undecorated, as is a rolled multi-pass layer.

## Node-Visual Render Toolkit (provisional)

Sprint C extends the provisional node-visual surface with PIL-only render
primitives and three additional templates. These names are provisional pending
review-day signoff.

The low-level primitives live under `tl.viz`:

```python
heatmap_image = tl.viz.heatmap(array)
lineplot_image = tl.viz.lineplot(series)
scatter_image = tl.viz.image_scatter(coords, images=image_list)
```

They return bounded `PIL.Image.Image` objects directly. Matplotlib is
intentionally not used, so graph drawing can compose compact deterministic
images without adding a plotting dependency. These are separate from the older
`tl.viz.nodes.heatmap(max_size=...)` visualizer factory, which returns a callable for
tensor payload display.

RDM annotations use the representation-geometry helpers:

```python
rdm_matrix = tl.repgeom.rdm(activations)
rdms_by_layer = tl.repgeom.rdm_evolution(trace, save=mds_layers)
trace.draw(node_spec_fn=tl.viz.nodes.rdm(max_stimuli=8))
```

`rdm_evolution(...)` stores an `N x N` representational dissimilarity matrix
for each selected saved activation site. `rdm_node_spec(...)` renders the
stored matrix as a heatmap, optionally with symmetric raw-input thumbnails on
both axes when the captured `trace.input` still matches the stimulus batch.

Feature and activation map annotations live under `tl.viz` because they are
activation displays, not representation-geometry numerics:

```python
feature_maps = tl.viz.feature_maps(trace, save=mds_layers)
trace.draw(node_spec_fn=tl.viz.nodes.feature_map())
```

`feature_maps(...)` requires saved spatial activations with shape
`[N, C, H, W]`; capture the convolutional layers you want to inspect before
calling it. Non-spatial sites are skipped, and the call raises only when no
selected site has spatial activations. By default, `feature_map_node_spec()`
renders a channel-aggregate heatmap overlay on the matching stimulus thumbnail.
Use `channels=None` for the aggregate view, explicit channel indices for
specific channels, or `channels="top"` to select the strongest channels per
stimulus.

Scree annotations expose the layerwise eigenspectrum:

```python
eigenvalues = tl.repgeom.scree(activations)
dim_info = tl.repgeom.effective_dimensionality(activations)
scree_by_layer = tl.repgeom.scree_evolution(trace, save=mds_layers)
trace.draw(node_spec_fn=tl.viz.nodes.scree())
```

`effective_dimensionality(...)` returns variance-explained and dimensionality
summary fields, including participation ratio and the number of components
needed for a variance threshold. `scree_node_spec(...)` renders the stored
eigenvalues as a variance-explained line plot with a compact dimensionality
callout.

All four templates render at draw time through `Trace.draw(node_spec_fn=...)`.
The evolution helpers store tensor payloads in `Trace._annotation_blobs`, so
the visuals survive `.tlspec` save/load with the annotation payloads. No new
`Trace` field was added for Sprint C.

## Keystone Image-Classifier Flow

The copy-paste flow for an image classifier is:

```python
mds_layers = tl.in_module("block1") | tl.in_module("block2")
trace = tl.trace(
    model,
    image_list,
    save=mds_layers,
    capture=tl.options.CaptureOptions(
        input_transform=image_batch_to_tensor,
        save_raw_input=True,
    ),
)

trace.model_profile
trace.output_table(k=5)
tl.repgeom.mds_evolution(trace, save=mds_layers, min_stimuli=8)
trace.draw(node_spec_fn=tl.viz.nodes.mds_scatter(max_thumbnails=8))
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
- `output_table()`, `decode_output()`, and gated pandas summaries;
- the read-time `decode_output(style=, head=)` override and `tl.postprocessing.register(...)`;
- original text input display with `show_input_transform_summary=True`;
- `trace.input` and `trace.provenance.input_preprocessor` provenance.

See `notebooks/semantic_io_keystone_demo.ipynb` for the image-classifier keystone demo.
It uses a tiny deterministic image classifier and synthetic PIL stimuli so the
core flow runs without downloading external model weights.

## Provisional Public Names

- `Trace.output_table(k=5, batch_items=None, *, style=None, head=None)`
- `Trace.decode_output(k=None, *, style=None, head=None, tokenizer=None)`
- `Trace.provenance.output_postprocessor`
- `tl.postprocessing.register(...)`
- `tl.postprocessing.unregister(...)`
- `tl.postprocessing.list()`
- `tl.postprocessing.info(...)`
- `tl.postprocessing.snapshot()`
- `torchlens.postprocessing.ResolvedPostprocessing`
- `tl.preprocessing.register(...)`
- `tl.preprocessing.unregister(...)`
- `tl.trace(..., capture=CaptureOptions(input_transform=..., save_raw_input=..., batch_render=...))`
- `Trace.input`
- `Trace.provenance.input_preprocessor`
- `Trace.draw(show_input_transform_summary=True)`
- `Trace.to_pandas(include_decoded_output_summary=True)`
- `Trace.annotate(selector, data=..., image=..., max_fanout=..., copy=...)`
- `Trace.model_profile`
- `tl.repgeom.classical_mds(...)`
- `tl.repgeom.procrustes_align(...)`
- `tl.repgeom.activation_distance_matrix(...)`
- `tl.repgeom.mds_evolution(...)`
- `tl.viz.nodes.mds_scatter(...)`
- `Trace.draw(node_spec_fn=...)`
- `Trace._annotation_blobs`
