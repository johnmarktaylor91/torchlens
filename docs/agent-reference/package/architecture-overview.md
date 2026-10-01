## Architecture Overview

```
import torchlens
  |- exposes 116 top-level public names in __all__
  |- eagerly imports ONLY the light spine: options, errors/_state, ir.*,
  |  captured_run, observers, quantities, _deprecations, _errors, _io,
  |  _literals, _save_budget, utils,
  |  and visualization (hard-pinned by tests/test_import_hygiene.py
  |  _EAGER_TORCHLENS_MODULES). capture, intervention, fastlog, autoroute,
  |  bridge, compat, export, report, stats, validation, and viz-rendering
  |  internals are ALL lazy (_LAZY_ATTRS) — do not add eager imports
  |
trace(model, input, save=..., intervene=..., lookback=..., storage=...)
  |- backends/registry.py      - resolve torch / MLX / JAX / tinygrad / Paddle / TensorFlow backend
  |- backends/torch/model_prep.py - ensure torch is wrapped, prepare modules/buffers/params
  |- capture/trace.py          - run forward pass with active logging
  |- backends/torch/ops.py     - build raw torch op records during wrapper calls
  |- postprocess/              - current 26-step graph cleanup/finalization pipeline
                                 (contract keys 0..20 + 5 fractional inserts)
  +- returns Trace

tl.record(model, input, save=...)
  |- uses the same wrapper hot path
  |- stores predicate-selected RecordContext/ActivationRecord values
  +- returns Recording; Recording.to_trace() materializes full structure
```

Selective `layers_to_save` uses a predicate-backed single pass when early labels are
sufficient and falls back to the two-pass strategy for final-numbering selectors:
negative indexes, integer ordinals, indexed label strings (`relu_1_2`, `relu_1` — any
`_<digit>` component; orphan removal renumbers ordinals AND type indexes after capture),
output labels, identity labels, and gradient selection (integer `save_grads` ordinals
included — deferred grad hooks install post-postprocess from the reference escrow, never
from raw-index prediction). A mixed selection with a negative tail disables the escrow
eviction window so early final-numbering components keep their payloads. String
selectors keep the legacy substring contract. Unqualified recurrent
labels save all passes; pass-qualified labels such as `"attn:2"` save one 1-based pass.
Prefer `save=tl.func(...)`, `save=tl.in_module(...)`, and composed predicates for new
single-pass selective capture. The old `keep_op=`/`keep_module=` `record()` alias
kwargs are removed; `save=` is the only predicate spelling and `default_module=`
gates module-boundary event recording (uniformly — ALL module enter/exit events;
predicate-gated module-event selection has no public spelling).

Common unified capture examples:

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
paddle_trace = tl.trace(paddle_model, paddle_x, backend="paddle")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("attn"),
    intervene=tl.when(tl.func("attn"), tl.scale(0.5)),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
trace.draw(show_containers="nodes")
```

Provisional semantic I/O surface (review-day names):

```python
log = tl.trace(
    model, x, capture=tl.options.CaptureOptions(output_style="classification", output_head="logits")
)
log.output_table(top_n=5)
log.summary(level="output")
log.to_pandas(include_decoded_output_summary=True)

input_log = tl.trace(
    model,
    raw_text,
    capture=tl.options.CaptureOptions(transform=text_to_tensor, save_raw_input="small"),
)
input_log.draw(show_input_transform_summary=True)

mds_layers = tl.in_module("block1") | tl.in_module("block2")
image_log = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)
image_log.model_profile
image_log.output_table(top_n=5)
tl.repgeom.mds_evolution(image_log, save=mds_layers, min_n=8)
tl.repgeom.rdm_evolution(image_log, save=mds_layers)
tl.viz.feature_map_evolution(image_log, save=mds_layers)
tl.repgeom.scree_evolution(image_log, save=mds_layers)
image_log.draw(node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8))
image_log.draw(node_spec_fn=tl.repgeom.rdm_node_spec(max_stimuli=8))
image_log.draw(node_spec_fn=tl.viz.feature_map_node_spec())
image_log.draw(node_spec_fn=tl.repgeom.scree_node_spec())
```

Sprint B annotation/MDS names are provisional until review-day signoff. `Trace.model_profile`
is computed, not persisted. `tl.repgeom.mds_evolution(...)` requires the target batch
activations to have been saved at capture time; use a curated `save=` subset, not exhaustive
`layers_to_save="all"`,
for image batches. `Trace._annotation_blobs` is public-provisional only for render-time
annotation payloads and compatibility review.
Sprint C RDM, feature-map, and scree node visuals are PIL-only render-time images composed
from `tl.viz.render_*` primitives and are provisional until review-day signoff.

`backward_ready=True` is the public opt-in for losses built from saved outs. It keeps
floating tensors graph-connected, preserves user `requires_grad`, and rejects incompatible
detaching or disk-only out storage.
`inference_only=True` is the opt-in no-grad capture path for forward-only analysis; it is mutually
exclusive with backward-related capture because it discards the autograd graph.
