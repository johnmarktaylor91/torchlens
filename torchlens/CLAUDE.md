# torchlens/ - Core Package

## What This Is
TorchLens extracts outs and metadata from backend-resolved captures. PyTorch eager capture is the
stable default; MLX, JAX, tinygrad, Paddle, and TensorFlow are technical-preview backends. `import torchlens`
exposes the public API and compatibility shims, but torch wrapping is lazy: the first torch capture
prepares the model and calls `wrap_torch()` from `decoration/`.

## Architecture Overview

```
import torchlens
  |- exposes 89 top-level public names in __all__
  |- imports submodule namespaces: fastlog, bridge, compat, export, options, report, stats, viz
  |
trace(model, input, save=..., intervene=..., lookback=..., storage=...)
  |- backends/registry.py      - resolve torch / MLX / JAX / tinygrad / Paddle / TensorFlow backend
  |- decoration/model_prep.py  - ensure torch is wrapped, prepare modules/buffers/params
  |- capture/trace.py          - run forward pass with active logging
  |- capture/output_tensors.py - build Op records
  |- postprocess/              - current 20-step graph cleanup/finalization pipeline
  +- returns Trace

tl.record(model, input, save=...)
  |- uses the same wrapper hot path
  |- stores predicate-selected RecordContext/ActivationRecord values
  +- returns Recording; Recording.to_trace() materializes full structure
```

Selective `layers_to_save` uses the two-pass strategy: Pass 1 exhaustive metadata, Pass
2 fast out save. Unqualified recurrent labels save all passes; pass-qualified labels such
as `"attn:2"` save one 1-based pass. Prefer `save=tl.func(...)`, `save=tl.in_module(...)`,
and composed predicates for single-pass selective capture. `keep_op=` and `keep_module=`
remain deprecated `record()` aliases for `save=`.

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
log = tl.trace(model, x, output_style="classification", output_head="logits")
log.output_table(top_n=5)
log.summary(level="output")
log.to_pandas(include_decoded_output_summary=True)

input_log = tl.trace(model, raw_text, transform=text_to_tensor, save_raw_input="small")
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
activations to have been saved at capture time; use a curated `save=` subset, not `save="all"`,
for image batches. `Trace._annotation_blobs` is public-provisional only for render-time
annotation payloads and compatibility review.
Sprint C RDM, feature-map, and scree node visuals are PIL-only render-time images composed
from `tl.viz.render_*` primitives and are provisional until review-day signoff.

`backward_ready=True` is the public opt-in for losses built from saved outs. It keeps
floating tensors graph-connected, preserves user `requires_grad`, and rejects incompatible
detaching or disk-only out storage.
`inference_only=True` is the opt-in no-grad capture path for forward-only analysis; it is mutually
exclusive with backward-related capture because it discards the autograd graph.

## Top-Level Modules

| Path | Purpose |
|------|---------|
| `__init__.py` | Top-level API, 89-name `__all__`, deprecation shims, `peek`/`extract` helpers |
| `_state.py` | Global logging toggle, active log, decoration maps, prepared-model registry; no torchlens imports |
| `_trace_state.py` | Small runtime state enum exposed through `torchlens.io` |
| `_errors.py`, `errors/` | Public and legacy exception classes |
| `_io/`, `io/` | Portable `.tlspec` save/load, manifest, lazy tensor refs, public I/O helpers |
| `options.py` | Capture, save, visualization, replay, intervention, and streaming option groups |
| `observers.py` | `tap()` and `record_span()` observer helpers |
| `report/` | `report.explain(log)` and capture-time scalar logging |
| `stats/` | Streaming stats and `aggregate()` over dataloaders |
| `types.py`, `accessors/` | Moved type/accessor aliases for non-top-level public names |

## Subpackages
- `capture/` - real-time forward and backward operation logging.
- `data_classes/` - `Trace`, `Layer`, `Op`, module/param/buffer/grad logs.
- `decoration/` - lazy torch function wrapping, explicit wrap/unwrap, module prep.
- `fastlog/` - sparse predicate recording with RAM/disk storage and recovery.
- `postprocess/` - graph cleanup, conditionals, loop detection, labeling, finalization.
- `validation/` - forward replay, backward validation, metadata invariants, `.tlspec` schema checks.
- `visualization/` - Graphviz rendering, rank layout, NodeSpec, themes, overlays, bundle diff.
- `intervention/` - selectors, sites, hooks, helpers, Bundle, fork/replay/rerun/save.
- `intervention/_super/` - internal Bundle-level Super* aligned views and accessors.
- `intervention/_topology/` - internal bundle supergraph and topology diff support.
- `bridge/`, `compat/`, `callbacks/` - optional integrations and migration facades.
- `viewer/`, `paper/`, `notebook/`, `llm/`, `neuro/` - appliance package boundaries gated by extras.

## Key Concepts

### Toggle Architecture
- Lazy wrapping: `wrap_torch()` installs wrappers on first capture or explicit call.
- Persistent wrappers: after wrapping, calls only pay a `_state._logging_enabled` check when
  logging is off.
- `active_logging(trace)` enables logging during the forward; `pause_logging()` protects
  internal TorchLens tensor ops from recursive capture.
- `patch_detached_references()` patches `from torch import cos` style references in loaded modules.

### Module Containment
Module containment is captured via a wrap-forward stack helper at
`decoration/_module_stack.py`. Both fastlog and exhaustive modes share the helper. Each
captured op snapshots the stack at op-creation time; downstream postprocess only appends
the canonical module-path suffix to `equivalence_class` for loop detection. This replaces
the older tensor-entry/exit thread-replay system removed in v2.18 (sprint
module-containment-refactor).

### Data Flow
1. Decoration intercepts torch function calls.
2. Barcode nesting detection identifies bottom-level operations.
3. `capture/` builds raw `Op` entries.
4. `postprocess/` removes orphans, marks conditionals, detects loops, labels nodes, builds logs.
5. `Trace` exposes lookup, visualization, validation, save/load, intervention, and summary helpers.

### Portable Artifacts
`tl.save()` and `tl.load()` route through `_io/bundle.py`. Unified `.tlspec` directories have
`manifest.json` plus safetensors blobs; public schema validation lives in `validation/__init__.py`.
Non-torch preview backends use `payload_policy="array_payloads"` when their codecs can materialize
payloads; Paddle bf16 payloads carry logical dtype metadata because NumPy transports them as
`uint16`. TensorFlow preview payloads also use `array_payloads` for dense numeric/bool forward
arrays and preserve `tf.bfloat16` logical dtype metadata.
Intervention specs can be saved at audit, executable-with-callables, or portable levels.

### Appliances
The five appliance subfolders are part of the 2.x package layout. `viewer`, `paper`, and
`llm` are empty stubs with docstring intent. `notebook` and `neuro` currently enforce their
extras by importing required dependencies, but export no public objects yet.
