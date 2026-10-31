# viz/ - Implementation Guide

Exposed as `tl.viz` through the lazy attribute map in `torchlens/__init__.py`
(`"viz": ("torchlens.viz", None)`).

## viz/ vs visualization/ (dual home - factual split, by imports)
- `viz/` owns PIL-based tensor/node visualizers: visualizer factories, node-spec
  render primitives, batch summaries, and the `Layer.show`/`Op.show` display path.
- `visualization/` owns Graphviz graph rendering (rank layout, NodeSpec, themes,
  overlays, bundle diff).
- Cross imports today: `viz/__init__.py` re-exports `bundle_diff` from
  `..visualization.bundle_diff`; `viz/feature_maps.py` imports `NodeSpec` /
  `NodeSpecFn` from `..visualization.node_spec`; and
  `visualization/_render_nodes.py` imports `..viz.batch_summary`
  (`batch_summary.text_table` at its string-summary node path and
  `batch_summary.montage` at its image-grid node path).

## __init__.py
- Visualizer factories returning `tensor -> PIL.Image | None` callables:
  `heatmap()`, `channel_grid()`, `histogram()`.
- `channel_grid()` renders the FIRST batch element only and draws a `+K more`
  marker (via `_draw_more_marker`) when channels are capped; tiles are
  independently min-max normalized.
- `causal_trace_heatmap()` is the one matplotlib entry point here; it raises
  `ImportError` when matplotlib is missing.
- Shape/normalize helpers: `_to_2d_activation()`, `_to_channel_stack()`,
  `_normalize_2d()`, `_resize_image()`.

## _tensor_display.py
- `show_tensor()` backs `Layer.show` and `Op.show` (imported from
  `data_classes/layer.py` and `data_classes/op.py`).
- `TensorShowMethod` is the closed method vocabulary:
  `"auto" | "heatmap" | "channels" | "rgb" | "hist"`; `_auto_method()` picks one.
- `_prepare_signed_data()` and `_clip_outliers()` are shared with
  `causal_trace_heatmap()`.

## batch_summary.py
- `montage()` (image grid with cap disclosure) and `text_table()` (truncated
  text summary); reused by `visualization/_render_nodes.py` (both call sites
  live there, not in `_render_common.py`).

## feature_maps.py
- `feature_maps()` renders per-layer activation feature-map grids;
  `feature_map_node_spec()` returns a node-spec function for `Trace.draw()`.
- Channel selection/aggregation helpers: `_select_maps()`,
  `_aggregate_channels()`, `_top_channel_indices()`.

## node_plots.py
- PIL-only render primitives: `heatmap()`, `lineplot()`,
  `image_scatter()`.
- Consumed outside the package by `repgeom/__init__.py` and
  `receptive_field/_viz.py`; treat their signatures as shared surface.

## Local Invariants / Gotchas
- `node_plots.py`, `feature_maps.py`, and `batch_summary.py` are PIL-only by
  design; do not add matplotlib imports there. Matplotlib is reached only
  through `causal_trace_heatmap()` (clear `ImportError` without matplotlib)
  and `show_tensor()` — which by DESIGN returns a text message when
  matplotlib is absent so `Layer.show`/`Op.show` keep working on
  matplotlib-less installs; do not "fix" it to raise.
- Hidden-content honesty is load-bearing: capped grids/scatters must keep their
  `+K more` / more-indicator markers so a partial render never reads as
  complete.
- Visualizer callables accept a keyword-only `layer_label` and return `None`
  for incompatible shapes instead of raising.
