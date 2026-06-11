# visualization/ - Graph Rendering and Visual Helpers

## What This Does
Renders TorchLens graph structures and related previews. Graphviz is the primary public
renderer. Long-range-edge graphs can switch to the internal rank layout backend.
Experimental Dagua rendering lives under `torchlens.experimental.dagua` and requires
explicit opt-in.

## Files

| File | Purpose |
|------|---------|
| `rendering.py` | Main Graphviz forward/backward rendering, module focus, skip/collapse, intervention nodes |
| `_rank_layout_internal/layout.py` | Cost estimator and pure-Python rank layout backend for large graphs |
| `code_panel.py` | Source-code side panel helpers for Graphviz |
| `node_spec.py` | Public `NodeSpec`, HTML label rendering, intervention node specs |
| `modes.py` | Node mode presets: default, profiling, vision, attention |
| `themes.py` | Built-in themes and semantic node/edge attrs |
| `overlays.py` | Overlay score normalization, node lines, border attrs |
| `bundle_diff.py` | SVG bundle diff renderer used by demos/tests |
| `fastlog_preview.py` | Predicate preview graph overlay for fastlog |
| `fastlog_live.py` | Live fastlog helper rendering |
| `_render_utils.py` | Shared render helpers |
| `_summary_internal/` | Internal summary builders |

## Entry Points
- `Trace.draw()` and `show_model_graph()` call `rendering.draw()`.
- `show_backward_graph()` calls `rendering.render_backward_graph()`.
- `torchlens.viz.bundle_diff()` calls `visualization.bundle_diff.bundle_diff()`.
- `torchlens.fastlog.preview()` uses `visualization.fastlog_preview.preview_fastlog()`.

## Code Panel
`code_panel=True`, `"forward"`, `"class"`, or `"init+forward"` adds a Graphviz side panel
from source captured at logging time. Callable code panels require a live model.

## Visualization Modes
- `vis_mode="unrolled"` renders per-pass `Op` nodes.
- `vis_mode="rolled"` renders aggregate `Layer` nodes.
- `module=...` focuses a submodule and inserts synthetic boundary nodes.
- `skip_fn` can hide layers while chaining edges through them.
- `collapse_fn` or module-depth options collapse module subgraphs.
- `order_siblings=True` (default) applies a Graphviz `dot` post-pass in forward unrolled
  mode to place true parallel siblings in execution order after local stretch verification.

## Node Customization
`VisualizationOptions.node_spec_fn` receives `(layer_log, default_spec)` and returns a
`NodeSpec` or `None`. `collapsed_node_spec_fn` customizes collapsed module nodes. Node mode
presets are applied before user callbacks, so callbacks win.

Graph, edge, grad-edge, and module styling use dict/callable override options.

## Conditional and Intervention Rendering
Graphviz renders IF/THEN/ELIF/ELSE arm labels from cond-id-aware metadata. Intervention-ready
logs can render hook nodes and cone/site styling through `node_spec.py` helpers.

## Layout Backend
`vis_node_placement="auto"` chooses Graphviz `dot` for local-topology graphs and switches
to `_rank_layout_internal/layout.py` when the estimated cost exceeds 20,000 units. The
cost is `num_nodes + sum(rank_span)` for edges whose topological rank span is greater than
12. Explicit `"dot"` and `"rank"` values force the engine.
